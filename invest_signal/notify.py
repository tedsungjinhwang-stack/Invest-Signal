"""텔레그램 알림 발송 및 메시지 포맷."""

import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")
TG_LIMIT = 4096          # 텔레그램 메시지 최대 길이
CHUNK = 3800             # 여유를 둔 분할 기준


def _fmt_price(v: float) -> str:
    """가격 자릿수 — 코인(0.0001달러대)부터 ETF(수백달러)까지 커버."""
    if v >= 1000:
        return f"{v:,.1f}"
    if v >= 1:
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return f"{v:.6g}"


def _kst(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(KST).strftime("%m-%d %H:%M")


def chart_url(symbol: str, kind: str, market: str = "US") -> str:
    if kind == "crypto":
        return f"https://www.binance.com/en/futures/{symbol}"
    if market == "KR":
        return f"https://www.tradingview.com/symbols/KRX-{symbol}/"
    return f"https://www.tradingview.com/symbols/{symbol}/"


SIGNAL_EMOJI = {"상승초입": "🟢", "눌림목": "🔵", "펌핑초기": "🌱", "파동": "🌊",
                "크립토 모멘텀 눌림목/이탈": "⚡", "하락전환": "🔻"}
SIGNAL_ORDER = ["상승초입", "눌림목", "펌핑초기", "파동",
                "크립토 모멘텀 눌림목/이탈", "하락전환"]
PULLBACK_STAGES = ("타점", "대기")   # 그 외 stage는 시그널별로 따로 표기한다
DISPLAY_GROUP = {"MSS": "눌림목", "풀백": "눌림목"}   # MSS는 눌림목 칸에 태그로 표시
MARKET_EMOJI = {"크립토": "🪙", "ETF": "📊", "주식": "🏛"}
DIVIDER = "─" * 16


def _short_symbol(symbol: str, kind: str, name: str) -> str:
    """표시용 심볼 — 크립토는 USDT 접미사 제거, 미국은 티커+이름 병기,
    한국(숫자 코드)은 코드 대신 종목명만."""
    if kind == "crypto":
        return symbol[:-4] if symbol.endswith("USDT") else symbol
    if symbol[:1].isdigit() and name:
        return name
    return f"{symbol} {name}".strip()


def _age_days(bar_time) -> int:
    t = pd.Timestamp(bar_time)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return max(0, (pd.Timestamp.now(tz="UTC") - t).days)


def _pct(x: float) -> str:
    """수익률 표기 — 10% 미만은 소수 첫째 자리까지, 그 이상은 정수."""
    v = x * 100
    return f"{v:+.1f}%" if abs(v) < 10 else f"{v:+.0f}%"


# 신규 줄에 구간별 수익률을 붙일 시그널. 어떤 구간이 나오는지는 시그널이
# 채워 준 키에 달렸다 — 파동만 4h·24h·7d 셋을 다 준다.
RETURN_SIGNALS = {"uptrend_onset", "leader_break", "wave_setup"}


def _daily_gain(d: dict) -> float | None:
    """24h 수익률 — 바이낸스 티커 값 우선, 없으면 캔들 역산."""
    g = d.get("gain_24h")
    return d.get("ret_24h") if g is None else g


WAVE_VARIANTS = ("ABC", "임펄스")   # 파동 칸 안에서 이 순서로 묶는다


def _variant(e) -> int:
    """같은 칸 안에서 먼저 갈라 놓을 하위 종류. 파동의 ABC/임펄스가 유일하다.

    한 칸에 두 변형이 섞이면 4h 장기선 터치와 일봉 단기선 터치가 번갈아
    나와 읽기 어렵다 —
    정렬 키의 맨 앞에 둬서 블록으로 묶는다.
    """
    if e.signal != "wave_setup":
        return 0
    stage = e.detail.get("stage")
    return WAVE_VARIANTS.index(stage) if stage in WAVE_VARIANTS else len(WAVE_VARIANTS)


def _by_gain_desc(e):
    """24h 수익률 내림차순 정렬 조각 — 못 구한 항목은 뒤로.

    거래대금은 이미 랭크 필터의 하드 하한(min_turnover_usd)을 통과한 종목만
    남아 있어 다시 줄 세워도 새 정보가 없다. 반면 24h 수익률은 or 모드에서
    거래대금으로 통과한 종목은 보지도 않으므로, 화면에서 처음 드러나는
    정보이고 "이 셋업이 지금 먹히는가"에 직접 답한다.
    """
    g = _daily_gain(e.detail)
    return (1, 0.0) if g is None else (0, -g)


def _quiet_first(e):
    """🍃조용을 칸 맨 위로 올리는 정렬 조각.

    ⚡ 알림은 16%쯤만 조용한 종목인데 상승률 순으로만 세우면 목록 여기저기
    흩어져, 폰에서 스크롤하며 이모지를 찾아야 한다. 이 축의 성과 차이가
    가장 크므로(leader_break.quiet 참고) 먼저 묶어 보여준다.

    quiet 키는 ⚡만 채운다 — 다른 시그널은 전부 같은 값이라 순서가 그대로다.
    """
    return 0 if e.detail.get("quiet") else 1


def _new_order(e):
    """신규 줄 — 변형·🍃로 묶고, 그 안에서 24h 수익률 순, 없으면 심볼 순."""
    return (_variant(e), _quiet_first(e), *_by_gain_desc(e), e.symbol)


def _hold_order(e):
    """추적 줄 — 변형·🍃로 묶고, 24h 수익률 순, 동률이면 최신 발생 순."""
    return (_variant(e), _quiet_first(e), *_by_gain_desc(e), -e.bar_time.timestamp())


def _returns_tag(d: dict) -> str | None:
    """1h·4h·24h·7d 종목 수익률 — 구할 수 있는 것만 이어 붙인다.

    구간은 시그널마다 다르다. 7d를 채우는 건 파동뿐이고, 1h는 상승초입·
    크립토 모멘텀만 받는다 — 없는 키는 그냥 빠진다.

    24h는 바이낸스 24hr 티커 값(gain_24h)이 있으면 그쪽을 쓴다. 캔들 역산은
    마지막 마감봉 기준이라 최대 4시간 묵는데, 티커는 호출 시점 롤링이다.
    """
    parts = []
    for label, key in (("1h", "ret_1h"), ("4h", "ret_4h")):
        v = d.get(key)
        if v is not None:
            parts.append(f"{label} {_pct(v)}")
    day = _daily_gain(d)
    if day is not None:
        parts.append(f"24h {_pct(day)}")
    if d.get("ret_7d") is not None:
        parts.append(f"7d {_pct(d['ret_7d'])}")
    return " · ".join(parts) if parts else None


# 소제목이 수익률 세 칸의 순서까지 말해 준다 — 줄에서는 숫자만 이어 붙여
# 폭을 줄인다. 파동 줄은 라벨을 반복하면 62칸이 되어 폰에서 두 줄로 접힌다
# (다른 칸은 41칸이라 한 줄에 들어간다).
WAVE_HEADERS = {"ABC": "  ⓐ <b>ABC</b> · 4h 추세선 터치 · 4h/24h/7d",
                "임펄스": "  ⓑ <b>임펄스</b> · 일봉 터치 · 4h/24h/7d"}


def _returns_compact(d: dict) -> str | None:
    """4h·24h·7d를 라벨 없이 한 덩어리로 — `-0.3/+2.3/-6.7%`.

    구간 이름은 블록 소제목에 한 번만 적으므로 자리만 지키면 된다.
    못 구한 구간은 건너뛰지 않고 –로 채운다 — 빼면 남은 숫자가 어느
    구간인지 알 수 없게 된다.
    """
    vals = (d.get("ret_4h"), _daily_gain(d), d.get("ret_7d"))
    if all(v is None for v in vals):
        return None
    return "/".join("–" if v is None else _pct(v)[:-1] for v in vals) + "%"


def _wave_tag(d: dict) -> str:
    """파동 — 어느 변형인지와, 무슨 선에 닿았는지.

    둘 다 '터치'라서 변형 이름만으로는 어느 선인지 알 수 없다. 수퍼트렌드선
    값은 싣지 않고 어느 선인지만 적는다 — ABC는 장기선(저항), 임펄스는
    단기선이다. 봉 주기(4h/일봉)는 적지 않는다: 줄에 붙는 수익률의 `4h`와
    글자가 겹쳐 서로 다른 뜻으로 두 번 나온다.
    """
    stage = d.get("stage", "")
    line = d.get("touched") or ("장기선" if stage == "ABC" else "단기선")
    return f"{stage} {line} 터치"


def _event_line(e, url: str, name: str, kind: str) -> str:
    d = e.detail
    head = (f"• <a href=\"{url}\">{_short_symbol(e.symbol, kind, name)}</a>"
            f"  <b>{_fmt_price(e.price)}</b>")
    tags = []
    if d.get("intrabar"):
        tags.append("⏳진행봉")     # 봉 미마감 잠정 판정 — 마감 때 되돌릴 수 있음
    if d.get("quiet"):
        tags.append(QUIET_TAG)      # 거래대금·변동성이 작은 종목 (leader_break.quiet)
    if _early(e):
        tags.append(EARLY_TAG)
    if e.signal in RETURN_SIGNALS:
        rt = _returns_tag(d)
        if rt:
            tags.append(rt)
    if e.signal == "mss" or d.get("label") == "MSS":
        # MSS는 MSS 태그만 — 배열 상태는 붙이지 않는다
        tags.append(f"⚠️MSS 저점 {_fmt_price(d['broken_low'])} 이탈"
                    if d.get("broken_low") else "⚠️MSS")
    else:
        if d.get("stage") in PULLBACK_STAGES:
            # 눌림목 — 타점(밴드 터치)과 대기(밴드 위)를 한눈에 구분
            tags.append("🎯타점" if d["stage"] == "타점" else "대기")
        if e.signal == "wave_setup":
            tags.append(_wave_tag(d))
        if e.signal == "pump_early" and d.get("rise") is not None:
            hours = int(d.get("rise_bars", 1)) * 4
            tags.append(f"{hours}h +{d['rise'] * 100:.1f}%"
                        + (f" · 저점대비 +{d['total_gain'] * 100:.0f}%"
                           if d.get("total_gain") is not None else ""))
        if e.signal == "downtrend_reversal" and d.get("broken_low"):
            tags.append(f"직전저점 {_fmt_price(d['broken_low'])} 이탈")
        if e.signal == "leader_break":
            # 24h 상승률 순위로 뽑힌 종목이 15m 추세선을 깬 자리
            # (24h 상승률은 위 수익률 태그에 이미 들어간다)
            tags.append(f"{d.get('interval', '15m')} "
                        f"{d.get('ma_period', 60)}SMA {_fmt_price(d['ma'])} 이탈")
            if d.get("watch_days") is not None:
                # 지금은 상위권 밖 — 등재 후 며칠째 추적 중인지
                tags.append(f"추적 {d['watch_days']}일차")
        if d.get("align") and not _early(e):
            tags.append(d["align"])
    return head + (" · " + " · ".join(tags) if tags else "")


LEADER_LABEL = "크립토 모멘텀 눌림목/이탈"

# ⚡ 줄에만 붙는 표시. 거래대금·변동성이 작아 위아래로 덜 흔들리는 종목이라는
# 뜻이고, 거르지는 않는다 — 왜 이 축을 표시하는지는 leader_break.quiet 참고.
# 줄 앞쪽(수익률보다 먼저)에 둔다: 뒤에 붙이면 모바일에서 줄바꿈에 묻힌다.
QUIET_TAG = "🍃조용"

# ⚡ 역배열 줄에 붙는 표시. 정배열이 '상승 추세 안의 눌림'이라면 역배열은
# '하락 추세에서 막 돌아서는 자리'라 성격이 정반대인데, 두 종류가 한 칸에
# 섞여 나오므로 눈에 띄는 마크로 가른다. 정배열은 뒤쪽 배열 태그 그대로.
# (🌱는 꺼져 있는 🌱펌핑초기 칸의 이모지이기도 하다 — 그 시그널을 되살리면
#  둘 중 하나를 바꿔야 한다.)
EARLY_TAG = "🌱상승초기"


def _early(e) -> bool:
    """⚡의 역배열 줄인지 — 이 표시가 붙으면 뒤쪽 배열 태그는 생략한다."""
    return e.signal == "leader_break" and e.detail.get("align") == "역배열"


def _board_lines(board: list) -> list[str]:
    """24h 상승률 순위표 — 조건과 무관하게 상위 몇 종을 그대로 적는다."""
    out = ["📈 <b>24h 상승률 TOP</b>"]
    for b in board:
        gain = b.get("gain_24h")
        price = b.get("price")
        out.append(f"{b['rank']}. <a href=\"{chart_url(b['symbol'], 'crypto')}\">"
                   f"{_short_symbol(b['symbol'], 'crypto', '')}</a>"
                   + (f"  {_fmt_price(price)}" if price else "")
                   + (f" · {gain * 100:+.0f}%" if gain is not None else ""))
    return out


def format_events(events_crypto: list, events_etf: list,
                  etf_names: dict[str, str],
                  ongoing_crypto: list = (), ongoing_etf: list = (),
                  events_stocks: list = (), ongoing_stocks: list = (),
                  crypto_board: list = ()) -> str:
    """텔레그램 메시지 — 시그널별 → 시장별. 신규는 상세 줄, 추적 중 종목은
    같은 칸 아래 ↳ 한 줄로 붙는다.

    crypto_board가 있으면 ⚡칸 맨 위에 24h 상승률 순위표를 먼저 싣는다 —
    이탈·제외 조건과 무관한 '지금 뭐가 오르고 있나' 목록이다.
    """
    now_kst = pd.Timestamp.now(tz=KST).strftime("%m-%d %H:%M")
    lines = [f"🚨 <b>4h 시그널</b> · {now_kst} KST"]

    markets = [("크립토", events_crypto, ongoing_crypto, "crypto"),
               ("ETF", events_etf, ongoing_etf, "etf"),
               ("주식", events_stocks, ongoing_stocks, "etf")]   # 주식 링크 규칙은 ETF와 동일

    def label_of(e):
        label = e.detail.get("label", e.signal)
        return DISPLAY_GROUP.get(label, label)

    present = {label_of(e) for _, new, hold, _ in markets for e in list(new) + list(hold)}
    if crypto_board:            # 순위표만 있고 이벤트가 없어도 칸은 만든다
        present.add(LEADER_LABEL)
    ordered = [k for k in SIGNAL_ORDER if k in present] + \
              sorted(k for k in present if k not in SIGNAL_ORDER)

    def hold_line(e, kind):
        d = e.detail
        name = etf_names.get(e.symbol, "") if kind != "crypto" else ""
        if e.signal == "leader_break":
            # 감시 창 안이면 선 위로 복귀해도 남는다 — 순위·선 위아래를 같이 보여준다
            ma = d.get("ma_period", 60)
            tags = [f"{d['rank']}위" if d.get("rank")
                    else f"추적 {d.get('watch_days', 0)}일차",
                    f"{ma}선 위" if d.get("above_ma") else f"🔻{ma}선 아래"]
            if _early(e):
                tags.insert(0, EARLY_TAG)
            if d.get("quiet"):
                tags.insert(0, QUIET_TAG)
            day = _daily_gain(d)
            if day is not None:
                tags.append(f"24h {_pct(day)}")
            return (f"↳ {_short_symbol(e.symbol, kind, name)}"
                    + (f"  {_fmt_price(d['last_price'])}" if d.get("last_price") else "")
                    + " · " + " · ".join(tags))
        tags = [f"{_age_days(e.bar_time)}d"]
        if d.get("quiet"):
            tags.append(QUIET_TAG)      # ⚡·파동 공통 — 조용한 종목 표시
        if e.signal == "wave_setup":
            # 파동만 구간 세트를 싣는다 — refresh_detail이 매 스캔 갱신하므로
            # 세 값 모두 지금 값이다. 다른 시그널의 ret_4h는 트리거 봉에
            # 얼어붙은 값이라 추적 줄에 실으면 묵은 숫자가 나간다.
            rt = _returns_compact(d)
            if rt:
                tags.append(rt)
            if d.get("touched"):
                # ⓐ는 장기선·단기선 두 자리를 다 잡으므로 소제목만으론
                # 구분이 안 된다. 세 글자라 한 줄 폭에 들어간다.
                tags.append(d["touched"])
        else:
            day = _daily_gain(d)
            if day is not None:
                tags.append(f"24h {_pct(day)}")   # 정렬 기준을 줄에도 드러낸다
        if e.signal == "mss" or d.get("label") == "MSS":
            # MSS는 신규 줄과 같은 형식으로 — 깨진 저점 레벨 포함
            tags.append(f"⚠️MSS 저점 {_fmt_price(d['broken_low'])} 이탈"
                        if d.get("broken_low") else "⚠️MSS")
        else:
            if d.get("stage") in PULLBACK_STAGES:
                tags.append("🎯타점" if d["stage"] == "타점" else "대기")
            # 파동은 배열 태그를 뺀다 — 판정이 수퍼트렌드라 이평선 배열은
            # 조건에 안 들어가고, 한글 두 글자가 줄 폭에서 제일 비싸다
            if d.get("align") and e.signal != "wave_setup":
                tags.append(d["align"])
        price = d.get("last_price")
        return (f"↳ {_short_symbol(e.symbol, kind, name)}"
                + (f"  {_fmt_price(price)}" if price else "")
                + " · " + " · ".join(tags))

    for label in ordered:
        lines.append("")
        lines.append(DIVIDER)
        lines.append(f"{SIGNAL_EMOJI.get(label, '▪')} <b>{label}</b>")
        for mtitle, new, hold, kind in markets:
            new_sel = sorted((e for e in new if label_of(e) == label),
                             key=_new_order)
            hold_sel = sorted((e for e in hold if label_of(e) == label),
                              key=_hold_order)
            board = crypto_board if (label == LEADER_LABEL and kind == "crypto") else ()
            if not new_sel and not hold_sel and not board:
                continue
            if board:
                lines.append("")
                lines.extend(_board_lines(board))
            if not new_sel and not hold_sel:
                continue
            lines.append("")
            lines.append(f"{MARKET_EMOJI.get(mtitle, '▪')} <b>{mtitle}</b>")
            variant = None      # 파동 추적 블록의 변형 소제목 추적용
            for e in new_sel:
                name = etf_names.get(e.symbol, "") if kind != "crypto" else ""
                market = "KR" if (kind != "crypto" and e.symbol[:1].isdigit()) else "US"
                lines.append(_event_line(e, chart_url(e.symbol, kind, market), name, kind))
            # 추적 리스트 — 종목마다 한 줄, 현재가 포함. 자르지 않고 전부 보여준다
            # (길어지면 split_chunks가 여러 메시지로 나눠 보낸다).
            # 파동은 변형이 바뀌는 자리에 소제목을 넣는다 — 이미 변형별로
            # 묶여 있으므로 줄마다 변형 이름을 반복할 이유가 없고, 서른 줄
            # 내내 같은 말이 붙으면 정작 다른 값이 눈에 안 들어온다.
            for e in hold_sel:
                stage = e.detail.get("stage") if e.signal == "wave_setup" else None
                if stage is not None and stage != variant:
                    lines.append(WAVE_HEADERS.get(stage, f"  {stage}"))
                    variant = stage
                lines.append(hold_line(e, kind))

    return "\n".join(lines)


def split_chunks(text: str, size: int = CHUNK) -> list[str]:
    """텔레그램 길이 제한에 맞춰 줄 단위로 분할."""
    if len(text) <= size:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > size:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def _post_chunk(token: str, chat: str, chunk: str, log) -> bool:
    """청크 1개 발송 — 429는 retry_after만큼 쉬고 최대 3회 재시도."""
    for attempt in range(3):
        try:
            rsp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": chunk, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
                timeout=15)
        except requests.RequestException as e:
            log(f"[telegram] 예외(시도 {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
            continue
        if rsp.status_code == 200:
            return True
        if rsp.status_code == 429:
            try:
                wait = int(rsp.json().get("parameters", {}).get("retry_after", 5))
            except Exception:   # noqa: BLE001
                wait = 5
            time.sleep(min(wait, 60))
            continue
        log(f"[telegram] 실패 {rsp.status_code}: {rsp.text[:200]}")
        return False
    return False


def send_telegram(text: str, log=print) -> bool:
    """TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID 환경변수로 발송. 전체 성공 여부 반환.

    미설정이면 메시지를 로그로만 출력하고 False — 호출 측이 상태를 저장하지
    않아 다음 스캔에서 재시도된다.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log("[telegram] 토큰/챗ID 없음 — 발송 불가(메시지는 아래 로그로 출력)")
        log(text)
        return False
    ok = True
    for i, chunk in enumerate(split_chunks(text)):
        if i:
            time.sleep(1.1)     # 텔레그램 초당 1건 제한 회피
        ok = _post_chunk(token, chat, chunk, log) and ok
    return ok
