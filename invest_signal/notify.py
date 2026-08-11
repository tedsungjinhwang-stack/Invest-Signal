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


SIGNAL_EMOJI = {"상승초입": "🟢", "눌림목": "🔵",
                "크립토 모멘텀 눌림목/이탈": "⚡", "하락전환": "🔻"}
SIGNAL_ORDER = ["상승초입", "눌림목", "크립토 모멘텀 눌림목/이탈", "하락전환"]
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


def _event_line(e, url: str, name: str, kind: str) -> str:
    d = e.detail
    head = (f"• <a href=\"{url}\">{_short_symbol(e.symbol, kind, name)}</a>"
            f"  <b>{_fmt_price(e.price)}</b>")
    tags = []
    if d.get("intrabar"):
        tags.append("⏳진행봉")     # 봉 미마감 잠정 판정 — 마감 때 되돌릴 수 있음
    if e.signal == "mss" or d.get("label") == "MSS":
        # MSS는 MSS 태그만 — 배열 상태는 붙이지 않는다
        tags.append(f"⚠️MSS 저점 {_fmt_price(d['broken_low'])} 이탈"
                    if d.get("broken_low") else "⚠️MSS")
    else:
        if d.get("stage"):
            # 눌림목 — 타점(밴드 터치)과 대기(밴드 위)를 한눈에 구분
            tags.append("🎯타점" if d["stage"] == "타점" else "대기")
        if e.signal == "downtrend_reversal" and d.get("broken_low"):
            tags.append(f"직전저점 {_fmt_price(d['broken_low'])} 이탈")
        if e.signal == "leader_break":
            # 24h 상승률 순위로 뽑힌 종목이 15m 추세선을 깬 자리
            if d.get("gain_24h") is not None:
                tags.append(f"24h {d['gain_24h'] * 100:+.0f}%")
            tags.append(f"{d.get('interval', '15m')} "
                        f"{d.get('ma_period', 60)}SMA {_fmt_price(d['ma'])} 이탈")
            if d.get("watch_days") is not None:
                # 지금은 상위권 밖 — 등재 후 며칠째 추적 중인지
                tags.append(f"추적 {d['watch_days']}일차")
        if d.get("align"):
            tags.append(d["align"])
    return head + (" · " + " · ".join(tags) if tags else "")


LEADER_LABEL = "크립토 모멘텀 눌림목/이탈"


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
            if d.get("gain_24h") is not None:
                tags.append(f"24h {d['gain_24h'] * 100:+.0f}%")
            return (f"↳ {_short_symbol(e.symbol, kind, name)}"
                    + (f"  {_fmt_price(d['last_price'])}" if d.get("last_price") else "")
                    + " · " + "·".join(tags))
        tags = [f"{_age_days(e.bar_time)}d"]
        if e.signal == "mss" or d.get("label") == "MSS":
            # MSS는 신규 줄과 같은 형식으로 — 깨진 저점 레벨 포함
            tags.append(f"⚠️MSS 저점 {_fmt_price(d['broken_low'])} 이탈"
                        if d.get("broken_low") else "⚠️MSS")
        else:
            if d.get("stage"):
                tags.append("🎯타점" if d["stage"] == "타점" else "대기")
            if d.get("align"):
                tags.append(d["align"])
        price = d.get("last_price")
        return (f"↳ {_short_symbol(e.symbol, kind, name)}"
                + (f"  {_fmt_price(price)}" if price else "")
                + " · " + "·".join(tags))

    for label in ordered:
        lines.append("")
        lines.append(DIVIDER)
        lines.append(f"{SIGNAL_EMOJI.get(label, '▪')} <b>{label}</b>")
        for mtitle, new, hold, kind in markets:
            new_sel = sorted((e for e in new if label_of(e) == label),
                             key=lambda x: x.symbol)
            hold_sel = sorted((e for e in hold if label_of(e) == label),
                              key=lambda x: x.bar_time, reverse=True)   # 최신 순
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
            for e in new_sel:
                name = etf_names.get(e.symbol, "") if kind != "crypto" else ""
                market = "KR" if (kind != "crypto" and e.symbol[:1].isdigit()) else "US"
                lines.append(_event_line(e, chart_url(e.symbol, kind, market), name, kind))
            # 추적 리스트 — 종목마다 한 줄, 현재가 포함. 자르지 않고 전부 보여준다
            # (길어지면 split_chunks가 여러 메시지로 나눠 보낸다).
            lines.extend(hold_line(e, kind) for e in hold_sel)

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
