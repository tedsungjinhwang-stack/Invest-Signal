"""텔레그램 알림 발송 및 메시지 포맷."""

import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")
TG_LIMIT = 4096          # 텔레그램 메시지 최대 길이
CHUNK = 3800             # 여유를 둔 분할 기준
ONGOING_MAX_PER_LABEL = 15   # 유지 중 목록 — 라벨당 최대 표시 종목 수


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


SIGNAL_EMOJI = {"상승초입": "🟢", "눌림목": "🔵", "MSS": "🔴"}
SIGNAL_ORDER = ["상승초입", "눌림목", "MSS"]


def _short_symbol(symbol: str, kind: str, name: str) -> str:
    """표시용 심볼 — 크립토는 USDT 접미사 제거, ETF/주식은 이름 병기."""
    if kind == "crypto":
        return symbol[:-4] if symbol.endswith("USDT") else symbol
    return f"{symbol} {name}".strip()


def _age_days(bar_time) -> int:
    t = pd.Timestamp(bar_time)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return max(0, (pd.Timestamp.now(tz="UTC") - t).days)


def _event_line(e, url: str, name: str, kind: str) -> str:
    d = e.detail
    head = f"· <a href=\"{url}\">{_short_symbol(e.symbol, kind, name)}</a> {_fmt_price(e.price)}"
    tags = []
    if e.signal == "mss" and d.get("broken_low"):
        tags.append(f"저점 {_fmt_price(d['broken_low'])} 이탈")
    if d.get("above_qvwap") is not None:
        tags.append("QVWAP↑" if d["above_qvwap"] else "QVWAP↓")
    if d.get("align"):
        tags.append(d["align"])
    return " · ".join([head] + tags)


def format_events(events_crypto: list, events_etf: list,
                  etf_names: dict[str, str],
                  ongoing_crypto: list = (), ongoing_etf: list = (),
                  events_stocks: list = (), ongoing_stocks: list = ()) -> str:
    """이번 스캔의 신규 시그널 + '유지 중' 목록을 텔레그램 HTML 메시지로."""
    now_kst = pd.Timestamp.now(tz=KST).strftime("%m-%d %H:%M")
    lines = [f"🚨 <b>4h 시그널</b> · {now_kst} KST"]

    def by_label(events):
        out = {}
        for e in sorted(events, key=lambda x: x.symbol):
            out.setdefault(e.detail.get("label", e.signal), []).append(e)
        order = [k for k in SIGNAL_ORDER if k in out] + \
                [k for k in out if k not in SIGNAL_ORDER]
        return [(k, out[k]) for k in order]

    def block(title, events, kind):
        if not events:
            return
        lines.append(f"\n<b>━ {title} 신규 ━</b>")
        for label, evs in by_label(events):
            lines.append(f"{SIGNAL_EMOJI.get(label, '▪')} <b>{label}</b>")
            for e in evs:
                name = etf_names.get(e.symbol, "") if kind != "crypto" else ""
                market = "KR" if (kind != "crypto" and e.symbol[:1].isdigit()) else "US"
                lines.append(_event_line(e, chart_url(e.symbol, kind, market), name, kind))

    def hold_block(title, events, kind):
        """트리거 후 조건이 계속 유지 중인 종목들 — 경과일(Nd)로 압축 표기."""
        if not events:
            return
        lines.append(f"\n📌 <b>{title} 유지 중</b>")

        def item(e):
            align = e.detail.get("align")
            sym = _short_symbol(e.symbol, kind, "")
            return f"{sym}({_age_days(e.bar_time)}d" + (f"·{align}" if align else "") + ")"

        for label, evs in by_label(events):
            evs = sorted(evs, key=lambda x: x.bar_time, reverse=True)   # 최신 순
            shown = evs[:ONGOING_MAX_PER_LABEL]
            extra = len(evs) - len(shown)
            lines.append(f"{SIGNAL_EMOJI.get(label, '▪')} {label}: "
                         + " · ".join(item(e) for e in shown)
                         + (f" 외 {extra}종" if extra > 0 else ""))

    block("크립토", events_crypto, "crypto")
    block("ETF", events_etf, "etf")
    block("주식", events_stocks, "etf")      # 링크 규칙은 ETF와 동일(야후/KRX)
    hold_block("크립토", ongoing_crypto, "crypto")
    hold_block("ETF", ongoing_etf, "etf")
    hold_block("주식", ongoing_stocks, "etf")
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
