"""텔레그램 알림 발송 및 메시지 포맷."""

import os
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


def format_events(events_crypto: list, events_etf: list,
                  etf_names: dict[str, str]) -> str:
    """이번 스캔에서 새로 뜬 시그널들을 텔레그램 HTML 메시지로."""
    now_kst = pd.Timestamp.now(tz=KST).strftime("%m-%d %H:%M")
    lines = [f"🚨 <b>상승초입 4h 시그널</b> ({now_kst} KST)"]

    def block(title, events, kind):
        if not events:
            return
        lines.append(f"\n<b>[{title}]</b>")
        for e in sorted(events, key=lambda x: x.symbol):
            name = etf_names.get(e.symbol, "") if kind == "etf" else ""
            market = "KR" if (kind == "etf" and e.symbol[:1].isdigit()) else "US"
            url = chart_url(e.symbol, kind, market)
            label = f"{e.symbol} {name}".strip()
            ma60 = e.detail.get("ma60")
            touch = e.detail.get("touch_time", "")
            above_qv = e.detail.get("above_qvwap")
            qv_tag = "" if above_qv is None else (" · QVWAP↑" if above_qv else " · QVWAP↓")
            lines.append(
                f"· <a href=\"{url}\">{label}</a> — 종가 {_fmt_price(e.price)}"
                + (f" &lt; 60선 {_fmt_price(ma60)}" if ma60 else "")
                + (f" (240터치 {_kst(touch)})" if touch else "")
                + qv_tag)

    block("크립토 USDT-P", events_crypto, "crypto")
    block("레버리지 ETF", events_etf, "etf")
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


def send_telegram(text: str, log=print) -> bool:
    """TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID 환경변수로 발송. 성공 여부 반환."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log("[telegram] 토큰/챗ID 없음 — 발송 건너뜀(메시지는 아래 로그로 출력)")
        log(text)
        return False
    ok = True
    for chunk in split_chunks(text):
        try:
            rsp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": chunk, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
                timeout=15)
            if rsp.status_code != 200:
                log(f"[telegram] 실패 {rsp.status_code}: {rsp.text[:200]}")
                ok = False
        except requests.RequestException as e:
            log(f"[telegram] 예외: {e}")
            ok = False
    return ok
