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


SIGNAL_EMOJI = {"상승초입": "🟢", "풀백": "🔵", "하락전환": "🔻"}
SIGNAL_ORDER = ["상승초입", "풀백", "하락전환"]
DISPLAY_GROUP = {"MSS": "풀백", "눌림목": "풀백"}   # MSS는 풀백 칸에 태그로 표시


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
    if e.signal == "mss" or d.get("label") == "MSS":
        tags.append(f"MSS(저점 {_fmt_price(d['broken_low'])} 이탈)"
                    if d.get("broken_low") else "MSS")
    elif e.signal == "downtrend_reversal" and d.get("broken_low"):
        tags.append(f"직전저점 {_fmt_price(d['broken_low'])} 이탈")
    if d.get("align"):
        tags.append(d["align"])
    return " · ".join([head] + tags)


def format_events(events_crypto: list, events_etf: list,
                  etf_names: dict[str, str],
                  ongoing_crypto: list = (), ongoing_etf: list = (),
                  events_stocks: list = (), ongoing_stocks: list = ()) -> str:
    """텔레그램 메시지 — 시그널별 → 시장별. 신규는 상세 줄, 추적 중 종목은
    같은 칸 아래 ↳ 한 줄로 붙는다."""
    now_kst = pd.Timestamp.now(tz=KST).strftime("%m-%d %H:%M")
    lines = [f"🚨 <b>4h 시그널</b> · {now_kst} KST"]

    markets = [("크립토", events_crypto, ongoing_crypto, "crypto"),
               ("ETF", events_etf, ongoing_etf, "etf"),
               ("주식", events_stocks, ongoing_stocks, "etf")]   # 주식 링크 규칙은 ETF와 동일

    def label_of(e):
        label = e.detail.get("label", e.signal)
        return DISPLAY_GROUP.get(label, label)

    present = {label_of(e) for _, new, hold, _ in markets for e in list(new) + list(hold)}
    ordered = [k for k in SIGNAL_ORDER if k in present] + \
              sorted(k for k in present if k not in SIGNAL_ORDER)

    def hold_item(e, kind):
        align = e.detail.get("align")
        is_mss = e.signal == "mss" or e.detail.get("label") == "MSS"
        return (f"{_short_symbol(e.symbol, kind, '')}({_age_days(e.bar_time)}d"
                + (f"·{align}" if align else "")
                + ("·MSS" if is_mss else "") + ")")

    for label in ordered:
        lines.append(f"\n{SIGNAL_EMOJI.get(label, '▪')} <b>{label}</b>")
        for mtitle, new, hold, kind in markets:
            new_sel = sorted((e for e in new if label_of(e) == label),
                             key=lambda x: x.symbol)
            hold_sel = sorted((e for e in hold if label_of(e) == label),
                              key=lambda x: x.bar_time, reverse=True)   # 최신 순
            if not new_sel and not hold_sel:
                continue
            lines.append(f"[{mtitle}]")
            for e in new_sel:
                name = etf_names.get(e.symbol, "") if kind != "crypto" else ""
                market = "KR" if (kind != "crypto" and e.symbol[:1].isdigit()) else "US"
                lines.append(_event_line(e, chart_url(e.symbol, kind, market), name, kind))
            if hold_sel:
                shown = hold_sel[:ONGOING_MAX_PER_LABEL]
                extra = len(hold_sel) - len(shown)
                lines.append("↳ " + " · ".join(hold_item(e, kind) for e in shown)
                             + (f" 외 {extra}종" if extra > 0 else ""))

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
