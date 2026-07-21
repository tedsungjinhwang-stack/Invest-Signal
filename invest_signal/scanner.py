"""스캔 오케스트레이션 — 데이터 수집 → 시그널 검출 → 중복 필터 → 알림."""

import requests

from . import config as cfg_mod
from . import data_binance, data_etf, notify
from .signals import uptrend_onset
from .state import AlertState


def scan_crypto(cfg: dict, params, log=print) -> list:
    c = cfg.get("crypto") or {}
    if not c.get("enabled", True):
        return []
    exclude = set(c.get("exclude") or [])
    limit = int(c.get("kline_limit", 750))
    requested = c.get("source", "auto")
    try:
        with requests.Session() as s:
            source, symbols = data_binance.resolve_source(s, requested, exclude, log)
            log(f"[binance] {source} · USDT {len(symbols)}종 스캔 시작")
            frames = data_binance.fetch_all(s, symbols, source, limit=limit, log=log)
    except data_binance.GeoBlockedError as e:
        log(f"[binance] {e} — 크립토 스캔 건너뜀 (README의 프록시/해외 실행 참고)")
        return []
    events = []
    for sym, df in frames.items():
        ev = uptrend_onset.detect(df, sym, params)
        if ev:
            events.append(ev)
    log(f"[binance] 시그널 {len(events)}건")
    return events


def scan_etf(cfg: dict, params, log=print) -> tuple[list, dict]:
    e = cfg.get("etf") or {}
    tickers = e.get("tickers") or []
    if not e.get("enabled", True) or not tickers:
        return [], {}
    names = {t["code"]: t.get("name", "") for t in tickers}
    log(f"[etf] 레버리지 ETF {len(tickers)}종 스캔 시작")
    frames = data_etf.fetch_all(tickers, log=log)
    events = []
    for code, df in frames.items():
        ev = uptrend_onset.detect(df, code, params)
        if ev:
            events.append(ev)
    log(f"[etf] 시그널 {len(events)}건")
    return events, names


def run(config_path: str, state_path: str, only: str | None = None,
        dry_run: bool = False, log=print) -> int:
    cfg = cfg_mod.load(config_path)
    params = cfg_mod.uptrend_params(cfg)
    state = AlertState(state_path)

    crypto_events = scan_crypto(cfg, params, log) if only in (None, "crypto") else []
    etf_events, etf_names = scan_etf(cfg, params, log) if only in (None, "etf") else ([], {})

    fresh_crypto = [e for e in crypto_events if state.is_new(e.dedup_key)]
    fresh_etf = [e for e in etf_events if state.is_new(e.dedup_key)]
    skipped = (len(crypto_events) - len(fresh_crypto)) + (len(etf_events) - len(fresh_etf))
    if skipped:
        log(f"[state] 이미 알림 보낸 {skipped}건 제외")

    if not fresh_crypto and not fresh_etf:
        log("새 시그널 없음")
        return 0

    msg = notify.format_events(fresh_crypto, fresh_etf, etf_names)
    if dry_run:
        log("[dry-run] 발송 생략 — 메시지 미리보기:")
        log(msg)
        return 0

    sent = notify.send_telegram(msg, log=log)
    # 발송 성공(또는 토큰 미설정으로 콘솔 출력)한 경우에만 기록해서,
    # 텔레그램 장애 시 다음 실행(grace_bars 이내)에 재시도되게 한다.
    if sent or not _telegram_configured():
        for ev in fresh_crypto + fresh_etf:
            state.mark(ev.dedup_key)
        state.save()
    return 0


def _telegram_configured() -> bool:
    import os
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))
