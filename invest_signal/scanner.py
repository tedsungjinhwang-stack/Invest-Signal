"""스캔 오케스트레이션 — 데이터 수집 → 시그널 검출 → 중복 필터 → 알림.

종료 코드: 0 = 정상. 1 = 한쪽 스캔 실패 또는 알림 발송 실패/미설정 —
GitHub Actions에서 런이 빨갛게 떠서 문제를 바로 알 수 있게 한다.
발송에 실패한 시그널은 상태에 기록하지 않으므로 다음 4h 스캔에서
(grace_bars 이내면) 다시 시도된다.
"""

import dataclasses
import os

import requests

from . import config as cfg_mod
from . import data_binance, data_etf, data_qp, indicators, notify
from .state import AlertState

ONGOING_LOOKBACK_BARS = 42      # '유지 중' 판정 시 트리거를 찾아볼 범위 — 42봉 = 7일


def _detect_all(frames: dict, detectors, log=print) -> tuple[list, list]:
    """전 종목 시그널 검출 + '유지 중'(트리거 후 조건이 계속 참) 목록.

    각 이벤트에 현재 이평선 배열 상태(역배열→혼조→정배열 전환 추적)를 붙인다.
    """
    events, ongoing = [], []
    for sym, df in frames.items():
        sym_events, sym_ongoing = [], []
        for mod, params in detectors:
            sym_events.extend(mod.detect(df, sym, params))
            wide = dataclasses.replace(params, grace_bars=ONGOING_LOOKBACK_BARS)
            past = mod.detect(df, sym, wide)
            if past:
                latest = max(past, key=lambda e: e.bar_time)
                if mod.still_active(df, latest, params):
                    sym_ongoing.append(latest)
        if sym_events or sym_ongoing:
            align = indicators.alignment(df)
            if align:
                for e in sym_events + sym_ongoing:
                    e.detail["align"] = align
        events.extend(sym_events)
        ongoing.extend(sym_ongoing)
    return events, ongoing


def scan_crypto(cfg: dict, detectors, log=print) -> tuple[list, list]:
    c = cfg.get("crypto") or {}
    if not c.get("enabled", True):
        return [], []
    exclude = set(c.get("exclude") or [])
    limit = int(c.get("kline_limit", 750))
    requested = c.get("source", "auto")
    with requests.Session() as s:
        source, symbols = data_binance.resolve_source(s, requested, exclude, log)
        log(f"[binance] {source} · USDT {len(symbols)}종 스캔 시작")
        frames = data_binance.fetch_all(s, symbols, source, limit=limit, log=log)
    events, ongoing = _detect_all(frames, detectors, log)
    log(f"[binance] 시그널 {len(events)}건 · 유지 중 {len(ongoing)}건")
    return events, ongoing


def _load_stock_tickers(cfg: dict, log=print) -> list[dict]:
    """주식 감시 목록 — 기본은 퀀트포트폴리오 최신 리포트에서 동적 로드.

    QP_GITHUB_TOKEN이 없거나 조회/파싱에 실패하면 config의 정적 목록 사용.
    """
    s = cfg.get("stocks") or {}
    if not s.get("enabled", True):
        return []
    static = s.get("tickers") or []
    if s.get("source", "quant_portfolio") != "quant_portfolio":
        return static
    token = os.environ.get("QP_GITHUB_TOKEN")
    if not token:
        log("[stocks] QP_GITHUB_TOKEN 없음 — config 정적 목록 사용")
        return static
    try:
        dyn = data_qp.parse_report(data_qp.fetch_report(token))
        if dyn:
            log(f"[stocks] 퀀트포트폴리오 최신 리포트에서 {len(dyn)}종 로드")
            return dyn
        log("[stocks] 리포트 파싱 결과 0종 — 정적 목록 사용")
    except Exception as e:                          # noqa: BLE001 — 동적 로드 실패는 폴백으로
        log(f"[stocks] 리포트 로드 실패({e}) — 정적 목록 사용")
    return static


def scan_yfinance(cfg: dict, detectors, log=print) -> tuple[list, list, dict, dict]:
    """레버리지 ETF + 퀀트포트폴리오 주식을 한 번에 수집·검출."""
    tickers, groups = [], {}
    e = cfg.get("etf") or {}
    if e.get("enabled", True):
        for t in e.get("tickers") or []:
            tickers.append(t)
            groups[t["code"]] = "etf"
    for t in _load_stock_tickers(cfg, log):
        if t["code"] not in groups:
            tickers.append(t)
            groups[t["code"]] = "stock"
    if not tickers:
        return [], [], {}, {}
    names = {t["code"]: t.get("name", "") for t in tickers}
    log(f"[yfinance] ETF·주식 {len(tickers)}종 스캔 시작")
    frames = data_etf.fetch_all(tickers, log=log)
    events, ongoing = _detect_all(frames, detectors, log)
    log(f"[yfinance] 시그널 {len(events)}건 · 유지 중 {len(ongoing)}건")
    return events, ongoing, names, groups


def _collapse(events: list) -> list:
    """(종목, 시그널)별로 가장 최신 봉의 이벤트만 남긴다."""
    best = {}
    for e in events:
        k = (e.symbol, e.signal)
        if k not in best or e.bar_time > best[k].bar_time:
            best[k] = e
    return list(best.values())


def run(config_path: str, state_path: str, only: str | None = None,
        dry_run: bool = False, log=print) -> int:
    cfg = cfg_mod.load(config_path)
    detectors = cfg_mod.detectors(cfg)
    state = AlertState(state_path)
    errors = []

    crypto_events, crypto_ongoing = [], []
    yf_events, yf_ongoing, yf_names, yf_groups = [], [], {}, {}
    if only in (None, "crypto"):
        try:
            crypto_events, crypto_ongoing = scan_crypto(cfg, detectors, log)
        except Exception as e:                      # noqa: BLE001 — 한쪽 실패가 다른 쪽을 막지 않게
            errors.append(f"crypto: {e}")
            log(f"[binance] 크립토 스캔 실패: {e}")
    if only in (None, "etf"):
        try:
            yf_events, yf_ongoing, yf_names, yf_groups = scan_yfinance(cfg, detectors, log)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"etf: {e}")
            log(f"[etf] ETF·주식 스캔 실패: {e}")

    def grp(ev):
        return yf_groups.get(ev.symbol, "etf")

    fresh_crypto = [e for e in crypto_events if state.is_new(e.dedup_key)]
    fresh_yf = [e for e in yf_events if state.is_new(e.dedup_key)]
    # 같은 종목·같은 시그널이 grace 소급으로 두 봉에서 잡히면 최신 봉만 표시
    # (상태에는 둘 다 기록해 다음 실행에서 재등장하지 않게 한다)
    show_crypto = _collapse(fresh_crypto)
    show_etf = [e for e in _collapse(fresh_yf) if grp(e) == "etf"]
    show_stock = [e for e in _collapse(fresh_yf) if grp(e) == "stock"]
    skipped = (len(crypto_events) - len(fresh_crypto)) + (len(yf_events) - len(fresh_yf))
    if skipped:
        log(f"[state] 이미 알림 보낸 {skipped}건 제외")

    if not fresh_crypto and not fresh_yf:
        log("새 시그널 없음")
        return 1 if errors else 0

    # '유지 중' 목록 — 이번에 새로 알리는 (종목, 시그널)은 신규 섹션에 있으므로 제외
    fresh_keys = {(e.symbol, e.signal) for e in fresh_crypto + fresh_yf}
    hold_crypto = [e for e in crypto_ongoing if (e.symbol, e.signal) not in fresh_keys]
    hold_yf = [e for e in yf_ongoing if (e.symbol, e.signal) not in fresh_keys]
    hold_etf = [e for e in hold_yf if grp(e) == "etf"]
    hold_stock = [e for e in hold_yf if grp(e) == "stock"]

    msg = notify.format_events(show_crypto, show_etf, yf_names, hold_crypto, hold_etf,
                               events_stocks=show_stock, ongoing_stocks=hold_stock)
    if dry_run:
        log("[dry-run] 발송 생략 — 메시지 미리보기:")
        log(msg)
        return 1 if errors else 0

    sent = notify.send_telegram(msg, log=log)
    if sent:
        # 발송이 확인된 경우에만 기록 — 실패/미설정 시그널은 다음 스캔에서 재시도
        for ev in fresh_crypto + fresh_yf:
            state.mark(ev.dedup_key)
        state.save()
    else:
        errors.append("telegram: 발송 실패 또는 토큰/챗ID 미설정 — 상태 미저장, 다음 스캔에서 재시도")
        log("[telegram] 발송 실패/미설정 — 상태를 저장하지 않음(다음 4h 스캔에서 재시도)")
    return 1 if errors else 0
