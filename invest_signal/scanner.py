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


CHOCH_EVICTS = {"pullback"}     # CHoCH 발생 시 리스트에서 걷어낼 셋업 (상승초입은 240선 재이탈로만 제거)

# 크립토 랭크 필터 적용 범위 — signal.<키>.crypto_rank_filter 설정별로 걸러낼
# 시그널 집합. 풀백 칸은 MSS 줄까지 함께 거르고, 상승초입은 자체(완화) 기준을 쓴다.
RANK_FILTER_SCOPE = {
    "pullback": frozenset({"pullback", "mss"}),
    "uptrend_onset": frozenset({"uptrend_onset"}),
}
RANK_FILTER_LABEL = {"pullback": "풀백·MSS", "uptrend_onset": "상승초입"}


def _detect_all(frames: dict, detectors, log=print, show_choch=False) -> tuple[list, list]:
    """전 종목 시그널 검출 + '유지 중' 목록.

    하락전환(downtrend_reversal)은 눌림목 셋업 이후에 발생했으면 그 항목을
    유지 중 리스트에서 영구 제거하는 청소 역할을 한다(상승초입은 240선
    재이탈로만 제거). 크립토(show_choch=False)는 그 용도로만 쓰고 표시하지
    않지만, ETF·주식(show_choch=True)은 하락전환 칸에 알림·추적도 한다.
    각 이벤트에 현재 이평선 배열 상태(역배열→혼조→정배열 전환 추적)를 붙인다.
    """
    events, ongoing = [], []
    for sym, df in frames.items():
        sym_events, sym_ongoing = [], []
        choch_time = None               # 조회 범위 내 가장 최근 하락전환 시각
        for mod, params in detectors:
            wide = dataclasses.replace(params, grace_bars=ONGOING_LOOKBACK_BARS)
            past = mod.detect(df, sym, wide)
            latest = max(past, key=lambda e: e.bar_time) if past else None
            if mod.NAME == "downtrend_reversal":
                if latest is not None:
                    choch_time = latest.bar_time
                if not show_choch:
                    continue
            sym_events.extend(mod.detect(df, sym, params))
            if latest is not None and mod.still_active(df, latest, params):
                sym_ongoing.append(latest)
        if choch_time is not None:      # CHoCH 이후에 만들어진 눌림목만 살아남는다
            sym_ongoing = [e for e in sym_ongoing
                           if e.signal not in CHOCH_EVICTS or e.bar_time > choch_time]
        if sym_events or sym_ongoing:
            align = indicators.alignment(df)
            if align:
                for e in sym_events + sym_ongoing:
                    e.detail["align"] = align
            if len(df):
                last = float(df["Close"].iloc[-1])
                for e in sym_ongoing:       # 추적 리스트에 현재가(최근 4h 종가) 표시용
                    e.detail["last_price"] = last
        events.extend(sym_events)
        ongoing.extend(sym_ongoing)
    return events, ongoing


def _crypto_rank_eligible(frames: dict, rcfg: dict) -> set:
    """풀백 랭크 필터 대상.

    (24h 거래대금 상위 N ∪ 최근 상승률 상위 N) 중에서
    24h 거래대금이 min_turnover_usd 이상인 종목만 남긴다.
    """
    vol_top = int(rcfg.get("volume_top", 100))
    gain_top = int(rcfg.get("gain_top", 50))
    lookback = int(rcfg.get("gain_lookback_bars", 42))
    min_turnover = float(rcfg.get("min_turnover_usd", 0))
    turnover, gain = {}, {}
    for sym, df in frames.items():
        if len(df) >= 6 and "Volume" in df.columns:
            turnover[sym] = float((df["Close"].iloc[-6:] * df["Volume"].iloc[-6:]).sum())
        if len(df) > lookback:
            base = float(df["Close"].iloc[-1 - lookback])
            if base > 0:
                gain[sym] = float(df["Close"].iloc[-1]) / base - 1
    top_vol = set(sorted(turnover, key=turnover.get, reverse=True)[:vol_top])
    top_gain = set(sorted(gain, key=gain.get, reverse=True)[:gain_top])
    eligible = top_vol | top_gain
    if min_turnover > 0:
        eligible = {s for s in eligible if turnover.get(s, 0.0) >= min_turnover}
    return eligible


def _filter_ranked(items: list, eligible: set, signals: frozenset) -> list:
    """대상 시그널은 랭크 필터 통과 종목만 남긴다 (나머지 시그널은 그대로)."""
    return [e for e in items if e.signal not in signals or e.symbol in eligible]


def scan_crypto(cfg: dict, detectors, log=print) -> tuple[list, list]:
    c = cfg.get("crypto") or {}
    if not c.get("enabled", True):
        return [], []
    exclude = set(c.get("exclude") or [])
    limit = int(c.get("kline_limit", 750))
    workers = int(c.get("fetch_workers", 6))
    requested = c.get("source", "auto")
    with requests.Session() as s:
        source, symbols = data_binance.resolve_source(s, requested, exclude, log)
        log(f"[binance] {source} · USDT {len(symbols)}종 스캔 시작 (워커 {workers})")
        frames = data_binance.fetch_all(s, symbols, source, limit=limit, log=log,
                                        workers=workers)
    events, ongoing = _detect_all(frames, detectors, log)

    # 크립토 전용 랭크 필터 — 주도주(거래대금·상승률 상위)만 남긴다.
    # 풀백 칸(풀백+MSS 줄)과 상승초입이 각자 기준을 갖는다 —
    # 상승초입은 3파 전 초입이라 기준을 낮게 잡는다.
    scfg = cfg.get("signal") or {}
    for key, signals in RANK_FILTER_SCOPE.items():
        rcfg = (scfg.get(key) or {}).get("crypto_rank_filter") or {}
        if not rcfg.get("enabled", True):
            continue
        eligible = _crypto_rank_eligible(frames, rcfg)
        before = sum(1 for e in events if e.signal in signals)
        events = _filter_ranked(events, eligible, signals)
        ongoing = _filter_ranked(ongoing, eligible, signals)
        after = sum(1 for e in events if e.signal in signals)
        log(f"[binance] {RANK_FILTER_LABEL[key]} 랭크 필터: "
            f"대상 {len(eligible)}종 — 신규 {before}→{after}건")

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
    token = os.environ.get("QP_GITHUB_TOKEN") or None   # 퍼블릭 레포면 토큰 없이도 OK
    try:
        dyn = data_qp.parse_report(data_qp.fetch_report(token))
        if dyn:
            log(f"[stocks] 퀀트포트폴리오 최신 리포트에서 {len(dyn)}종 로드")
            return dyn
        log("[stocks] 리포트 파싱 결과 0종 — 정적 목록 사용")
    except Exception as e:                          # noqa: BLE001 — 동적 로드 실패는 폴백으로
        hint = "" if token else " (프라이빗 레포면 QP_GITHUB_TOKEN 필요)"
        log(f"[stocks] 리포트 로드 실패({e}){hint} — 정적 목록 사용")
    return static


def _load_theme_etfs(cfg: dict, log=print) -> list[dict]:
    """퀀트포트폴리오 'ETF 모멘텀 TOP10(테마포착)' 보유 ETF — 매 스캔 동적 로드.

    조회 실패는 스캔을 막지 않는다(테마 ETF만 빠진 채 진행).
    """
    e = cfg.get("etf") or {}
    if not e.get("theme_from_qp", True):
        return []
    token = os.environ.get("QP_GITHUB_TOKEN") or None
    try:
        etfs = data_qp.theme_etfs(token)
        if etfs:
            log(f"[etf] 퀀트포트폴리오 테마포착 TOP10에서 {len(etfs)}종 로드")
        return etfs
    except Exception as e:                          # noqa: BLE001 — 동적 로드 실패는 건너뜀
        log(f"[etf] 테마 ETF 로드 실패({e}) — 정적 목록만 사용")
        return []


def scan_yfinance(cfg: dict, detectors, log=print) -> tuple[list, list, dict, dict]:
    """레버리지 ETF + 테마 ETF + 퀀트포트폴리오 주식을 한 번에 수집·검출."""
    tickers, groups = [], {}
    e = cfg.get("etf") or {}
    if e.get("enabled", True):
        for t in e.get("tickers") or []:
            tickers.append(t)
            groups[t["code"]] = "etf"
        for t in _load_theme_etfs(cfg, log):
            if t["code"] not in groups:
                tickers.append(t)
                groups[t["code"]] = "etf"
    for t in _load_stock_tickers(cfg, log):
        if t["code"] not in groups:
            tickers.append(t)
            groups[t["code"]] = "stock"
    if not tickers:
        return [], [], {}, {}
    data_etf.fill_missing_kr_names(tickers, log)    # 리포트가 이름을 빠뜨린 국장 종목 보충
    names = {t["code"]: t.get("name", "") for t in tickers}
    log(f"[yfinance] ETF·주식 {len(tickers)}종 스캔 시작")
    frames = data_etf.fetch_all(tickers, log=log)
    events, ongoing = _detect_all(frames, detectors, log, show_choch=True)
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
