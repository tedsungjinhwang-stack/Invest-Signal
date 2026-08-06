"""스캔 오케스트레이션 — 데이터 수집 → 시그널 검출 → 중복 필터 → 알림.

종료 코드: 0 = 정상. 1 = 한쪽 스캔 실패 또는 알림 발송 실패/미설정 —
GitHub Actions에서 런이 빨갛게 떠서 문제를 바로 알 수 있게 한다.
발송에 실패한 시그널은 상태에 기록하지 않으므로 다음 4h 스캔에서
(grace_bars 이내면) 다시 시도된다.
"""

import dataclasses
import os

import pandas as pd
import requests

from . import config as cfg_mod
from . import data_binance, data_etf, data_qp, indicators, notify
from .signals import leader_break
from .state import AlertState

ONGOING_LOOKBACK_BARS = 42      # '유지 중' 판정 시 트리거를 찾아볼 범위 — 42봉 = 7일


CHOCH_EVICTS = {"pullback"}     # CHoCH 발생 시 리스트에서 걷어낼 셋업 (상승초입은 240선 재이탈로만 제거)

# 크립토 랭크 필터 적용 범위 — signal.<키>.crypto_rank_filter 설정별로 걸러낼
# 시그널 집합. 눌림목 칸은 진입 게이트를 공유하는 MSS 줄까지 함께 거르고,
# 펌핑·상승초입·펌핑초기는 성격이 달라 각자 기준을 쓴다.
RANK_FILTER_SCOPE = {
    "pullback": frozenset({"pullback", "mss"}),
    "pump_dip": frozenset({"pump_dip"}),
    "uptrend_onset": frozenset({"uptrend_onset"}),
    "pump_early": frozenset({"pump_early"}),
}
RANK_FILTER_LABEL = {"pullback": "눌림목·MSS", "pump_dip": "펌핑",
                     "uptrend_onset": "상승초입", "pump_early": "펌핑초기"}


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
    """랭크 필터 대상 종목 집합.

    24h 거래대금 상위 N위와 최근 상승률 상위권을 각각 구해 mode에 따라
    합집합(or, 기본) 또는 교집합(and)을 취하고, 24h 거래대금이
    min_turnover_usd 이상인 종목만 남긴다.

    상승률 상위권 크기는 gain_top(절대 종목 수) 또는 gain_top_pct(유니버스
    대비 비율)로 정한다 — 비율은 상장 종목 수가 변해도 기준이 흔들리지 않는다.
    mode=and는 "거래량도 상승률도 상위" = 그 시기의 주도주만 남기는 설정.
    """
    vol_top = int(rcfg.get("volume_top", 100))
    gain_top = int(rcfg.get("gain_top", 50))
    gain_top_pct = float(rcfg.get("gain_top_pct", 0) or 0)
    lookback = int(rcfg.get("gain_lookback_bars", 42))
    min_turnover = float(rcfg.get("min_turnover_usd", 0))
    require_both = str(rcfg.get("mode", "or")).lower() == "and"
    turnover, gain = {}, {}
    for sym, df in frames.items():
        if len(df) >= 6 and "Volume" in df.columns:
            turnover[sym] = float((df["Close"].iloc[-6:] * df["Volume"].iloc[-6:]).sum())
        if len(df) > lookback:
            base = float(df["Close"].iloc[-1 - lookback])
            if base > 0:
                gain[sym] = float(df["Close"].iloc[-1]) / base - 1
    if gain_top_pct > 0:
        gain_top = max(1, round(len(gain) * gain_top_pct))
    top_vol = set(sorted(turnover, key=turnover.get, reverse=True)[:vol_top])
    top_gain = set(sorted(gain, key=gain.get, reverse=True)[:gain_top])
    eligible = (top_vol & top_gain) if require_both else (top_vol | top_gain)
    if min_turnover > 0:
        eligible = {s for s in eligible if turnover.get(s, 0.0) >= min_turnover}
    return eligible


def _scan_leader_break(session, source: str, symbols: list, cfg: dict,
                       state=None, log=print) -> list:
    """주도주 이탈 — 24h 상승률 상위 N종을 15m봉으로 따로 판정한다.

    나머지 시그널과 달리 4h 프레임을 쓰지 않는다: 대상 선정은 24hr 티커
    (전 종목 1회 요청)로 하고, 뽑힌 종목만 15m 캔들을 추가로 받는다.
    한 번 상위권에 들었던 종목은 순위에서 밀려도 watch_days 동안 계속
    본다(상태 파일의 leaders에 마지막 등재 시각을 기록).
    실패는 크립토 스캔 전체를 막지 않는다 — 이 시그널만 비운다.
    """
    s = (cfg.get("signal") or {}).get("leader_break") or {}
    if not s.get("enabled", True):
        return []
    params = leader_break.Params(
        top_n=int(s.get("top_n", 10)),
        ma=int(s.get("ma", 60)),
        grace_bars=int(s.get("grace_bars", 4)),
        min_turnover_usd=float(s.get("min_turnover_usd", 1_000_000)),
        watch_days=int(s.get("watch_days", 7)),
        max_watch=int(s.get("max_watch", 60)),
    )
    try:
        ticker = data_binance.ticker_24h(session, source)
    except Exception as e:                      # noqa: BLE001
        log(f"[binance] 24h 티커 조회 실패({data_binance._safe(e)}) — 주도주 이탈 건너뜀")
        return []
    universe = set(symbols)
    top = leader_break.leaders(ticker, universe, params)
    if not top:
        log("[binance] 주도주 이탈: 거래대금 하한을 넘는 종목 없음")
        return []
    recent = {}
    if state is not None:
        state.touch_leaders(sym for sym, _ in top)
        recent = state.recent_leaders(days=params.watch_days)
    watch = leader_break.watch_list(top, recent, universe, params)
    log("[binance] 주도주 이탈 상위 — " + ", ".join(
        f"{sym}({t['change_pct'] * 100:+.0f}%)" for sym, t in top))
    carried = len(watch) - len(top)
    if carried:
        log(f"[binance] 주도주 이탈 추적 유지 {carried}종 "
            f"(상위권 이탈 후 {params.watch_days}일 이내)")
    dropped = sum(1 for sym in recent
                  if sym in universe and sym not in {w for w, _ in watch})
    if dropped:
        log(f"[binance] 주도주 이탈: max_watch={params.max_watch} 초과분 "
            f"{dropped}종은 이번 스캔에서 제외 (최근 등재 순으로 남김)")

    now = pd.Timestamp.now(tz="UTC")
    events = []
    for sym, since in watch:
        try:
            df = data_binance.klines(session, sym, source, leader_break.INTERVAL,
                                     limit=leader_break.KLINE_LIMIT)
        except Exception as e:                  # noqa: BLE001 — 종목별 실패는 건너뛴다
            log(f"[binance] {sym} 15m 수집 실패: {data_binance._safe(e)}")
            continue
        stat = ticker.get(sym) or {}
        for ev in leader_break.detect(df, sym, params):
            ev.detail["gain_24h"] = stat.get("change_pct")
            ev.detail["turnover_24h"] = stat.get("quote_volume")
            if since is not None:       # 지금은 상위권 밖 — 며칠째 추적 중인지
                ev.detail["watch_days"] = max(
                    0, (now - pd.Timestamp(since)).days)
            events.append(ev)
    return events


def _filter_ranked(items: list, eligible: set, signals: frozenset) -> list:
    """대상 시그널은 랭크 필터 통과 종목만 남긴다 (나머지 시그널은 그대로)."""
    return [e for e in items if e.signal not in signals or e.symbol in eligible]


def scan_crypto(cfg: dict, detectors, log=print, intrabar: bool = False,
                state=None) -> tuple[list, list]:
    """intrabar=True: 진행 중인 4h봉을 포함해 인트라바 판정이 안전한
    시그널(펌핑 터치)만 돌린다 — 종가 조건 시그널은 마감 스캔에서만."""
    c = cfg.get("crypto") or {}
    if not c.get("enabled", True):
        return [], []
    scfg = cfg.get("signal") or {}
    # crypto_enabled: false인 시그널은 크립토에서 빼고 ETF·주식에서만 본다
    skipped = [m.NAME for m, _ in detectors
               if not (scfg.get(m.NAME) or {}).get("crypto_enabled", True)]
    if skipped:
        detectors = [(m, p) for m, p in detectors if m.NAME not in skipped]
        log(f"[binance] 크립토 제외 시그널: {', '.join(skipped)}")
    if intrabar:
        detectors = [(m, p) for m, p in detectors
                     if getattr(m, "INTRABAR_OK", False)]
    exclude = set(c.get("exclude") or [])
    limit = int(c.get("kline_limit", 750))
    workers = int(c.get("fetch_workers", 6))
    requested = c.get("source", "auto")
    with requests.Session() as s:
        source, symbols = data_binance.resolve_source(s, requested, exclude, log)
        log(f"[binance] {source} · USDT {len(symbols)}종 스캔 시작 (워커 {workers}"
            + (" · 인트라바" if intrabar else "") + ")")
        # 주도주 이탈은 4h 프레임을 안 쓰므로 마감·인트라바 양쪽에서 매번 돈다
        leader_events = _scan_leader_break(s, source, symbols, cfg, state, log)
        if not detectors:           # 인트라바인데 4h 대상 시그널이 없을 때
            return leader_events, []
        frames = data_binance.fetch_all(s, symbols, source, limit=limit, log=log,
                                        workers=workers, include_live=intrabar)
    events, ongoing = _detect_all(frames, detectors, log)
    if intrabar:
        ongoing = []                        # 인트라바 스캔은 신규 알림만
        # 진행 중인 봉에서 잡힌 이벤트는 '미확정' 표시 — 마감 때 되돌릴 수 있음
        live_open = pd.Timestamp.now(tz="UTC").floor("4h")
        for e in events:
            if e.bar_time == live_open:
                e.detail["intrabar"] = True

    # 크립토 전용 랭크 필터 — 주도주(거래대금·상승률 상위)만 남긴다.
    # 눌림목 칸(눌림목+MSS 줄)과 상승초입이 각자 기준을 갖는다 —
    # 상승초입은 3파 전 초입이라 기준을 낮게 잡는다.
    for key, signals in RANK_FILTER_SCOPE.items():
        rcfg = (scfg.get(key) or {}).get("crypto_rank_filter") or {}
        if not rcfg.get("enabled", True):
            continue
        if not any(e.signal in signals for e in events + ongoing):
            continue        # 해당 시그널이 크립토에서 아예 안 도는 경우
        eligible = _crypto_rank_eligible(frames, rcfg)
        before = sum(1 for e in events if e.signal in signals)
        events = _filter_ranked(events, eligible, signals)
        ongoing = _filter_ranked(ongoing, eligible, signals)
        after = sum(1 for e in events if e.signal in signals)
        log(f"[binance] {RANK_FILTER_LABEL[key]} 랭크 필터: "
            f"대상 {len(eligible)}종 — 신규 {before}→{after}건")

    # 주도주 이탈은 자체 선정(24h 상승률 상위)이라 위 랭크 필터를 타지 않는다
    events.extend(leader_events)
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
    dets = [(m, p) for m, p in detectors
            if not getattr(m, "CRYPTO_ONLY", False)]   # 펌핑 등 크립토 전용 제외
    events, ongoing = _detect_all(frames, dets, log, show_choch=True)
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
        dry_run: bool = False, intrabar: bool = False, log=print) -> int:
    cfg = cfg_mod.load(config_path)
    detectors = cfg_mod.detectors(cfg)
    state = AlertState(state_path)
    errors = []
    if intrabar:
        only = "crypto"                     # 인트라바는 크립토 펌핑 터치 전용

    crypto_events, crypto_ongoing = [], []
    yf_events, yf_ongoing, yf_names, yf_groups = [], [], {}, {}
    if only in (None, "crypto"):
        try:
            crypto_events, crypto_ongoing = scan_crypto(cfg, detectors, log,
                                                        intrabar=intrabar, state=state)
        except Exception as e:                      # noqa: BLE001 — 한쪽 실패가 다른 쪽을 막지 않게
            errors.append(f"crypto: {e}")
            log(f"[binance] 크립토 스캔 실패: {e}")
    if only in (None, "etf"):
        try:
            yf_events, yf_ongoing, yf_names, yf_groups = scan_yfinance(cfg, detectors, log)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"etf: {e}")
            log(f"[etf] ETF·주식 스캔 실패: {e}")

    # 주도주 상위권 등재 이력은 알림 유무와 무관하게 매 스캔 남긴다 —
    # 알림이 나갈 때만 저장하면 조용한 스캔에서 추적 창이 끊긴다.
    if not dry_run:
        state.save()

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
