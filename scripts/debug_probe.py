"""심볼 진단 프로브 — 러너에서 실데이터로 특정 코인의 시그널 판정을 재현.

PROBE_SYMBOLS 환경변수(쉼표 구분)의 각 심볼에 대해 최근 봉들의 OHLC와
월간/분기 VWAP, 일자별 거래량·거래대금, 4h 시그널 재현 결과를 출력한다.
알림이 왜 왔는지/안 왔는지 검증할 때 .github/workflows/debug-probe.yml의
심볼을 바꿔 푸시.

PROBE_DAYS를 주면 그 날짜(UTC, 쉼표 구분)의 일봉 요약을 따로 찍는다 —
샌드박스에서 못 보는 퍼프 전용 종목의 과거 거래대금을 확인할 때 쓴다.

PROBE_MODE=turnover면 심볼 진단 대신 **거래대금 하한 스윕**을 돈다:
유니버스 전체를 받아 하한별 통과 종목 수를 세고, 최근 발송된 알림
(state/sent_log.jsonl)의 종목이 각 하한에서 살아남는지 하나씩 찍는다.
샌드박스는 현물 미러로 폴백해 퍼프보다 거래대금이 훨씬 작게 찍히므로,
하한을 조정할 때는 반드시 여기(러너, 프록시 경유 퍼프)서 재야 한다.
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from invest_signal import data_binance, indicators
from invest_signal.signals import pullback, pump_early, uptrend_onset

PROBES = (uptrend_onset, pullback, pump_early)   # 4h 종가 시그널 — 모듈을 늘리면 그대로 재현된다

symbols = [s.strip().upper() for s in
           os.environ.get("PROBE_SYMBOLS", "BTCUSDT").split(",") if s.strip()]

FLOORS = [float(x) * 1e6 for x in
          os.environ.get("PROBE_FLOORS", "1,2,3,5,10").split(",") if x.strip()]
RECENT_MSGS = int(os.environ.get("PROBE_RECENT_MSGS", "6"))
SECTION_KEY = {"상승초입": "uptrend_onset", "펌핑초기": "pump_early",
               "파동": "wave_setup", "크립토 모멘텀 눌림목/이탈": "leader_break"}


def _recent_alert_symbols(limit: int) -> dict:
    """최근 발송 메시지에서 칸별 등장 종목 — {시그널키: {심볼}}."""
    import json
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "state", "sent_log.jsonl")
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except OSError:
        return {}
    out, sec = {}, None
    for r in rows[-limit:]:
        for ln in r.get("text", "").splitlines():
            m = re.match(r"^([🟢🔵🌱🌊⚡🔻]) <b>(.+?)</b>", ln)
            if m:
                sec = SECTION_KEY.get(m.group(2))
                continue
            if sec is None or not ln.startswith(("• ", "↳ ")):
                continue
            plain = re.sub(r"<[^>]+>", "", ln)
            parts = plain.split()
            if len(parts) > 1:
                out.setdefault(sec, set()).add(parts[1] + "USDT")
    return out


def _turnover_sweep(sess) -> None:
    """유니버스 전체를 받아 하한별 통과 종목 수와 최근 알림 생존 여부를 찍는다."""
    from invest_signal import config as cfg_mod
    from invest_signal import scanner

    cfg = cfg_mod.load()
    c = cfg.get("crypto") or {}
    source, syms = data_binance.resolve_source(sess, c.get("source", "auto"),
                                               set(c.get("exclude") or []), print)
    frames = data_binance.fetch_all(sess, syms, source, limit=750, log=print,
                                    workers=int(c.get("fetch_workers", 6)))
    ticker = {}
    try:
        ticker = data_binance.ticker_24h(sess, source)
    except Exception as e:                          # noqa: BLE001
        print(f"  24h 티커 조회 실패({e}) — 크립토 모멘텀은 건너뜀")

    scfg = cfg.get("signal") or {}
    keys = ["uptrend_onset", "pump_early", "wave_setup"]
    print(f"\n== 거래대금 하한 스윕 · {source} · {len(frames)}종\n")
    header = "  하한  " + "".join(f"{k:>14}" for k in keys) + f"{'모멘텀후보':>12}"
    print(header)
    eligible = {}
    for floor in FLOORS:
        cells = []
        for k in keys:
            rcfg = dict((scfg.get(k) or {}).get("crypto_rank_filter") or {})
            rcfg["min_turnover_usd"] = floor
            e = scanner._crypto_rank_eligible(frames, rcfg)
            eligible[(k, floor)] = e
            cells.append(f"{len(e):>14}")
        lb = sum(1 for s in syms
                 if (ticker.get(s) or {}).get("quote_volume", 0) >= floor)
        eligible[("leader_break", floor)] = {
            s for s in syms if (ticker.get(s) or {}).get("quote_volume", 0) >= floor}
        print(f"  ${floor / 1e6:>4.0f}M" + "".join(cells) + f"{lb:>12}")

    recent = _recent_alert_symbols(RECENT_MSGS)
    if not recent:
        print("\n  (state/sent_log.jsonl 없음 — 최근 알림 대조 생략)")
        return
    print(f"\n== 최근 {RECENT_MSGS}개 메시지에 등장한 종목의 생존 여부\n")
    for key in keys + ["leader_break"]:
        got = sorted(recent.get(key, ()))
        if not got:
            continue
        print(f"  [{key}] 등장 {len(got)}종")
        for floor in FLOORS:
            alive = [s for s in got if s in eligible[(key, floor)]]
            cut = [s for s in got if s not in eligible[(key, floor)]]
            note = ", ".join(x.replace("USDT", "") for x in cut[:12])
            if len(cut) > 12:
                note += f" 외 {len(cut) - 12}"
            print(f"    ${floor / 1e6:>4.0f}M → {len(alive):>3}종 생존"
                  + (f" · 컷: {note}" if cut else ""))
        print()

with requests.Session() as sess:
    if os.environ.get("PROBE_MODE", "").lower() == "turnover":
        _turnover_sweep(sess)
        raise SystemExit(0)
    for sym in symbols:
        try:
            df = data_binance.klines_4h(sess, sym, "fapi", limit=750,
                                        include_live=True)
        except Exception as e:                      # noqa: BLE001
            print(f"== {sym}: 수집 실패 {e}")
            continue
        mv = indicators.monthly_vwap(df)
        qv = indicators.quarterly_vwap(df)
        m120 = indicators.sma(df["Close"], 120)
        m240 = indicators.sma(df["Close"], 240)
        m480 = indicators.sma(df["Close"], 480)
        print(f"== {sym} — 최근 10봉 (봉 오픈시각 KST, 마지막 줄은 진행봉)")
        for i in range(max(0, len(df) - 10), len(df)):
            t = df.index[i].tz_convert("Asia/Seoul").strftime("%m-%d %H시")
            touch = bool(df["Low"].iloc[i] <= mv.iloc[i] <= df["High"].iloc[i])
            c = df["Close"].iloc[i]
            print(f"  {t}: O {df['Open'].iloc[i]:.6g} H {df['High'].iloc[i]:.6g}"
                  f" L {df['Low'].iloc[i]:.6g} C {c:.6g}"
                  f" | MVWAP {mv.iloc[i]:.6g} 터치={touch}"
                  f" | QVWAP {qv.iloc[i]:.6g} {'위' if c > qv.iloc[i] else '아래'}"
                  f" | 120 {m120.iloc[i]:.6g} 240 {m240.iloc[i]:.6g}"
                  f" 480 {m480.iloc[i]:.6g}")
        # 일자별 거래량·거래대금 — 4h봉을 UTC 날짜로 묶는다.
        # 거래대금은 봉마다 종가×거래량을 더한 근사치다(정확한 quote volume은
        # kline의 별도 필드지만 이 프레임은 OHLCV만 들고 있다).
        daily = df.resample("1D").agg({"Open": "first", "High": "max",
                                       "Low": "min", "Close": "last",
                                       "Volume": "sum"}).dropna()
        turnover = (df["Close"] * df["Volume"]).resample("1D").sum()
        want = [d.strip() for d in os.environ.get("PROBE_DAYS", "").split(",") if d.strip()]
        rows = ([daily.loc[[d]] for d in want if d in daily.index.strftime("%Y-%m-%d")]
                if want else [daily.tail(10)])
        print(f"  -- 일봉 요약 (UTC{'  · 지정일' if want else ' · 최근 10일'})")
        for chunk in rows:
            for t, r in chunk.iterrows():
                key = t.strftime("%Y-%m-%d")
                print(f"     {key}  O {r.Open:.6g} H {r.High:.6g} L {r.Low:.6g}"
                      f" C {r.Close:.6g}  거래량 {r.Volume:,.0f}"
                      f"  거래대금 ${turnover.loc[t] / 1e6:,.2f}M")
        for d in want:
            if d not in daily.index.strftime("%Y-%m-%d"):
                print(f"     {d}: 프레임 범위 밖 (750봉 = 약 125일)")

        closed = df.iloc[:-1]                       # 시그널 재현은 마감봉 기준
        for mod in PROBES:
            evs = mod.detect(closed, sym,
                             dataclasses.replace(mod.Params(), grace_bars=12))
            if not evs:
                print(f"  {mod.NAME}: 최근 12봉 내 발화 없음")
            for e in evs:
                kst = e.bar_time.tz_convert("Asia/Seoul").strftime("%m-%d %H시")
                print(f"  {mod.NAME} 발화: {kst} KST · {e.detail}")
