"""심볼 진단 프로브 — 러너에서 실데이터로 특정 코인의 시그널 판정을 재현.

PROBE_SYMBOLS 환경변수(쉼표 구분)의 각 심볼에 대해 최근 봉들의 OHLC와
월간/분기 VWAP, 일자별 거래량·거래대금, 4h 시그널 재현 결과를 출력한다.
알림이 왜 왔는지/안 왔는지 검증할 때 .github/workflows/debug-probe.yml의
심볼을 바꿔 푸시.

PROBE_DAYS를 주면 그 날짜(UTC, 쉼표 구분)의 일봉 요약을 따로 찍는다 —
샌드박스에서 못 보는 퍼프 전용 종목의 과거 거래대금을 확인할 때 쓴다.
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

with requests.Session() as sess:
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
