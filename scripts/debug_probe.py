"""심볼 진단 프로브 — 러너에서 실데이터로 특정 코인의 시그널 판정을 재현.

PROBE_SYMBOLS 환경변수(쉼표 구분)의 각 심볼에 대해 최근 봉들의 OHLC와
월간/분기 VWAP, 4h 시그널 재현 결과를 출력한다. 알림이 왜 왔는지/안
왔는지 검증할 때 .github/workflows/debug-probe.yml의 심볼을 바꿔 푸시.
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
        closed = df.iloc[:-1]                       # 시그널 재현은 마감봉 기준
        for mod in PROBES:
            evs = mod.detect(closed, sym,
                             dataclasses.replace(mod.Params(), grace_bars=12))
            if not evs:
                print(f"  {mod.NAME}: 최근 12봉 내 발화 없음")
            for e in evs:
                kst = e.bar_time.tz_convert("Asia/Seoul").strftime("%m-%d %H시")
                print(f"  {mod.NAME} 발화: {kst} KST · {e.detail}")
