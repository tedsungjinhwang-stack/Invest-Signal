"""하락전환(CHoCH) 시그널 검출 로직 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import downtrend_reversal
from invest_signal.signals.downtrend_reversal import Params

P = Params()


def make_df(closes, lows=None, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    lo = pd.Series([float(x) for x in (lows or closes)], index=idx)
    return pd.DataFrame({"Open": c, "High": np.maximum(c, lo),
                         "Low": np.minimum(c, lo), "Close": c})


def uptrend_with_completed_dip(dip_wick=None, recover_bars=5):
    """상승 → 60선 아래 눌림(저점 형성) → 60선 위 복귀 구조를 만든다.

    반환: (closes, lows, level) — level은 눌림 구간 최저 저가.
    """
    closes = list(np.linspace(100.0, 200.0, 200))    # 상승 — 종가가 60선 위
    lows = closes.copy()
    m60 = float(pd.Series(closes).rolling(60).mean().iloc[-1])    # ≈ 185
    dip = m60 - 10.0
    level = dip_wick if dip_wick is not None else dip - 5.0
    closes += [dip, dip - 2, dip]                     # 눌림 3봉 (종가 < 60선)
    lows += [dip, level, dip]                         # 가운데 봉 꼬리가 최저점
    for i in range(recover_bars):                     # 60선 위 복귀 → 눌림 완성
        closes.append(m60 + 15.0 + i)
        lows.append(closes[-1])
    return closes, lows, level


def test_fires_on_first_close_below_prior_dip_low():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1.0)                        # 직전저점 하향돌파 마감
    lows.append(level - 1.0)
    df = make_df(closes, lows)
    evs = downtrend_reversal.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "downtrend_reversal"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["broken_low"] == level
    # 저점이 만들어진 봉(눌림 가운데 봉)의 시각이 기록된다
    assert ev.detail["low_time"] == df.index[len(closes) - 1 - 5 - 2].isoformat()


def test_no_fire_when_close_stays_above_level():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level + 1.0)                        # 저점 위에서 마감 — 돌파 아님
    lows.append(level - 3.0)                          # 꼬리만 뚫은 것도 돌파 아님(종가 기준)
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", P) == []


def test_no_fire_on_new_lows_inside_same_dip():
    """복귀 없이 진행 중인 눌림에서 저점 갱신하는 건 CHoCH가 아니다."""
    closes, lows, _ = uptrend_with_completed_dip(recover_bars=0)  # 복귀 없음
    closes.append(closes[-1] - 30.0)                  # 같은 눌림에서 폭락
    lows.append(closes[-1])
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", P) == []


def test_fires_once_then_silent():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1.0)                        # 돌파 봉(트리거)
    lows.append(level - 1.0)
    closes.append(level - 2.0)                        # 계속 그 아래 — 재발화 없음
    lows.append(level - 2.0)
    df = make_df(closes, lows)
    evs = downtrend_reversal.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]
    assert downtrend_reversal.detect(df, "TEST", Params(grace_bars=0)) == []


def test_lookback_expiry():
    closes, lows, level = uptrend_with_completed_dip(recover_bars=60)
    closes.append(level - 1.0)
    lows.append(level - 1.0)
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", Params(lookback_bars=40)) == []
    assert len(downtrend_reversal.detect(df, "TEST", Params(lookback_bars=180))) == 1
