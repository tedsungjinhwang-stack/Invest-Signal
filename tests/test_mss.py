"""MSS(직전 스윙저점 하향돌파) 시그널 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import mss
from invest_signal.signals.mss import Params

P = Params()


def make_df(closes, lows=None, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    lo = pd.Series([float(x) for x in (lows or closes)], index=idx)
    return pd.DataFrame({"Open": c, "High": np.maximum(c, lo),
                         "Low": np.minimum(c, lo), "Close": c})


def swing_low_setup():
    """상승 중 눌림으로 스윙저점(105, 좌우 3봉 대비 최저) 형성 후 반등 구조."""
    closes = [100, 102, 104, 106, 108, 110, 112, 110, 108, 106,   # 인덱스 9까지
              105,                                                # 인덱스 10 = 스윙저점
              107, 109, 111, 113, 115]                            # 반등 확정
    return list(map(float, closes)), 10, 105.0


def test_fires_on_first_close_below_swing_low():
    closes, piv_idx, level = swing_low_setup()
    closes.append(level - 1)                     # 직전저점 하향돌파 마감
    df = make_df(closes)
    evs = mss.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "mss"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["broken_low"] == level
    assert ev.detail["low_time"] == df.index[piv_idx].isoformat()


def test_no_fire_above_swing_low():
    closes, _, level = swing_low_setup()
    closes.append(level + 0.5)                   # 저점 위에서 마감
    df = make_df(closes)
    assert mss.detect(df, "TEST", P) == []


def test_fires_once_then_silent():
    closes, _, level = swing_low_setup()
    closes.append(level - 1)                     # 돌파 봉
    closes.append(level - 2)                     # 계속 아래 — 재발화 없음
    df = make_df(closes)
    evs = mss.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]
    assert mss.detect(df, "TEST", Params(grace_bars=0)) == []


def test_unconfirmed_low_is_not_reference():
    """저점 직후 k봉이 지나기 전 돌파 — 그 저점은 미확정이라 기준이 아니다."""
    closes = [100, 102, 104, 106, 108, 110, 112, 110, 108, 106,
              105,          # 새 저점 (아직 우측 3봉 미확보 → 미확정)
              104.5]        # 바로 아래로 마감
    df = make_df(list(map(float, closes)))
    # 확정된 스윙저점이 아직 없으므로(105는 우측 봉 부족) 발화하지 않는다
    assert mss.detect(df, "TEST", P) == []
