"""지표 계산 테스트."""

import numpy as np
import pandas as pd

from invest_signal.indicators import quarterly_vwap, sma


def _df(closes, volumes, start="2025-03-30"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": [float(v) for v in volumes]})


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = sma(s, 2)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.5
    assert out.iloc[3] == 3.5


def test_quarterly_vwap_resets_at_quarter_start():
    # 3/30~4/1에 걸친 데이터 — 2분기 첫 봉에서 VWAP이 리셋되어야 한다
    df = _df([10, 20, 30, 40] * 6, [1] * 24, start="2025-03-31")
    qv = quarterly_vwap(df)
    q2_mask = df.index >= pd.Timestamp("2025-04-01", tz="UTC")
    first_q2 = np.where(q2_mask)[0][0]
    # 분기 첫 봉의 VWAP == 그 봉의 typical price (H=L=C라 close와 동일)
    assert qv.iloc[first_q2] == df["Close"].iloc[first_q2]
    # 분기 내 누적 평균: 2번째 봉은 두 봉의 가중평균
    v = (df["Close"].iloc[first_q2] + df["Close"].iloc[first_q2 + 1]) / 2
    assert abs(qv.iloc[first_q2 + 1] - v) < 1e-9


def test_quarterly_vwap_weighted_by_volume():
    df = _df([10, 30], [1, 3], start="2025-04-01")
    qv = quarterly_vwap(df)
    assert abs(qv.iloc[1] - (10 * 1 + 30 * 3) / 4) < 1e-9


def test_quarterly_vwap_without_volume_returns_none():
    df = _df([10, 20], [1, 1])
    assert quarterly_vwap(df.drop(columns=["Volume"])) is None
    assert quarterly_vwap(_df([10, 20], [0, 0])) is None
