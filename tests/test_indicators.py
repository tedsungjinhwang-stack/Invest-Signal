"""지표 계산 테스트."""

import numpy as np
import pandas as pd
import pytest

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


def test_monthly_vwap_resets_at_month_start():
    from invest_signal.indicators import monthly_vwap
    df = _df([10, 20, 30, 40] * 6, [1] * 24, start="2025-04-30")
    mv = monthly_vwap(df)
    first_may = np.where(df.index >= pd.Timestamp("2025-05-01", tz="UTC"))[0][0]
    assert mv.iloc[first_may] == df["Close"].iloc[first_may]


def test_alignment_states():
    from invest_signal.indicators import alignment
    idx = pd.date_range("2025-01-01", periods=600, freq="4h", tz="UTC")
    down = pd.Series(np.linspace(400.0, 100.0, 600), index=idx)
    df_down = pd.DataFrame({"Open": down, "High": down, "Low": down, "Close": down})
    assert alignment(df_down) == "역배열"
    up = pd.Series(np.linspace(100.0, 400.0, 600), index=idx)
    df_up = pd.DataFrame({"Open": up, "High": up, "Low": up, "Close": up})
    assert alignment(df_up) == "정배열"
    assert alignment(df_up.iloc[:100]) is None   # 480선 계산 불가


def test_pct_since_uses_timestamps_not_bar_counts():
    """장 시간에만 봉이 있는 프레임에서도 이름 그대로의 기간을 재야 한다.

    4h 버킷이 빠진 ETF 프레임에서 봉을 세면 '24h'가 며칠 전이 된다.
    """
    from invest_signal.indicators import pct_since

    # 하루에 4h봉 2개만 있는 프레임 (장 시간만) — 3일치
    idx = pd.DatetimeIndex([
        "2026-08-10T12:00Z", "2026-08-10T16:00Z",
        "2026-08-11T12:00Z", "2026-08-11T16:00Z",
        "2026-08-12T12:00Z", "2026-08-12T16:00Z",
    ])
    close = pd.Series([100.0, 101.0, 110.0, 111.0, 120.0, 132.0], index=idx)

    # 24시간 전(08-11 16:00)이 기준 → 132/111 - 1
    assert pct_since(close, pd.Timedelta(hours=24)) == pytest.approx(132 / 111 - 1)
    # 봉 개수(6봉=24h)로 셌다면 100이 기준이 되어 완전히 다른 값이 나온다
    assert pct_since(close, pd.Timedelta(hours=24)) != pytest.approx(132 / 100 - 1)


def test_pct_since_returns_none_when_history_is_too_short():
    """이력이 그 기간까지 못 닿으면 0이 아니라 None."""
    from invest_signal.indicators import pct_since

    idx = pd.date_range("2026-08-15", periods=3, freq="4h", tz="UTC")
    close = pd.Series([1.0, 2.0, 3.0], index=idx)
    assert pct_since(close, pd.Timedelta(days=7)) is None
    assert pct_since(close, pd.Timedelta(hours=8)) == pytest.approx(2.0)


def test_pct_since_matches_pct_over_on_a_gapless_frame():
    """크립토처럼 빈 봉이 없는 프레임에서는 봉 세기와 결과가 같다."""
    from invest_signal.indicators import pct_over, pct_since

    idx = pd.date_range("2026-08-01", periods=60, freq="4h", tz="UTC")
    close = pd.Series(np.linspace(10.0, 20.0, 60), index=idx)
    assert pct_since(close, pd.Timedelta(hours=24)) == pytest.approx(pct_over(close, 6))
    assert pct_since(close, pd.Timedelta(days=7)) == pytest.approx(pct_over(close, 42))
