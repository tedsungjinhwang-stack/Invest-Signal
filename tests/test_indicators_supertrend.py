"""ATR·수퍼트렌드 지표 테스트 — 합성 데이터로 정의를 고정한다."""

import numpy as np
import pandas as pd

from invest_signal.indicators import atr, rma, supertrend, true_range


def frame(closes, highs=None, lows=None):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    h = pd.Series([float(x) for x in (highs if highs is not None else closes)], index=idx)
    lo = pd.Series([float(x) for x in (lows if lows is not None else closes)], index=idx)
    return pd.DataFrame({"Open": c, "High": np.maximum(c, h),
                         "Low": np.minimum(c, lo), "Close": c})


def test_rma_seeds_with_sma_then_smooths():
    """Wilder 평활 — 첫 값은 n개 단순평균, 이후 (prev*(n-1)+x)/n."""
    s = pd.Series([float(i) for i in range(1, 11)])
    out = rma(s, 3)
    assert out.iloc[:2].isna().all()          # n-1개는 아직 값 없음
    assert out.iloc[2] == 2.0                 # (1+2+3)/3
    assert out.iloc[3] == (2.0 * 2 + 4.0) / 3
    assert out.iloc[4] == (out.iloc[3] * 2 + 5.0) / 3


def test_true_range_uses_previous_close():
    df = frame(closes=[10, 12], highs=[10, 13], lows=[10, 11])
    tr = true_range(df)
    assert tr.iloc[1] == 3.0                  # max(13-11, |13-10|, |11-10|) = 3


def test_atr_is_rma_of_true_range():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    df = frame(c, c + 1.0, c - 1.0)
    assert np.allclose(atr(df, 14).dropna(), rma(true_range(df), 14).dropna())


def test_supertrend_flips_on_band_cross():
    """하락 추세를 타다가 상단 밴드를 종가로 넘기면 상승으로 뒤집힌다."""
    down = list(np.linspace(200.0, 100.0, 120))
    df = frame(down)
    st = supertrend(df, 22, 3.0)
    assert st.iloc[-1] == -1

    spike = down + [down[-1] * 1.5]          # ATR을 훌쩍 넘는 급등 봉
    assert supertrend(frame(spike), 22, 3.0).iloc[-1] == 1


def test_supertrend_holds_direction_through_shallow_pullback():
    """상승 전환 뒤 얕은 눌림으로는 방향이 안 바뀐다 (트레일링 스톱)."""
    closes = list(np.linspace(100.0, 200.0, 120)) + [199.0, 198.0]
    st = supertrend(frame(closes), 22, 3.0)
    assert st.iloc[-1] == 1
    # 밴드를 조이면 같은 눌림에도 뒤집힌다 — 배수가 실제로 폭을 정한다
    assert supertrend(frame(closes), 22, 0.2).iloc[-1] == -1


def test_supertrend_nan_until_atr_available():
    st = supertrend(frame(list(np.linspace(100.0, 90.0, 30))), 22, 3.0)
    assert st.iloc[:21].isna().all() and not pd.isna(st.iloc[-1])
