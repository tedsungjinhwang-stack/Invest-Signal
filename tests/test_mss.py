"""MSS(직전 스윙저점 하향돌파) 시그널 테스트 — 합성 데이터.

MSS는 눌림목과 **완전히 같은 진입 조건**(480선 위·60선 하회·분기 VWAP
하단 밴드 위)을 태우므로 픽스처도 눌림목과 같은 구조를 쓴다:
2024-10-01 시작 → 552봉째가 분기 경계(2025-01-01), 이전 분기 횡보 +
이번 분기 상승이라 하단 밴드가 480선 위에 놓인다.
"""

import numpy as np
import pandas as pd

from invest_signal.indicators import quarterly_vwap_bands, sma
from invest_signal.signals import mss, pullback
from invest_signal.signals.mss import Params

P = Params(pivot_k=3)          # 픽스처가 k=3 기준으로 구성됨
PREV_Q_BARS = 552

SWING_LOW = 118.0              # 아래 dip_bars()가 만드는 스윙저점


def make_df(bars, start="2024-10-01"):
    """bars: [(open, high, low, close, volume), ...] — 4h봉."""
    idx = pd.date_range(start, periods=len(bars), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"Open": [float(b[0]) for b in bars], "High": [float(b[1]) for b in bars],
         "Low": [float(b[2]) for b in bars], "Close": [float(b[3]) for b in bars],
         "Volume": [float(b[4]) for b in bars]}, index=idx)


def bar(close, low=None):
    """거래량이 거의 없는 봉 — 밴드를 움직이지 않게 한다."""
    return (close, close, close if low is None else low, close, 0.001)


def base_bars():
    """이전 분기 100 횡보 → 이번 분기 100→150 상승 (480선·밴드 배치용)."""
    flat = [(100.0, 100.0, 100.0, 100.0, 1.0)] * PREV_Q_BARS
    rise = [(c, c, c, c, 1.0) for c in np.linspace(100.0, 150.0, 68)]
    return flat + rise


def dip_bars():
    """60선 아래 조정 구간 — 118에서 스윙저점(좌우 3봉 대비 최저) 확정."""
    return [bar(124), bar(122), bar(120), bar(SWING_LOW), bar(120), bar(122), bar(124)]


def setup(*extra):
    return make_df(base_bars() + dip_bars() + list(extra))


def test_fixture_satisfies_pullback_gate():
    """전제 확인 — 조정 구간이 480선 위·60선 아래, 밴드가 그 사이에 있다."""
    df = setup(bar(117))
    ma60 = float(sma(df["Close"], 60).iloc[-1])
    ma480 = float(sma(df["Close"], pullback.Params().ma_above).iloc[-1])
    band = float(quarterly_vwap_bands(df, 1.0)[1].iloc[-1])
    assert ma480 < band < 117.0 < ma60


def test_fires_on_first_close_below_swing_low():
    df = setup(bar(117))
    evs = mss.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "mss"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["broken_low"] == SWING_LOW
    assert ev.detail["low_time"] == df.index[-5].isoformat()
    assert ev.detail["stage"] == "대기"          # 캔들이 밴드 위


def test_stage_marks_band_touch_as_entry():
    """돌파 봉이 하단 밴드에 닿으면 눌림목과 같은 '타점'으로 표기된다."""
    df = setup((117.0, 117.0, 108.0, 117.0, 0.001))
    evs = mss.detect(df, "TEST", P)
    assert len(evs) == 1 and evs[0].detail["stage"] == "타점"


def test_no_fire_when_pullback_gate_fails():
    """눌림목 조건을 못 넘는 저점 이탈은 MSS로 표시하지 않는다."""
    # 480선 아래로 마감 — 장기 추세가 깨진 이탈
    assert mss.detect(setup(bar(100)), "TEST", P) == []
    # 캔들 전체가 하단 밴드 아래 (고가 109 < 밴드 110.4)
    assert mss.detect(setup((109.0, 109.0, 105.0, 106.0, 0.001)), "TEST", P) == []


def test_no_fire_above_swing_low():
    assert mss.detect(setup(bar(119)), "TEST", P) == []


def test_fires_once_then_silent():
    df = setup(bar(117), bar(116))               # 돌파 후 계속 아래
    evs = mss.detect(df, "TEST", Params(pivot_k=3, grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]
    assert mss.detect(df, "TEST", Params(pivot_k=3, grace_bars=0)) == []


def test_unconfirmed_low_is_not_reference():
    """저점 직후 k봉이 지나기 전 돌파 — 그 저점은 미확정이라 기준이 아니다."""
    bars = base_bars() + [bar(124), bar(122), bar(120), bar(118), bar(117.5)]
    assert mss.detect(make_df(bars), "TEST", P) == []


def test_still_active_tracks_hold_below_broken_level():
    df = setup(bar(117))
    ev = mss.detect(df, "TEST", P)[0]
    assert mss.still_active(setup(bar(117), bar(116)), ev, P)      # 계속 저점 아래
    assert not mss.still_active(setup(bar(117), bar(119)), ev, P)  # 저점 위 복귀


def test_gate_comes_from_pullback_config():
    """MSS 진입 조건은 config의 눌림목 설정을 그대로 공유한다."""
    from invest_signal import config as cfg_mod

    cfg = {"signal": {"pullback": {"ma_entry": 120, "ma_above": 240, "band_mult": 2.0}}}
    p = cfg_mod.mss_params(cfg)
    assert (p.gate.ma_entry, p.gate.ma_above, p.gate.band_mult) == (120, 240, 2.0)
    assert p.gate == cfg_mod.pullback_params(cfg)
