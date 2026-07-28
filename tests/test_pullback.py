"""눌림목 시그널 검출 로직 테스트 — 합성 데이터.

픽스처는 '이전 분기 횡보 → 이번 분기 상승' 구조라 분기 VWAP 하단 밴드가
480선 위에 놓인다 (실제 상승 추세 종목의 배치). 2024-10-01 시작이므로
552봉째가 분기 경계(2025-01-01)이고 그 뒤가 이번 분기다.
"""

import numpy as np
import pandas as pd

from invest_signal.indicators import quarterly_vwap_bands, sma
from invest_signal.signals import pullback
from invest_signal.signals.pullback import Params

P = Params()
PREV_Q_BARS = 552           # 2024-10-01 + 552봉(4h) = 2025-01-01 (분기 경계)


def make_df(bars, start="2024-10-01"):
    """bars: [(open, high, low, close, volume), ...] — 4h봉."""
    idx = pd.date_range(start, periods=len(bars), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"Open": [float(b[0]) for b in bars], "High": [float(b[1]) for b in bars],
         "Low": [float(b[2]) for b in bars], "Close": [float(b[3]) for b in bars],
         "Volume": [float(b[4]) for b in bars]}, index=idx)


def base_bars():
    """이전 분기 100 횡보 → 이번 분기 100→150 상승."""
    flat = [(100.0, 100.0, 100.0, 100.0, 1.0)] * PREV_Q_BARS
    rise = [(c, c, c, c, 1.0) for c in np.linspace(100.0, 150.0, 68)]
    return flat + rise


def levels(df):
    lower = quarterly_vwap_bands(df, P.band_mult)[1]
    return (float(sma(df["Close"], P.ma_entry).iloc[-1]),
            float(sma(df["Close"], P.ma_above).iloc[-1]),
            float(lower.iloc[-1]))


def test_fixture_places_band_between_the_two_mas():
    """전제 확인 — 하단 밴드가 480선 위, 60선 아래에 놓여야 테스트가 유효하다."""
    ma60, ma480, band = levels(make_df(base_bars()))
    assert ma480 < band < ma60


def test_waiting_stage_above_band():
    """480선 위 + 60선 하회 + 캔들 전체가 밴드 위 → 대기."""
    df = make_df(base_bars() + [(150.0, 150.0, 118.0, 120.0, 0.001)])
    evs = pullback.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "pullback"
    assert ev.detail["stage"] == "대기"
    assert ev.bar_time == df.index[-1]


def test_entry_stage_on_band_touch():
    """캔들이 하단 밴드에 닿으면 타점."""
    df = make_df(base_bars() + [(150.0, 150.0, 108.0, 118.0, 0.001)])
    evs = pullback.detect(df, "TEST", P)
    assert len(evs) == 1 and evs[0].detail["stage"] == "타점"
    assert evs[0].detail["band"] is not None


def test_no_fire_when_candle_below_band():
    """캔들이 밴드 아래로 완전히 벗어나면 대상이 아니다."""
    df = make_df(base_bars() + [(150.0, 109.0, 100.0, 105.0, 0.001)])
    assert pullback.detect(df, "TEST", P) == []


def test_no_fire_when_close_above_ma60():
    """60선을 하회하지 않으면 조정이 아니다."""
    df = make_df(base_bars() + [(150.0, 150.0, 140.0, 145.0, 0.001)])
    assert pullback.detect(df, "TEST", P) == []


def test_no_fire_when_close_below_ma480():
    """장기선(480) 아래로 마감하면 추세가 깨진 것 — 눌림목이 아니다."""
    bars = base_bars() + [(150.0, 150.0, 95.0, 100.0, 0.001)]
    df = make_df(bars)
    _, ma480, _ = levels(df)
    assert 100.0 < ma480                       # 전제: 종가가 480선 아래
    assert pullback.detect(df, "TEST", P) == []


def test_waiting_then_entry_in_same_dip():
    """같은 하회 구간에서 대기 → 타점 순으로 각각 한 번씩 알린다."""
    bars = base_bars() + [(150.0, 150.0, 118.0, 120.0, 0.001),   # 대기
                          (120.0, 121.0, 108.0, 118.0, 0.001)]   # 밴드 터치 → 타점
    df = make_df(bars)
    evs = pullback.detect(df, "TEST", Params(grace_bars=1))
    stages = {e.bar_time: e.detail["stage"] for e in evs}
    assert stages[df.index[-2]] == "대기"
    assert stages[df.index[-1]] == "타점"


def test_repeated_bars_alert_once_per_stage():
    """대기가 이어지는 동안 매 봉 알리지 않는다."""
    bars = base_bars() + [(150.0, 150.0, 118.0, 120.0, 0.001),
                          (120.0, 121.0, 117.0, 119.0, 0.001)]   # 여전히 대기
    df = make_df(bars)
    evs = pullback.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_entry_not_repeated_after_first_touch():
    """구간 내 첫 터치 이후의 재터치는 알리지 않는다."""
    bars = base_bars() + [(150.0, 150.0, 108.0, 118.0, 0.001),   # 첫 터치
                          (118.0, 119.0, 107.0, 117.0, 0.001)]   # 재터치
    df = make_df(bars)
    evs = pullback.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_band_condition_off_ignores_band():
    """band_condition=false면 ①480선 위 + ②60선 하회만 본다."""
    df = make_df(base_bars() + [(150.0, 109.0, 100.0, 105.0, 0.001)])
    assert pullback.detect(df, "TEST", P) == []            # 밴드 아래라 제외
    evs = pullback.detect(df, "TEST", Params(band_condition=False))
    assert len(evs) == 1 and evs[0].detail["stage"] == "대기"


def test_deeper_band_needs_deeper_dip():
    """band_mult를 키우면 밴드가 내려가 같은 눌림이 타점이 아닌 대기가 된다."""
    df = make_df(base_bars() + [(150.0, 150.0, 108.0, 118.0, 0.001)])
    assert pullback.detect(df, "TEST", P)[0].detail["stage"] == "타점"
    evs = pullback.detect(df, "TEST", Params(band_mult=2.0))
    assert len(evs) == 1 and evs[0].detail["stage"] == "대기"


def test_still_active_until_band_or_ma480_breaks():
    """밴드 위·480선 위를 지키는 동안 유지, 어느 쪽이든 깨지면 제거."""
    trigger = (150.0, 150.0, 108.0, 118.0, 0.001)
    df = make_df(base_bars() + [trigger])
    ev = pullback.detect(df, "TEST", P)[0]
    hold = make_df(base_bars() + [trigger, (118.0, 140.0, 118.0, 135.0, 0.001)])
    assert pullback.still_active(hold, ev, P)               # 밴드·480선 위 복귀
    below_band = make_df(base_bars() + [trigger, (118.0, 118.0, 105.0, 106.0, 0.001)])
    assert not pullback.still_active(below_band, ev, P)     # 밴드 아래 마감
    below_ma = make_df(base_bars() + [trigger, (118.0, 118.0, 95.0, 99.0, 0.001)])
    assert not pullback.still_active(below_ma, ev, P)       # 480선 아래 마감


def test_no_volume_skips_band_condition():
    """Volume이 없으면 밴드를 계산할 수 없으므로 ①②만으로 판정한다."""
    bars = base_bars() + [(150.0, 150.0, 118.0, 120.0, 0.001)]
    df = make_df(bars).drop(columns=["Volume"])
    evs = pullback.detect(df, "TEST", P)
    assert len(evs) == 1 and evs[0].detail["stage"] == "대기"
    assert evs[0].detail["band"] is None


def test_params_from_config():
    from invest_signal import config as cfg_mod

    p = cfg_mod.pullback_params(
        {"signal": {"pullback": {"ma_entry": 120, "ma_above": 240,
                                 "band_mult": 2.0, "band_condition": False}}})
    assert (p.ma_entry, p.ma_above, p.band_mult, p.band_condition) == (120, 240, 2.0, False)
    # 기본값은 새 스펙 — 60선 하회 / 480선 위 / 1σ 밴드
    d = cfg_mod.pullback_params({"signal": {}})
    assert (d.ma_entry, d.ma_above, d.band_mult) == (60, 480, 1.0)
