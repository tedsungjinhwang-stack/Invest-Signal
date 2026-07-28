"""펌핑초기(크립토 전용) 시그널 테스트 — 합성 데이터."""

import pandas as pd

from invest_signal.signals import pump_early
from invest_signal.signals.pump_early import Params

P = Params()


def make_df(bars, start="2025-01-06"):
    """bars: [(open, high, low, close), ...] — 4h봉."""
    idx = pd.date_range(start, periods=len(bars), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"Open": [float(b[0]) for b in bars], "High": [float(b[1]) for b in bars],
         "Low": [float(b[2]) for b in bars], "Close": [float(b[3]) for b in bars]},
        index=idx)


def flat(n, price=100.0):
    return [(price, price, price, price)] * n


def test_fires_on_5pct_bar():
    """44봉 횡보 뒤 한 봉에서 시가 대비 +5% → 발화."""
    df = make_df(flat(44) + [(100.0, 105.0, 99.5, 105.0)])
    evs = pump_early.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "pump_early"
    assert ev.bar_time == df.index[-1]
    assert abs(ev.detail["rise"] - 0.05) < 1e-9
    assert abs(ev.detail["total_gain"] - (105.0 / 99.5 - 1)) < 1e-9   # 창 저점 = 급등봉 저가
    assert ev.detail["trigger_low"] == 99.5


def test_no_fire_below_min_gain():
    """+4.9%는 문턱 미달."""
    df = make_df(flat(44) + [(100.0, 104.9, 100.0, 104.9)])
    assert pump_early.detect(df, "TEST", P) == []


def test_no_fire_when_already_pumped():
    """한 봉에 +40% — 급등이지만 이미 크게 간 것이라 '초기'가 아니다.

    이런 종목은 pump_dip이 월간 VWAP 눌림 자리에서 잡는다.
    """
    df = make_df(flat(44) + [(100.0, 140.0, 100.0, 140.0)])
    assert pump_early.detect(df, "TEST", P) == []
    # 상한을 풀면 같은 데이터로 발화한다 — 걸린 조건이 max_gain임을 확인
    assert len(pump_early.detect(df, "TEST", Params(max_gain=10.0))) == 1


def test_no_fire_when_cumulative_gain_over_max():
    """5% 급등이지만 7일 저점 대비 누적이 이미 30%를 넘으면 제외."""
    bars = flat(20) + flat(24, 125.0) + [(125.0, 131.25, 125.0, 131.25)]
    df = make_df(bars)                      # 저점 100 → 종가 131.25 = +31.25%
    assert pump_early.detect(df, "TEST", P) == []


def test_consecutive_pump_bars_alert_once():
    """연속 급등 구간에서는 첫 봉만 알린다."""
    bars = flat(44) + [(100.0, 105.0, 100.0, 105.0), (105.0, 110.25, 105.0, 110.25)]
    df = make_df(bars)
    evs = pump_early.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_rise_bars_two_spans_eight_hours():
    """rise_bars=2면 8시간(2봉) 누적으로 판정 — 봉당 3%씩도 잡힌다."""
    bars = flat(44) + [(100.0, 103.0, 100.0, 103.0), (103.0, 106.0, 103.0, 106.0)]
    df = make_df(bars)
    assert pump_early.detect(df, "TEST", P) == []          # 봉별로는 3%씩 — 미달
    evs = pump_early.detect(df, "TEST", Params(rise_bars=2))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-1]
    assert abs(evs[0].detail["rise"] - 0.06) < 1e-9        # 100 → 106


def test_still_active_until_trigger_low_breaks():
    """급등 봉 저가를 종가가 지키는 동안 유지, 아래로 마감하면 제거."""
    df = make_df(flat(44) + [(100.0, 105.0, 99.5, 105.0)])
    ev = pump_early.detect(df, "TEST", P)[0]
    hold = make_df(flat(44) + [(100.0, 105.0, 99.5, 105.0),
                               (105.0, 106.0, 100.0, 101.0)])
    assert pump_early.still_active(hold, ev, P)            # 종가 101 ≥ 저가 99.5
    dead = make_df(flat(44) + [(100.0, 105.0, 99.5, 105.0),
                               (105.0, 105.0, 95.0, 96.0)])
    assert not pump_early.still_active(dead, ev, P)        # 종가 96 < 99.5


def test_no_vwap_or_volume_requirement():
    """Volume 컬럼이 없어도 발화한다 — VWAP 조건이 없는 시그널."""
    df = make_df(flat(44) + [(100.0, 105.0, 100.0, 105.0)])
    assert "Volume" not in df.columns
    assert len(pump_early.detect(df, "TEST", P)) == 1


def test_short_history_is_skipped():
    df = make_df(flat(20) + [(100.0, 105.0, 100.0, 105.0)])
    assert pump_early.detect(df, "TEST", P) == []


def test_crypto_only_flag_and_registration():
    """ETF·주식 스캔 제외 플래그 + detectors() 기본 등록 + 랭크 필터 범위."""
    from invest_signal import config as cfg_mod
    from invest_signal.scanner import RANK_FILTER_SCOPE

    assert pump_early.CRYPTO_ONLY is True
    assert pump_early.INTRABAR_OK is True
    cfg = {"signal": {}, "crypto": {}, "etf": {}}
    assert "pump_early" in [m.NAME for m, _ in cfg_mod.detectors(cfg)]
    assert "pump_early" not in [m.NAME for m, _ in cfg_mod.detectors(cfg)
                                if getattr(m, "CRYPTO_ONLY", False) is False]
    assert RANK_FILTER_SCOPE["pump_early"] == frozenset({"pump_early"})


def test_params_from_config():
    from invest_signal import config as cfg_mod

    p = cfg_mod.pump_early_params(
        {"signal": {"pump_early": {"rise_bars": 2, "min_gain": 0.07,
                                   "lookback_bars": 18, "max_gain": 0.5}}})
    assert (p.rise_bars, p.min_gain, p.lookback_bars, p.max_gain) == (2, 0.07, 18, 0.5)
