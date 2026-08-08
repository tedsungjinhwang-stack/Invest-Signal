"""펌핑초기(크립토 전용) 시그널 테스트 — 합성 데이터."""

import pandas as pd

from invest_signal.signals import pump_early
from invest_signal.signals.pump_early import Params

P = Params()

# 480선 조건을 쓰므로 픽스처는 480봉 이상 + 캔들이 480선 아래여야 한다.
HIGH, LOW = 200.0, 100.0        # 고가권 → 저가권으로 내려온 시세


def make_df(bars, start="2025-01-06"):
    """bars: [(open, high, low, close), ...] — 4h봉."""
    idx = pd.date_range(start, periods=len(bars), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"Open": [float(b[0]) for b in bars], "High": [float(b[1]) for b in bars],
         "Low": [float(b[2]) for b in bars], "Close": [float(b[3]) for b in bars]},
        index=idx)


def flat(n, price=LOW):
    return [(price, price, price, price)] * n


def base(tail=42, tail_price=LOW, head=440):
    """480선 아래에 놓인 저가권 시세 — head봉 고가권 뒤 tail봉 저가권."""
    return flat(head, HIGH) + flat(tail, tail_price)


def test_fires_on_5pct_bar():
    """480봉 이상 이력 + 480선 아래에서 한 봉 시가 대비 +5% → 발화."""
    df = make_df(base() + [(100.0, 105.0, 99.5, 105.0)])
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
    df = make_df(base() + [(100.0, 104.9, 100.0, 104.9)])
    assert pump_early.detect(df, "TEST", P) == []


def test_no_fire_when_close_above_ref_ma():
    """캔들이 480선 위면 제외 — 이미 추세가 시작된 종목."""
    df = make_df(flat(482, 100.0) + [(100.0, 105.0, 100.0, 105.0)])
    assert pump_early.detect(df, "TEST", P) == []      # 480선 ≈ 100, 종가 105 → 위
    evs = pump_early.detect(df, "TEST", Params(below_ma_condition=False))
    assert len(evs) == 1                               # 조건을 끄면 발화


def test_no_fire_when_already_pumped():
    """한 봉에 +40% — 급등이지만 이미 크게 간 것이라 '초기'가 아니다.

    하루 만에 확 오른 종목은 이미 '초기'가 아니다.
    """
    df = make_df(base() + [(100.0, 140.0, 100.0, 140.0)])
    assert pump_early.detect(df, "TEST", P) == []
    # 두 상한을 다 풀면 같은 데이터로 발화한다 — 걸린 게 상한 조건임을 확인
    assert len(pump_early.detect(
        df, "TEST", Params(max_gain=10.0, max_pump_gain=10.0))) == 1


def test_no_fire_when_day_pump_happened_earlier():
    """되돌려졌더라도 조회 구간 안에 '하루 내 저점→고가 +30%'가 있으면 제외.

    누적 상승률은 +10%뿐이라 max_gain에는 안 걸리는 케이스 — 하루 급등
    이력만으로 걸러지는지 확인한다.
    """
    bars = (base(tail=23) + [(100.0, 132.0, 100.0, 132.0)]   # 하루 만에 +32% 급등
            + flat(18, 105.0)                                # 되돌림
            + [(105.0, 110.25, 105.0, 110.25)])              # +5% 재점화
    df = make_df(bars)
    assert pump_early.detect(df, "TEST", P) == []
    # 하루 급등 상한만 풀면 발화 — 누적(max_gain)에는 안 걸렸음을 확인
    evs = pump_early.detect(df, "TEST", Params(max_pump_gain=10.0))
    assert len(evs) == 1 and abs(evs[0].detail["total_gain"] - 0.1025) < 1e-9


def test_old_day_pump_outside_lookback_is_ignored():
    """조회 구간(42봉) 밖의 옛 급등은 제외 사유가 아니다."""
    bars = (flat(400, HIGH) + [(100.0, 132.0, 100.0, 132.0)] + flat(81, 105.0)
            + [(105.0, 110.25, 105.0, 110.25)])
    df = make_df(bars)
    evs = pump_early.detect(df, "TEST", P)
    assert len(evs) == 1 and evs[0].bar_time == df.index[-1]


def test_consecutive_pump_bars_alert_once():
    """연속 급등 구간에서는 첫 봉만 알린다."""
    bars = base() + [(100.0, 105.0, 100.0, 105.0), (105.0, 110.25, 105.0, 110.25)]
    df = make_df(bars)
    evs = pump_early.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_rise_bars_two_spans_eight_hours():
    """rise_bars=2면 8시간(2봉) 누적으로 판정 — 봉당 3%씩도 잡힌다."""
    bars = base() + [(100.0, 103.0, 100.0, 103.0), (103.0, 106.0, 103.0, 106.0)]
    df = make_df(bars)
    assert pump_early.detect(df, "TEST", P) == []          # 봉별로는 3%씩 — 미달
    evs = pump_early.detect(df, "TEST", Params(rise_bars=2))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-1]
    assert abs(evs[0].detail["rise"] - 0.06) < 1e-9        # 100 → 106


def test_still_active_until_trigger_low_breaks():
    """급등 봉 저가를 종가가 지키는 동안 유지, 아래로 마감하면 제거."""
    trigger = (100.0, 105.0, 99.5, 105.0)
    df = make_df(base() + [trigger])
    ev = pump_early.detect(df, "TEST", P)[0]
    hold = make_df(base() + [trigger, (105.0, 106.0, 100.0, 101.0)])
    assert pump_early.still_active(hold, ev, P)            # 종가 101 ≥ 저가 99.5
    dead = make_df(base() + [trigger, (105.0, 105.0, 95.0, 96.0)])
    assert not pump_early.still_active(dead, ev, P)        # 종가 96 < 99.5


def test_no_vwap_or_volume_requirement():
    """Volume 컬럼이 없어도 발화한다 — VWAP 조건이 없는 시그널."""
    df = make_df(base() + [(100.0, 105.0, 100.0, 105.0)])
    assert "Volume" not in df.columns
    assert len(pump_early.detect(df, "TEST", P)) == 1


def test_short_history_is_skipped():
    """480선을 계산할 수 없는 신규 상장은 대상이 아니다."""
    df = make_df(flat(100) + [(100.0, 105.0, 100.0, 105.0)])
    assert pump_early.detect(df, "TEST", P) == []
    # 480선 조건을 끄면 짧은 이력으로도 발화한다
    assert len(pump_early.detect(df, "TEST", Params(below_ma_condition=False))) == 1


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
                                   "lookback_bars": 18, "max_gain": 0.5,
                                   "pump_window_bars": 3, "max_pump_gain": 0.4,
                                   "ma_ref": 240, "below_ma_condition": False}}})
    assert (p.rise_bars, p.min_gain, p.lookback_bars, p.max_gain) == (2, 0.07, 18, 0.5)
    assert (p.pump_window_bars, p.max_pump_gain) == (3, 0.4)
    assert (p.ma_ref, p.below_ma_condition) == (240, False)
