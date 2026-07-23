"""스캐너 헬퍼 테스트."""

import pandas as pd

from invest_signal.scanner import _collapse
from invest_signal.signals import SignalEvent


def _ev(symbol, signal, hour):
    return SignalEvent(symbol=symbol, signal=signal,
                       bar_time=pd.Timestamp(f"2026-07-22 {hour:02d}:00", tz="UTC"),
                       price=1.0, detail={})


def test_collapse_keeps_latest_per_symbol_signal():
    evs = [_ev("ADXUSDT", "uptrend_onset", 16), _ev("ADXUSDT", "uptrend_onset", 20),
           _ev("ADXUSDT", "pullback", 20), _ev("BTCUSDT", "uptrend_onset", 20)]
    out = _collapse(evs)
    assert len(out) == 3
    ut = [e for e in out if e.symbol == "ADXUSDT" and e.signal == "uptrend_onset"]
    assert len(ut) == 1 and ut[0].bar_time.hour == 20


def test_detect_all_choch_evicts_prior_long_holds():
    """하락전환(CHoCH) 이후의 셋업만 유지 중 리스트에 살아남는다."""
    from types import SimpleNamespace
    from invest_signal.scanner import _detect_all
    from invest_signal.signals.pullback import Params

    idx = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                      index=idx)

    def ev2(sig, label, i):
        return SignalEvent(symbol="X", signal=sig, bar_time=idx[i],
                           price=1.0, detail={"label": label})

    def mod(name, events):
        return SimpleNamespace(NAME=name,
                               detect=lambda d, s, p: list(events),
                               still_active=lambda d, e, p: True)

    pb = ev2("pullback", "눌림목", 3)
    up = ev2("uptrend_onset", "상승초입", 2)
    mss_ev = ev2("mss", "MSS", 4)
    # CHoCH(5봉)가 셋업들 이후 → 눌림목만 제거, 상승초입·MSS는 유지
    dets = [(mod("pullback", [pb]), Params()),
            (mod("uptrend_onset", [up]), Params()),
            (mod("mss", [mss_ev]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 5)]), Params())]
    _, ongoing = _detect_all({"X": df}, dets)
    assert [e.signal for e in ongoing] == ["uptrend_onset", "mss"]
    # CHoCH(1봉)가 눌림목(3봉)보다 과거 → 눌림목 유지
    dets = [(mod("pullback", [pb]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 1)]), Params())]
    _, ongoing = _detect_all({"X": df}, dets)
    assert [e.signal for e in ongoing] == ["pullback"]


def test_detect_all_show_choch_displays_downtrend_events():
    """show_choch=True(ETF·주식)면 하락전환이 알림·추적에 잡히고,
    기본값(크립토)이면 제거 용도로만 쓰이고 표시되지 않는다."""
    from types import SimpleNamespace
    from invest_signal.scanner import _detect_all
    from invest_signal.signals.pullback import Params

    idx = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                      index=idx)

    def ev2(sig, label, i):
        return SignalEvent(symbol="X", signal=sig, bar_time=idx[i],
                           price=1.0, detail={"label": label})

    def mod(name, events):
        return SimpleNamespace(NAME=name,
                               detect=lambda d, s, p: list(events),
                               still_active=lambda d, e, p: True)

    dets = [(mod("pullback", [ev2("pullback", "눌림목", 3)]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 5)]), Params())]

    events, ongoing = _detect_all({"X": df}, dets, show_choch=True)
    assert "downtrend_reversal" in {e.signal for e in events}
    assert [e.signal for e in ongoing] == ["downtrend_reversal"]   # 풀백은 CHoCH로 제거

    events, ongoing = _detect_all({"X": df}, dets)                 # 크립토 기본 동작
    assert "downtrend_reversal" not in {e.signal for e in events}
    assert ongoing == []


def test_filter_ranked_covers_pullback_and_mss_only():
    """랭크 필터는 풀백 칸 표시분(풀백+MSS)만 거르고 상승초입은 통과."""
    from invest_signal.scanner import _filter_ranked

    evs = [_ev("BIGVOL", "pullback", 0), _ev("THIN", "pullback", 0),
           _ev("BIGVOL", "mss", 4), _ev("THIN", "mss", 4),
           _ev("THIN", "uptrend_onset", 8)]
    out = _filter_ranked(evs, {"BIGVOL"})
    assert {(e.symbol, e.signal) for e in out} == {
        ("BIGVOL", "pullback"), ("BIGVOL", "mss"), ("THIN", "uptrend_onset")}


def test_crypto_rank_eligible_union_of_volume_and_gain():
    from invest_signal.scanner import _crypto_rank_eligible

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")

    def frame(price, vol, rising=False):
        import numpy as np
        c = pd.Series(np.linspace(price * (0.5 if rising else 1.0), price, 50), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                             "Volume": float(vol)})

    frames = {
        "BIGVOL": frame(1.0, 1_000_000),          # 거래대금 1위, 상승률 없음
        "PUMP": frame(2.0, 10, rising=True),      # 상승률 1위, 거래대금 미미
        "SLEEPY": frame(1.0, 1),                  # 둘 다 하위
    }
    eligible = _crypto_rank_eligible(frames, {"volume_top": 1, "gain_top": 1,
                                              "gain_lookback_bars": 42})
    assert "BIGVOL" in eligible          # 거래대금 통과
    assert "PUMP" in eligible            # 상승률 통과
    assert "SLEEPY" not in eligible      # 어느 쪽도 아님


def test_crypto_rank_filter_min_turnover_floor():
    """상승률 상위라도 거래대금 하한($) 미달이면 제외."""
    from invest_signal.scanner import _crypto_rank_eligible
    import numpy as np

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")

    def frame(price, vol, rising=False):
        c = pd.Series(np.linspace(price * (0.5 if rising else 1.0), price, 50), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                             "Volume": float(vol)})

    frames = {"BIGVOL": frame(1.0, 10_000_000),        # 24h 거래대금 ≈ 6천만 달러
              "THINPUMP": frame(2.0, 10, rising=True)}  # 상승률 1위지만 유동성 없음
    rcfg = {"volume_top": 1, "gain_top": 1, "gain_lookback_bars": 42,
            "min_turnover_usd": 5_000_000}
    eligible = _crypto_rank_eligible(frames, rcfg)
    assert "BIGVOL" in eligible
    assert "THINPUMP" not in eligible
