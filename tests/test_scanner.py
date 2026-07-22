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
