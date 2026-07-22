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
