"""11-25 — 급등봉 24h 대금 하한($5M)과 하따봉($20M) 하한을 따로 건다."""
import numpy as np
import pandas as pd

from invest_signal import scanner
from invest_signal.signals import spike_bar


def _frame():
    n = 200
    idx = pd.date_range("2026-11-20", periods=n, freq="15min", tz="UTC")
    o = np.full(n, 1.0); c = o.copy(); h = o.copy(); l = o.copy(); v = np.full(n, 100.0)
    # 마지막 봉 — 몸통 +30% · 고가 +30% · 거래량 50배 · 종가가 고가 → 급등봉이자 하따봉
    c[-1] = h[-1] = 1.3; v[-1] = 5_000.0
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def _cfg():
    return {"signal": {"spike_bar": {"enabled": True, "min_turnover_usd": 5_000_000},
                       "big_bar": {"enabled": True, "min_rise": 0.25, "min_turnover_usd": 20_000_000}}}


def _run(turn):
    ticker = {"XUSDT": {"quote_volume": turn, "change_pct": 0.3}}
    ev, _ = scanner._scan_spike(_cfg(), {"XUSDT": _frame()}, ticker, log=lambda *_: None)
    return sorted(e.signal for e in ev)


def test_spike_passes_at_5m_but_big_bar_needs_20m():
    assert _run(10_000_000) == [spike_bar.NAME]
    assert _run(25_000_000) == sorted([spike_bar.NAME, spike_bar.BIG_NAME])
    assert _run(3_000_000) == []


def test_track_floor_defaults_to_spike_floor():
    p = scanner._spike_params(_cfg())
    assert p.min_turnover_usd == 5_000_000 and p.track_min_turnover_usd is None


def test_board_only_message_has_title_once():
    from invest_signal import notify
    board = [dict(rank=1, symbol="MOVEUSDT", price=0.0055, gain_24h=0.23)]
    msg = notify.format_events([], [], {}, crypto_board=board, title="📈 <b>24h 상승률 TOP</b>")
    assert msg.count("24h 상승률 TOP") == 1 and "MOVE" in msg and "크립토 모멘텀" not in msg
