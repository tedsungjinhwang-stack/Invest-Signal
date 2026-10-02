"""🔥🔥 — 🔥 줄에 선물 수급 조건(현선갭·OI)을 더 본다."""

import numpy as np
import pandas as pd

from invest_signal.signals import SignalEvent
from invest_signal.signals import fire2 as f2

NOW = pd.Timestamp("2026-10-02T09:02:00Z")


def _frames(gaps):
    """마지막 마감봉이 08:45인 15m 퍼프·인덱스 — gaps는 오래된 → 최근 순 현선갭."""
    n = len(gaps)
    idx = pd.date_range(end="2026-10-02T08:45:00Z", periods=n, freq="15min")
    px = pd.DataFrame({"Close": np.full(n, 1.0)}, index=idx)
    ix = pd.DataFrame({"Close": 1.0 / (1 + np.asarray(gaps, float))}, index=idx)
    return px, ix


def test_gap_stats_now_and_24h_mean():
    px, ix = _frames([0.002] * 95 + [-0.003])
    g = f2.gap_stats(px, ix, NOW)
    assert abs(g["gap_now"] + 0.003) < 1e-9
    assert abs(g["gap_mean24"] - (0.002 * 95 - 0.003) / 96) < 1e-9
    short = f2.gap_stats(*_frames([0.0] * 30), NOW)
    assert "gap_mean24" not in short                       # 50봉 미만이면 평균은 안 낸다
    assert f2.gap_stats(px, None, NOW) is None             # 현물 미러 소스


def test_gap_stats_needs_latest_bar_on_both_sides():
    px, ix = _frames([0.0] * 96)
    assert f2.gap_stats(px, ix.iloc[:-1], NOW) is None     # 인덱스가 한 봉 늦으면 안 잰다


def test_oi_change_4h_with_fresh_snapshots():
    t = pd.date_range(end="2026-10-02T09:00:00Z", periods=60, freq="5min")
    oi = pd.Series(np.linspace(110, 100, 60), index=t)
    v = f2.oi_change(oi, NOW)
    assert v is not None and -0.08 < v < -0.06              # 4h 전 ≈ 108 → 지금 100
    assert f2.oi_change(oi, NOW + pd.Timedelta(hours=1)) is None   # 마지막 스냅샷이 묵었다
    assert f2.oi_change(oi.iloc[-10:], NOW) is None         # 4h 전 값이 없다


def test_mark_per_signal():
    p = f2.Params()
    assert f2.mark("vwap_onset", {"gap_now": 0.01, "gap_mean24": -0.0001}, None, p)
    assert not f2.mark("vwap_onset", {"gap_now": -0.01, "gap_mean24": 0.0002}, None, p)
    assert f2.mark("leader_break", None, -0.006, p)
    assert not f2.mark("leader_break", None, -0.004, p)
    assert f2.mark("wave_setup", {"gap_now": -0.0012}, None, p)
    assert not f2.mark("wave_setup", {"gap_now": -0.0005, "gap_mean24": -0.01}, None, p)
    assert not f2.mark("spike_bar", {"gap_now": -1}, -1, p)
    assert not f2.mark("leader_break", None, None, p)      # 못 재면 안 붙인다
    assert not f2.mark("leader_break", None, -0.5, f2.Params(enabled=False))


def _ev(sym, signal, **d):
    label = {"vwap_onset": "상승초입", "wave_setup": "파동",
             "leader_break": "크립토 모멘텀 눌림목/이탈"}[signal]
    return SignalEvent(symbol=sym, signal=signal, bar_time=NOW, price=1.0,
                       detail={"label": label, **d})


def test_notify_puts_double_fire_first():
    from invest_signal.notify import format_events
    evs = [_ev("AUSDT", "vwap_onset", fire=True), _ev("BUSDT", "vwap_onset", fire=True, fire2=True),
           _ev("CUSDT", "vwap_onset")]
    out = format_events(evs, [], {})
    rows = [ln for ln in out.splitlines() if ln.startswith("•")]
    assert ">B<" in rows[0] and "🔥🔥" in rows[0]
    assert ">A<" in rows[1] and "🔥" in rows[1] and "🔥🔥" not in rows[1]
    assert ">C<" in rows[2]


def test_scanner_marks_only_fire_rows(monkeypatch):
    from invest_signal import scanner
    calls = []

    def frame(v):
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta("15min"),
                            periods=100, freq="15min")
        return pd.DataFrame({"Close": np.full(100, v)}, index=idx)

    def fake_index(s, sym, src, interval, limit=None):
        calls.append(("idx", sym))
        return frame(1.0 if sym == "AUSDT" else 0.99)       # B는 선물이 1% 비싸다

    def fake_px(s, sym, src, interval, limit=None, include_live=False):
        return frame(1.0)

    def fake_oi(s, sym, src, period, limit=None):
        calls.append(("oi", sym))
        t = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("5min"), periods=60, freq="5min")
        return pd.Series(np.linspace(110, 100, 60), index=t)

    monkeypatch.setattr(scanner.data_binance, "index_klines", fake_index)
    monkeypatch.setattr(scanner.data_binance, "klines", fake_px)
    monkeypatch.setattr(scanner.data_binance, "open_interest_hist", fake_oi)
    a = _ev("AUSDT", "vwap_onset", fire=True)          # 갭 0 → 🔥🔥
    b = _ev("BUSDT", "vwap_onset", fire=True)          # 선물이 1% 비쌈 → 🔥 그대로
    c = _ev("CUSDT", "leader_break", fire=True)        # OI 줄어듦 → 🔥🔥
    d = _ev("DUSDT", "vwap_onset")                     # 🔥 아님 → 안 본다
    cfg = {"signal": {"fire2": {"enabled": True}}}
    scanner._mark_fire2(cfg, "fapi", [a, b, c, d], {}, workers=2, log=lambda *x: None)
    assert a.detail.get("fire2") and not b.detail.get("fire2") and c.detail.get("fire2")
    assert "fire2" not in d.detail
    assert sorted(calls) == [("idx", "AUSDT"), ("idx", "BUSDT"), ("oi", "CUSDT")]
    calls.clear()
    scanner._mark_fire2(cfg, "spot_mirror", [a], {}, log=lambda *x: None)
    scanner._mark_fire2({}, "fapi", [a], {}, log=lambda *x: None)
    assert calls == []
