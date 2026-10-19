"""📅 일봉 눌림(10-19) — 볼밴 하단 · 상승 일봉 단기선 터치/이탈, ⚡·🚀 풀 50일."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from invest_signal.indicators import supertrend_full
from invest_signal.signals import daily_dip as dd
from invest_signal.state import AlertState


def _frame(c, spread=0.01):
    idx = pd.date_range("2026-06-01", periods=len(c), freq="1D", tz="UTC")
    c = np.asarray(c, float)
    return pd.DataFrame({"Open": c, "High": c * (1 + spread), "Low": c * (1 - spread), "Close": c,
                         "Volume": 1.0}, index=idx)


def test_short_line_touch_and_break_from_above():
    df = _frame(np.linspace(100, 200, 80), 0.005)
    line = supertrend_full(df.iloc[:-1], 22, 3.0)["line"].to_numpy()[-1]
    touch = df.copy()
    touch.iloc[-1, touch.columns.get_loc("Low")] = line * 0.999
    touch.iloc[-1, touch.columns.get_loc("Close")] = line * 1.02
    assert [e.detail["stage"] for e in dd.detect(touch, "X")] == [dd.ST_TOUCH]
    brk = df.copy()
    brk.iloc[-1, brk.columns.get_loc("Low")] = line * 0.95
    brk.iloc[-1, brk.columns.get_loc("Close")] = line * 0.97
    got = dd.detect(brk, "X")
    assert dd.ST_BREAK in [e.detail["stage"] for e in got]
    assert dd.detect(df, "X") == []                                   # 안 닿음


def test_bb_lower_touch():
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 60)
    df = _frame(c, 0.002)
    mid = df.Close.rolling(20).mean()
    sd = df.Close.rolling(20).std(ddof=0)
    lower = (mid - 2 * sd).iloc[-1]
    df.iloc[-2, df.columns.get_loc("Low")] = max(df.Low.iloc[-2], (mid - 2 * sd).iloc[-2] * 1.01)
    df.iloc[-1, df.columns.get_loc("Low")] = lower * 0.99
    got = dd.detect(df, "X")
    assert dd.BB_LOWER in [e.detail["stage"] for e in got]
    e = [e for e in got if e.detail["stage"] == dd.BB_LOWER][0]
    assert e.dedup_key.endswith("|볼밴하단") and e.bar_time == df.index[-1]


def test_pool_keeps_50_days(tmp_path):
    st = AlertState(str(tmp_path / "s.json"))
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    st.pool_add("AUSDT", "⚡", now - timedelta(days=40))
    st.pool_add("AUSDT", "🚀", now - timedelta(days=3))
    st.pool_add("BUSDT", "🚀", now - timedelta(days=55))
    p = st.pool(50, now=now)
    assert set(p) == {"AUSDT"} and p["AUSDT"]["kind"] == "⚡🚀"
    assert p["AUSDT"]["first"] < p["AUSDT"]["t"]


def test_notify_renders_daily_dip():
    from invest_signal import notify
    from invest_signal.signals import SignalEvent
    e = SignalEvent(symbol="AGTUSDT", signal="daily_dip", bar_time=pd.Timestamp("2026-10-19", tz="UTC"),
                    price=0.03, detail={"label": "일봉 눌림", "stage": dd.ST_TOUCH, "level": 0.0291,
                                        "pool_kind": "⚡🚀", "pool_days": 4, "gain_24h": -0.08})
    out = notify.format_events([e], [], {})
    assert "📅 <b>일봉 눌림</b>" in out
    line = [ln for ln in out.splitlines() if ">AGT<" in ln][0]
    assert "🔁일봉단기선터치" in line and "⚡🚀 4일째" in line and "24h -8.0%" in line
    assert notify.fire_only([e], ("daily_dip",)) == [e]


def test_seed_pool_from_alert_history(tmp_path):
    st = AlertState(str(tmp_path / "s.json"))
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    st.mark("AUSDT|leader_break|2026-10-01T00:00:00+00:00")
    st.mark("BUSDT|spike_bar|2026-10-10T00:15:00+00:00|장기선터치")
    st.mark("CUSDT|choch_warn|2026-10-10T00:00:00+00:00")
    assert st.seed_pool(50, now=now) == 2
    p = st.pool(50, now=now)
    assert p["AUSDT"]["kind"] == "⚡" and p["BUSDT"]["kind"] == "🚀" and "CUSDT" not in p
    assert st.seed_pool(50, now=now) == 0                      # 이미 있으면 안 채운다
