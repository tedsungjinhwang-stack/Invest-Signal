"""🔻 하락 CHoCH 경고 — 15m 스윙 구조 붕괴 판정 테스트."""

import numpy as np
import pandas as pd

from invest_signal.signals import choch_warn as cw


def _frame(closes, start="2026-09-20"):
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(np.asarray(closes, float), index=idx)
    return pd.DataFrame({"Open": c, "High": c * 1.002, "Low": c * 0.998, "Close": c,
                         "Volume": 1.0}, index=idx)


def _zigzag_up_then_break():
    """고점↑·저점↑ 지그재그 상승 → 마지막에 직전 스윙 저점 아래로 마감."""
    legs = []
    base = 100.0
    for k in range(5):                                    # 오르고(12봉) 조금 눌리고(8봉)
        legs += list(np.linspace(base, base + 6, 12)) + list(np.linspace(base + 6, base + 3, 8))
        base += 3
    legs += list(np.linspace(base, base + 4, 10))          # 마지막 상승
    legs += list(np.linspace(base + 4, base - 5, 10))      # 직전 스윙 저점 아래로 붕괴
    return _frame(legs)


def test_choch_fires_when_uptrend_structure_breaks():
    df = _zigzag_up_then_break()
    got = cw.choch_bars(df, 3)
    assert got, "상승 구조 뒤 스윙 저점 이탈을 못 잡았다"
    t, low, high = got[-1]
    assert df["Close"].iloc[t] < low < high


def test_no_choch_in_straight_uptrend():
    df = _frame(np.linspace(100, 150, 200))
    assert cw.choch_bars(df, 3) == []


def test_detect_only_recent_closed_bars():
    df = _zigzag_up_then_break()
    t, _, _ = cw.choch_bars(df, 3)[-1]
    p = cw.Params(pivot_bars=3, grace_bars=4)
    now = df.index[t] + pd.Timedelta(minutes=15)          # 그 봉이 막 마감된 시각
    got = cw.detect(df.iloc[:t + 1], "XUSDT", now, p)
    assert len(got) == 1 and got[0].detail["broken_low"] > got[0].price
    # 진행 중인 봉(아직 마감 전)이면 안 본다
    assert cw.detect(df.iloc[:t + 1], "XUSDT", now - pd.Timedelta(minutes=5), p) == []
    # 오래전 이탈은 grace 밖
    later = df.index[t] + pd.Timedelta(hours=5)
    assert cw.detect(df.iloc[:t + 1], "XUSDT", later, p) == []
    assert cw.detect(df, "XUSDT", now, cw.Params(enabled=False)) == []


def test_above_long_ma_uses_closed_4h_bars():
    idx = pd.date_range("2026-03-01", periods=1000, freq="4h", tz="UTC")
    up = pd.DataFrame({"Close": np.geomspace(50, 100, 1000)}, index=idx)
    down = pd.DataFrame({"Close": np.geomspace(100, 50, 1000)}, index=idx)
    when = idx[-1] + pd.Timedelta(hours=4)
    assert cw.above_long_ma(up, when) > 0
    assert cw.above_long_ma(down, when) is None
    assert cw.above_long_ma(up.iloc[:900], when) is None      # 960봉이 안 된다


def test_notify_renders_choch_section():
    from invest_signal.notify import format_events
    from invest_signal.signals import SignalEvent
    e = SignalEvent(symbol="QNTUSDT", signal="choch_warn",
                    bar_time=pd.Timestamp("2026-09-30T05:15:00Z"), price=280.0,
                    detail={"label": "하락 CHoCH", "broken_low": 284.5, "swing_high": 300.0,
                            "from_high": -0.0667, "ma_long_gap": 1.85, "ma_long": 960,
                            "rank": 2, "gain_24h": 0.12, "turnover_24h": 1.5e9})
    out = format_events([e], [], {})
    assert "🔻 <b>하락 CHoCH</b>" in out
    line = [ln for ln in out.splitlines() if ">QNT<" in ln][0]
    assert "🕒09-30 14:15" in line and "저점 284.5 이탈" in line and "고점대비 -6.7%" in line
    assert "4h960선 +185%" in line and "2위" in line
