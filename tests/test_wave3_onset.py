"""3️⃣ 3파 초입(1h) — 역배열 240선 돌파 → 월 VWAP 위로 갔다가 ±5% 복귀 + 거래량 5배 봉."""
import numpy as np
import pandas as pd

from invest_signal import notify
from invest_signal.indicators import anchored_vwap_bands
from invest_signal.signals import wave3_onset as w3


def _frame(spike=6.0, back=None, crash=False):
    """10-01부터 1h — 10월 200→100, 11월 100→90 하락(역배열) 뒤 240선 돌파·눌림."""
    idx = pd.date_range("2026-10-01", periods=744 + 300, freq="1h", tz="UTC")
    n = len(idx)
    c = np.r_[np.linspace(200, 100, 744), np.linspace(100, 90, 300)]
    v = np.full(n, 1000.0)
    b = n - 20                                   # 돌파 봉
    c[b:b + 6] = [108, 110, 112, 111, 110, 108]  # 240선(≈93) 위로 · 월 VWAP(≈95) +5% 위
    v[b] = 1000.0 * spike
    c[b + 6:] = back if back is not None else 97.0    # VWAP ±5% 안으로 복귀
    hi, lo = c * 1.002, c * 0.998
    if crash:
        lo[b + 6] = 50.0                          # 돌파 전 저점 이탈
    return pd.DataFrame({"Open": c, "High": hi, "Low": lo, "Close": c, "Volume": v}, index=idx), b


def test_fixture_is_bearish_breakout_above_vwap():
    df, b = _frame()
    m = {k: df.Close.rolling(k).mean().iloc[b] for k in (120, 240, 480)}
    assert m[120] < m[240] < m[480] and df.Close.iloc[b] > m[240] > df.Close.iloc[b - 1]
    vw = anchored_vwap_bands(df, "M")[0].iloc[b + 6]
    assert abs(df.Close.iloc[b + 6] / vw - 1) <= 0.05


def test_fires_on_first_return_bar():
    df, b = _frame()
    p = w3.Params(grace_bars=10**6)
    got = w3._find(df, p)
    assert [(x[0], x[1]) for x in got] == [(b, b + 6)]
    assert got[0][3] >= 5


def test_needs_volume_burst_and_pre_low_and_return():
    df, _ = _frame(spike=3.0)
    assert w3._find(df, w3.Params(grace_bars=10**6)) == []
    df, _ = _frame(crash=True)
    assert w3._find(df, w3.Params(grace_bars=10**6)) == []
    df, _ = _frame(back=108.0)                    # 안 내려온다
    assert w3._find(df, w3.Params(grace_bars=10**6)) == []


def test_detect_only_recent_and_line_tags():
    df, b = _frame()
    cut = df.iloc[:b + 7]                         # 신호 봉이 마지막 마감봉
    ev = w3.detect(cut, "XUSDT")
    assert len(ev) == 1 and ev[0].bar_time == df.index[b + 6]
    assert w3.detect(df.iloc[:b + 12], "XUSDT") == []      # 지나간 신호는 안 낸다
    ev[0].detail.update(turnover_24h=5e6, gain_24h=0.1)
    line = notify._event_line(ev[0], "https://x", "", "crypto")
    assert "240선돌파" in line and "월VWAP" in line and "💥1h거래량 6배" in line


def test_two_breakouts_same_return_bar_one_line(monkeypatch):
    df, b = _frame()
    monkeypatch.setattr(w3, "_find", lambda d, p: [(b - 3, b + 6, 0.4, 6.0), (b, b + 6, 0.3, 6.0)])
    ev = w3.detect(df.iloc[:b + 7], "XUSDT")
    assert len(ev) == 1 and ev[0].detail["break_time"] == df.index[b - 3].isoformat()
