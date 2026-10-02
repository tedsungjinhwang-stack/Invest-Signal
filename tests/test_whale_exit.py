"""🚨 세력 이탈 — 현선갭(퍼프 ÷ 인덱스). TAKE 09-23 실제 퍼프 15m로 시점을 고정한다."""

import os

import numpy as np
import pandas as pd

from invest_signal.signals import whale_exit as we

DATA = os.path.join(os.path.dirname(__file__), "data")
KST = "Asia/Seoul"


def _take():
    f = pd.read_csv(os.path.join(DATA, "take_15m_20260923.csv"), index_col=0, parse_dates=True)
    return pd.DataFrame({"Close": f["Close"]}), pd.DataFrame({"Close": f["Index"]})


def _scan(hhmm):
    return pd.Timestamp(f"2026-09-23 {hhmm}", tz=KST).tz_convert("UTC")


def _upto(df, now):
    return df[df.index <= now]


def test_take_fires_before_the_crash_via_pump_branch():
    """TAKE: 21:30봉 현선갭 −7%(24h +227% 펌핑 중) → 22:02 스캔(0.195)에서 잡는다.
    10%만 봤으면 23:02(폭락 시작 뒤)였다. 21:02엔 아직 5% 미만."""
    px, ix = _take()
    now = _scan("22:02")
    e = we.detect(_upto(px, now), _upto(ix, now), "TAKEUSDT", now)
    assert e.bar_time.tz_convert(KST).strftime("%H:%M") == "21:30"
    assert -0.10 < e.detail["gap"] <= -0.05 and e.detail["pump_24h"] > 2
    assert abs(e.price - 0.195) < 1e-3                  # 마지막 마감봉(21:45) 종가
    only10 = we.Params(gap_pump_min=0.10)
    assert we.detect(_upto(px, now), _upto(ix, now), "TAKEUSDT", now, only10) is None
    early = _scan("21:02")
    assert we.detect(_upto(px, early), _upto(ix, early), "TAKEUSDT", early) is None


def test_gap_10_needs_no_pump_but_5_does():
    idx = pd.date_range("2026-09-20", periods=120, freq="15min", tz="UTC")
    flat = pd.DataFrame({"Close": np.full(120, 1.0)}, index=idx)
    now = idx[-1] + pd.Timedelta(minutes=15)
    ix = flat.copy()
    ix.iloc[-1, 0] = 1.07                              # −6.5% 갭, 24h 상승 0%
    assert we.detect(flat, ix, "X", now) is None
    ix.iloc[-1, 0] = 1.12                              # −10.7% 갭 — 펌핑 없어도
    assert we.detect(flat, ix, "X", now).detail["gap"] < -0.10
    pumped = flat.copy()
    pumped.iloc[-30:, 0] = 1.4                         # 24h +40%
    ix2 = pumped.copy()
    ix2.iloc[-1, 0] = 1.4 * 1.07
    e = we.detect(pumped, ix2, "X", now)
    assert e is not None and abs(e.detail["pump_24h"] - 0.4) < 1e-9


def test_ignores_unclosed_bar_old_bars_and_disabled():
    idx = pd.date_range("2026-09-20", periods=120, freq="15min", tz="UTC")
    px = pd.DataFrame({"Close": np.full(120, 1.0)}, index=idx)
    ix = px.copy()
    ix.iloc[-1, 0] = 1.2
    now = idx[-1] + pd.Timedelta(minutes=15)
    assert we.detect(px, ix, "X", now - pd.Timedelta(minutes=1)) is None     # 진행 중
    ix2 = px.copy()
    ix2.iloc[-7, 0] = 1.2                                                    # 75분보다 전
    assert we.detect(px, ix2, "X", now) is None
    assert we.detect(px, ix, "X", now, we.Params(enabled=False)) is None
    assert we.detect(px, None, "X", now) is None                            # 현물 미러 소스


def test_notify_renders_whale_exit_line():
    from invest_signal.notify import fire_only_title, format_events
    from invest_signal.signals import SignalEvent
    e = SignalEvent(symbol="TAKEUSDT", signal="whale_exit",
                    bar_time=pd.Timestamp("2026-09-23T12:30:00Z"), price=0.1948,
                    detail={"label": "세력 이탈", "gap": -0.07, "interval": "15m",
                            "pump_24h": 2.27, "turnover_24h": 4.0e8})
    out = format_events([e], [], {})
    assert "🚨 <b>세력 이탈</b>" in out
    line = [ln for ln in out.splitlines() if ">TAKE<" in ln][0]
    assert "🕒09-23 21:30 15m봉" in line and "현선갭 -7.0%(선물<현물)" in line
    assert "그때 24h +227%" in line
    assert fire_only_title([e]) == "🚨 <b>세력 이탈</b>"


def test_scanner_checks_every_perp_and_skips_rearmed(monkeypatch, tmp_path):
    """⚡ 감시와 무관하게 전 종목을 본다. 24h 안에 알린 종목은 인덱스도 안 받는다."""
    from invest_signal import scanner
    from invest_signal.state import AlertState

    end = pd.Timestamp.now(tz="UTC").floor("15min")
    idx = pd.date_range(end=end - pd.Timedelta(minutes=15), periods=120, freq="15min")
    frame = pd.DataFrame({"Close": np.full(120, 1.0)}, index=idx)
    asked = []

    def fake_index(s, sym, src, interval, limit=None):
        asked.append(sym)
        return frame * (1.15 if sym == "AUSDT" else 1.0)

    monkeypatch.setattr(scanner.data_binance, "index_klines", fake_index)
    state = AlertState(str(tmp_path / "s.json"))
    state.mark("CUSDT|whale_exit|2026-09-30T00:00:00+00:00")
    cfg = {"signal": {"whale_exit": {"enabled": True}}}
    frames = {"AUSDT": frame, "BUSDT": frame, "CUSDT": frame}
    got = scanner._scan_whale_exit(cfg, "fapi", frames,
                                   {"AUSDT": {"change_pct": 0.1, "quote_volume": 2e6}},
                                   state, workers=2, log=lambda *a: None)
    assert [e.symbol for e in got] == ["AUSDT"] and got[0].detail["turnover_24h"] == 2e6
    assert sorted(asked) == ["AUSDT", "BUSDT"]
    assert scanner._scan_whale_exit(cfg, "spot_mirror", frames, {}, state) == []
    assert scanner._scan_whale_exit({}, "fapi", frames, {}, state) == []
