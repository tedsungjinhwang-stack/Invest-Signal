"""🚨 세력 이탈 — 현선갭·OI 되돌림. TAKE 09-23 실제 퍼프 데이터로 시점을 고정한다."""

import os

import numpy as np
import pandas as pd

from invest_signal.signals import whale_exit as we

DATA = os.path.join(os.path.dirname(__file__), "data")
KST = "Asia/Seoul"


def _take():
    f5 = pd.read_csv(os.path.join(DATA, "take_5m_20260923.csv"), index_col=0, parse_dates=True)
    px5 = pd.DataFrame({"Close": f5["Close"]})
    idx5 = pd.DataFrame({"Close": f5["Index"]})
    h = pd.read_csv(os.path.join(DATA, "take_1h_20260923.csv"), index_col=0, parse_dates=True)
    oi = pd.Series(h["OI_at_close"].to_numpy(), index=h.index + pd.Timedelta(hours=1))
    return px5, idx5, h[["Open", "High", "Low", "Close"]], oi


def _scan(hhmm):
    return pd.Timestamp(f"2026-09-23 {hhmm}", tz=KST).tz_convert("UTC")


def _upto(df, now):
    return df[df.index <= now]


def test_take_basis_gap_fires_before_the_crash():
    """TAKE: 21:35봉에 퍼프가 인덱스보다 11% 낮았다(가격 0.195 유지 중) — 22:02 스캔에서 잡는다.
    폭락은 22:30부터, 🔻CHoCH는 22:40봉이라 23:02 스캔이다."""
    px5, idx5, _, _ = _take()
    now = _scan("22:02")
    t, g = we.basis_gap(_upto(px5, now), _upto(idx5, now), now)
    assert t.tz_convert(KST).strftime("%H:%M") == "21:35" and -0.12 < g <= -0.10
    early = _scan("21:02")
    assert we.basis_gap(_upto(px5, early), _upto(idx5, early), early) is None   # 그땐 −6%


def test_take_oi_unwind_after_longs_left():
    """+266% 펌핑 동안 OI 1.68배 → 23시엔 시작 수준(1.00배), 가격은 상승분의 66%를 지킴."""
    _, _, k1, oi = _take()
    now = _scan("23:02")
    u = we.oi_unwind(_upto(k1, now), _upto(oi, now), now)
    assert u is not None
    assert u["pump"] > 2.5 and u["oi_peak"] > 1.6 and u["oi_now"] < 1.1 and u["keep"] > 0.6
    early = _scan("21:02")                                  # 아직 OI가 1.5배 — 롱이 남아 있다
    assert we.oi_unwind(_upto(k1, early), _upto(oi, early), early) is None


def test_detect_merges_both_and_respects_enabled():
    px5, idx5, k1, oi = _take()
    now = _scan("23:02")
    e = we.detect("TAKEUSDT", now, _upto(px5, now), _upto(idx5, now), _upto(k1, now),
                  _upto(oi, now))
    assert e.signal == "whale_exit" and e.detail["gap"] < -0.10 and e.detail["oi_now"] < 1.1
    assert e.price == px5["Close"][px5.index + pd.Timedelta(minutes=5) <= now].iloc[-1]
    off = we.Params(enabled=False)
    assert we.detect("TAKEUSDT", now, px5, idx5, k1, oi, off) is None
    # 소스가 현물 미러라 인덱스·OI가 없으면 아무것도 안 한다
    assert we.detect("TAKEUSDT", now, px5, None, k1, None) is None


def test_basis_gap_ignores_unclosed_bar_and_small_gaps():
    idx = pd.date_range("2026-09-23 00:00", periods=20, freq="5min", tz="UTC")
    px = pd.DataFrame({"Close": np.full(20, 1.0)}, index=idx)
    ix = pd.DataFrame({"Close": np.full(20, 1.05)}, index=idx)        # −4.8% — 기준 미달
    now = idx[-1] + pd.Timedelta(minutes=5)
    assert we.basis_gap(px, ix, now) is None
    ix.iloc[-1, 0] = 1.2                                              # 진행 중인 봉만 −17%
    assert we.basis_gap(px, ix, now - pd.Timedelta(minutes=1)) is None
    assert we.basis_gap(px, ix, now)[1] < -0.10


def test_notify_renders_whale_exit_line():
    from invest_signal.notify import format_events, fire_only_title
    from invest_signal.signals import SignalEvent
    e = SignalEvent(symbol="TAKEUSDT", signal="whale_exit",
                    bar_time=pd.Timestamp("2026-09-23T12:35:00Z"), price=0.1951,
                    detail={"label": "세력 이탈", "gap": -0.11, "interval": "5m",
                            "oi_peak": 1.68, "oi_now": 1.0, "oi_pump": 2.66, "oi_keep": 0.66,
                            "rank": 1, "gain_24h": 2.2, "turnover_24h": 4.0e8})
    out = format_events([e], [], {})
    assert "🚨 <b>세력 이탈</b>" in out
    line = [ln for ln in out.splitlines() if ">TAKE<" in ln][0]
    assert "🕒09-23 21:35 5m봉" in line and "현선갭 -11%(선물<현물)" in line
    assert "OI되돌림 1.68x→1.00x(펌핑 +266% · 유지 66%)" in line and "1위" in line
    assert fire_only_title([e]) == "🚨 <b>세력 이탈</b>"


def test_scanner_wires_whale_exit_for_watched_coins(monkeypatch):
    """⚡ 감시 종목이면 5m 퍼프·5m 인덱스·1h OI를 받아 🚨 줄을 만든다. 현물 미러면 안 받는다."""
    from invest_signal import scanner

    SHAPE = {"5m": (1000, "5min"), "15m": (400, "15min"), "1h": (600, "1h"), "4h": (750, "4h")}
    end = pd.Timestamp.now(tz="UTC").floor("5min")

    def frame(interval):
        k, freq = SHAPE[interval]
        idx = pd.date_range(end=end - pd.Timedelta(freq), periods=k, freq=freq)
        c = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1000.0})

    calls = []
    monkeypatch.setattr(scanner.data_binance, "klines",
                        lambda s, sym, src, interval, limit=None, include_live=False:
                        frame(interval))
    monkeypatch.setattr(scanner.data_binance, "index_klines",
                        lambda s, sym, src, interval, limit=None:
                        calls.append("idx") or frame("5m") * 1.2)     # 퍼프가 17% 낮다
    monkeypatch.setattr(scanner.data_binance, "open_interest_hist",
                        lambda *a, **k: calls.append("oi") or None)
    monkeypatch.setattr(scanner.data_binance, "fetch_all", lambda *a, **k: {})
    monkeypatch.setattr(scanner, "_crypto_ticker", lambda *a, **k: {
        "XUSDT": {"change_pct": 0.3, "quote_volume": 50_000_000.0, "last": 1.0}})
    cfg = {"crypto": {"enabled": True},
           "signal": {"leader_break": {"enabled": True, "require_aligned": False,
                                       "exhausted_filter": False},
                      "whale_exit": {"enabled": True}}}
    for source, want in (("fapi", 1), ("spot_mirror", 0)):
        calls.clear()
        monkeypatch.setattr(scanner.data_binance, "resolve_source",
                            lambda *a, _s=source, **k: (_s, ["XUSDT"]))
        events, _, _ = scanner.scan_crypto(cfg, [], log=lambda *a: None)
        got = [e for e in events if e.signal == "whale_exit"]
        assert len(got) == want, source
        if want:
            assert got[0].detail["gap"] < -0.10 and got[0].detail["rank"] == 1
            assert calls == ["idx", "oi"]
        else:
            assert calls == []
