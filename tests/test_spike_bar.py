"""급등봉 시그널 테스트 — 조건 셋과 소급·평균 정의."""

import numpy as np
import pandas as pd

from invest_signal.signals import spike_bar
from invest_signal.signals.spike_bar import Params, detect


def _frame(n=60, vol=100.0):
    """1h봉(10-04~ 기본). 60봉 = vol_ma(24) + 여유."""
    idx = pd.date_range("2026-09-16", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"Open": 1.0, "High": 1.005, "Low": 0.995, "Close": 1.0,
                         "Volume": vol}, index=idx)


def _spike(df, i, body=0.10, mult=20.0, top=True, base_vol=100.0):
    """i번째 봉을 장대양봉으로 바꾼다. top=False면 고가에서 크게 밀려 마감."""
    o = 1.0
    c = o * (1 + body)
    hi = c if top else c * 1.20
    df.iloc[i, df.columns.get_loc("Open")] = o
    df.iloc[i, df.columns.get_loc("Close")] = c
    df.iloc[i, df.columns.get_loc("High")] = hi
    df.iloc[i, df.columns.get_loc("Low")] = o * 0.999
    df.iloc[i, df.columns.get_loc("Volume")] = base_vol * mult
    return df


def test_catches_the_big_green_bar():
    got = detect(_spike(_frame(), -1), "XUSDT")
    assert len(got) == 1
    e = got[0]
    assert e.signal == "spike_bar" and e.detail["label"] == "급등봉"
    assert round(e.detail["body"], 3) == 0.10
    assert round(e.detail["vol_mult"]) == 20
    assert e.detail["close_pos"] == 1.0


def test_body_alone_is_not_enough():
    """몸통 +8%짜리 15분봉은 흔하다 — 거래량이 안 터지면 사건이 아니다."""
    assert detect(_spike(_frame(), -1, mult=2.0), "XUSDT") == []


def test_volume_alone_is_not_enough():
    assert detect(_spike(_frame(), -1, body=0.02, mult=50.0), "XUSDT") == []


def test_long_upper_wick_is_dropped():
    """고가에서 한참 밀려 마감한 봉은 그 시점에 이미 끝난 사건이다."""
    assert detect(_spike(_frame(), -1, top=False), "XUSDT") == []


def test_average_excludes_the_judging_bar():
    """급등봉 자신이 평균을 밀어 올리면 **클수록 안 잡히는** 거꾸로 된 기준이 된다.

    96봉 평균에 자기 자신이 들어가면 거래량 100배짜리 봉의 배수가
    100 → 50 근처로 반토막 난다.
    """
    got = detect(_spike(_frame(), -1, mult=100.0), "XUSDT")
    assert got and round(got[0].detail["vol_mult"]) == 100


def test_grace_catches_the_previous_bar():
    """크론이 밀려 한 봉을 건너뛰어도 직전 1h봉까지 소급해 잡는다(grace_bars 1)."""
    df = _spike(_frame(), -2)                  # 한 시간 전 봉
    assert len(detect(df, "XUSDT")) == 1
    assert detect(df, "XUSDT", Params(grace_bars=0)) == []


def test_two_spikes_in_a_row_both_fire():
    df = _spike(_spike(_frame(), -2), -1)
    assert len(detect(df, "XUSDT")) == 2


def test_short_history_is_skipped():
    assert detect(_spike(_frame(n=20), -1), "XUSDT") == []


def test_never_tracked():
    """급등봉은 상태가 아니라 순간 — 추적 줄을 만들지 않는다."""
    df = _spike(_frame(), -1)
    assert spike_bar.still_active(df, detect(df, "XUSDT")[0]) is False


def test_zero_volume_history_does_not_divide_by_zero():
    df = _spike(_frame(vol=0.0), -1, base_vol=0.0)
    assert detect(df, "XUSDT") == []


def test_tracked_for_a_day_after_firing():
    """발화 뒤 하루는 추적 줄로 남는다 — 터진 종목이 값을 지키는지가 궁금하다."""
    df = _spike(_frame(), -12)                  # 12시간 전 봉
    ev = spike_bar.recent(df, "XUSDT")
    assert ev is not None and ev.detail["label"] == "급등봉"
    # 급등봉 종가 대비 지금 — 이게 추적 줄의 존재 이유다
    assert ev.detail["last_price"] == 1.0
    assert round(ev.detail["since"], 3) == round(1.0 / 1.10 - 1, 3)


def test_fresh_bar_is_not_also_tracked():
    """같은 봉이 신규(•)와 추적(↳) 두 칸에 동시에 실리면 같은 말을 두 번 한다."""
    df = _spike(_frame(), -1)
    assert len(detect(df, "XUSDT")) == 1
    assert spike_bar.recent(df, "XUSDT") is None


def test_older_than_a_day_drops_out():
    df = _spike(_frame(), -30)                  # 30시간 전 — 창 밖
    assert spike_bar.recent(df, "XUSDT") is None


def test_latest_spike_wins_when_there_are_several():
    df = _spike(_spike(_frame(), -20), -8)
    ev = spike_bar.recent(df, "XUSDT")
    assert ev is not None and ev.bar_time == df.index[-8]


def test_tracking_can_be_turned_off():
    df = _spike(_frame(), -12)
    assert spike_bar.recent(df, "XUSDT", Params(track_bars=0)) is None


def test_to_hourly_keeps_only_complete_hours():
    idx = pd.date_range("2026-09-16 00:00", periods=10, freq="15min", tz="UTC")   # 2시간 + 2봉
    df = pd.DataFrame({"Open": np.arange(10.0), "High": np.arange(10.0) + 1,
                       "Low": np.arange(10.0) - 1, "Close": np.arange(10.0) + 0.5,
                       "Volume": 1.0}, index=idx)
    h = spike_bar.to_hourly(df)
    assert list(h.index.hour) == [0, 1]                       # 02시는 아직 2봉뿐 — 버린다
    assert h.iloc[0].tolist() == [0.0, 4.0, -1.0, 3.5, 4.0]


def test_band_zone_between_ma480_and_ma960_inverse():
    """🪜 1h MA480 < 종가 < MA960 — 1,000봉 하락 뒤 반등 봉."""
    n = 1000
    c = np.r_[np.linspace(200, 100, n - 1), 140.0]
    idx = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"Close": c}, index=idx)
    z = spike_bar.band_zone(df, idx[-1])
    assert z["ma_fast"] < 140 < z["ma_slow"] and z["zone"]
    assert spike_bar.band_zone(df.assign(Close=np.r_[c[:-1], 200.0]), idx[-1])["zone"] is False
    assert spike_bar.band_zone(df.iloc[-900:], idx[-1]) is None            # MA960 못 구함
    assert spike_bar.band_zone(df, idx[-2] + pd.Timedelta(minutes=30)) is None


def test_scan_spike_hourly_marks_band_and_can_filter(monkeypatch):
    from invest_signal import scanner
    idx15 = pd.date_range("2026-09-16", periods=240, freq="15min", tz="UTC")
    base = pd.DataFrame({"Open": 1.0, "High": 1.005, "Low": 0.995, "Close": 1.0,
                         "Volume": 25.0}, index=idx15)
    hot = base.copy()                                         # 마지막 1h(15m 넷)가 장대양봉
    hot.iloc[-4:, hot.columns.get_loc("Volume")] = 600.0
    hot.iloc[-4, hot.columns.get_loc("Open")] = 1.0
    hot.iloc[-1, hot.columns.get_loc("Close")] = 1.12
    hot.iloc[-1, hot.columns.get_loc("High")] = 1.12
    t = idx15[-4]
    n = 1000
    h1 = pd.DataFrame({"Close": np.r_[np.linspace(2.0, 0.8, n - 1), 1.12]},
                      index=pd.date_range(end=t, periods=n, freq="1h"))
    monkeypatch.setattr(scanner.data_binance, "klines", lambda *a, **k: h1)
    cfg = {"signal": {"spike_bar": {"enabled": True}}}
    ev, _ = scanner._scan_spike(cfg, {"AUSDT": hot, "BUSDT": base}, {}, log=lambda *a: None)
    assert [e.symbol for e in ev] == ["AUSDT"] and ev[0].bar_time == t
    assert ev[0].detail["interval"] == "1h" and ev[0].detail["band_1h"]
    cfg["signal"]["spike_bar"]["band_filter"] = True
    h1.iloc[-1, 0] = 3.0                                      # 960선 위 — 사이가 아니다
    ev, _ = scanner._scan_spike(cfg, {"AUSDT": hot}, {}, log=lambda *a: None)
    assert ev == []


def test_notify_spike_line_shows_band_and_interval():
    from invest_signal.notify import format_events
    from invest_signal.signals import SignalEvent
    t = pd.Timestamp("2026-10-04T09:00:00Z")
    mk = lambda sym, band, v: SignalEvent(symbol=sym, signal="spike_bar", bar_time=t, price=1.0,
                                          detail={"label": "급등봉", "body": 0.1, "vol_mult": v,
                                                  "interval": "1h", "band_1h": band})
    rows = [ln for ln in format_events([mk("AUSDT", False, 50), mk("BUSDT", True, 12)], [], {})
            .splitlines() if ln.startswith("•")]
    assert ">B<" in rows[0] and "🪜1h480~960" in rows[0] and "🕒10-04 18:00 1h봉" in rows[0]
    assert ">A<" in rows[1] and "🪜" not in rows[1]
