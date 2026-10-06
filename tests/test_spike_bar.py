"""급등봉 시그널 테스트 — 조건 셋과 소급·평균 정의."""

import numpy as np
import pandas as pd

from invest_signal.signals import spike_bar
from invest_signal.signals.spike_bar import Params, detect


def _frame(n=200, vol=100.0):
    idx = pd.date_range("2026-09-16", periods=n, freq="15min", tz="UTC")
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


def test_grace_catches_bars_from_earlier_in_the_hour():
    """15분봉이라 소급이 없으면 네 봉 중 셋을 놓친다 — 스캔은 한 시간에 한 번이다."""
    df = _spike(_frame(), -4)                  # 한 시간 전 봉
    assert len(detect(df, "XUSDT")) == 1
    assert detect(df, "XUSDT", Params(grace_bars=0)) == []


def test_two_spikes_in_one_hour_both_fire():
    df = _spike(_spike(_frame(), -4), -1)
    assert len(detect(df, "XUSDT")) == 2


def test_short_history_is_skipped():
    assert detect(_spike(_frame(n=50), -1), "XUSDT") == []


def test_never_tracked():
    """급등봉은 상태가 아니라 순간 — 추적 줄을 만들지 않는다."""
    df = _spike(_frame(), -1)
    assert spike_bar.still_active(df, detect(df, "XUSDT")[0]) is False


def test_zero_volume_history_does_not_divide_by_zero():
    df = _spike(_frame(vol=0.0), -1, base_vol=0.0)
    assert detect(df, "XUSDT") == []


def test_tracked_for_a_day_after_firing():
    """발화 뒤 하루는 추적 줄로 남는다 — 터진 종목이 값을 지키는지가 궁금하다."""
    df = _spike(_frame(), -50)                  # 12시간 전 봉
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


def test_tracked_for_three_days_then_drops_out():
    """10-06~ 추적 3일(288봉). 62시간 전 급등은 남고 75시간 전은 빠진다."""
    assert spike_bar.recent(_spike(_frame(n=500), -250), "XUSDT") is not None
    assert spike_bar.recent(_spike(_frame(n=500), -300), "XUSDT") is None
    one_day = Params(track_bars=96)
    assert spike_bar.recent(_spike(_frame(n=500), -120), "XUSDT", one_day) is None


def test_latest_spike_wins_when_there_are_several():
    df = _spike(_spike(_frame(), -80), -30)
    ev = spike_bar.recent(df, "XUSDT")
    assert ev is not None and ev.bar_time == df.index[-30]


def test_tracking_can_be_turned_off():
    df = _spike(_frame(), -50)
    assert spike_bar.recent(df, "XUSDT", Params(track_bars=0)) is None


def _pump_then_fall(n_flat=200, fall=60):
    """평평 → 급등봉 하나 → 천천히 하락. 15m 장기 수퍼트렌드(30×6)는 급등 뒤 아래에서 따라온다."""
    df = _frame(n=n_flat + 1 + fall)
    _spike(df, n_flat, body=0.30, mult=40.0)
    c = np.linspace(1.30, 0.90, fall)
    for k, v in enumerate(c):
        i = n_flat + 1 + k
        df.iloc[i, df.columns.get_loc("Open")] = v * 1.002
        df.iloc[i, df.columns.get_loc("High")] = v * 1.004
        df.iloc[i, df.columns.get_loc("Low")] = v * 0.996
        df.iloc[i, df.columns.get_loc("Close")] = v
    return df


def test_long_touch_fires_once_on_first_touch_after_spike():
    from invest_signal.indicators import supertrend_full
    df = _pump_then_fall()
    st = supertrend_full(df, 30, 6.0)["line"].to_numpy()
    lo, hi = df["Low"].to_numpy(), df["High"].to_numpy()
    touch = [t for t in range(201, len(df)) if lo[t] <= st[t] <= hi[t]]
    assert touch, "합성 데이터가 선을 안 건드린다"
    t0 = touch[0]
    ev = spike_bar.long_touch(df.iloc[:t0 + 1], "XUSDT")
    assert ev is not None and ev.bar_time == df.index[t0]
    assert ev.detail["stage"] == "장기선터치" and ev.detail["spike_time"] == df.index[200]
    assert ev.detail["since"] < 0 and ev.dedup_key.endswith("|장기선터치")
    # 첫 터치에서 grace(4봉)보다 더 지나면 다시 안 나온다 — 급등봉 하나당 한 번
    assert spike_bar.long_touch(df.iloc[:t0 + 10], "XUSDT") is None
    assert spike_bar.long_touch(df.iloc[:t0 + 1], "XUSDT", Params(touch_enabled=False)) is None


def test_notify_renders_long_touch_and_st30_only_keeps_it():
    from invest_signal import notify
    from invest_signal.signals import SignalEvent
    t = pd.Timestamp("2026-10-06T05:00:00Z")
    ev = SignalEvent(symbol="AUSDT", signal="spike_bar", bar_time=t, price=1.1,
                     detail={"label": "급등봉", "stage": "장기선터치", "interval": "15m",
                             "line": 1.08, "line_up": True,
                             "spike_time": pd.Timestamp("2026-10-05T03:00:00Z"),
                             "spike_price": 1.3, "body": 0.3, "vol_mult": 40.0, "since": -0.154})
    line = [ln for ln in notify.format_events([ev], [], {}).splitlines() if ">A<" in ln][0]
    assert "🧱15m장기선터치 🕒10-06 14:00" in line and "상승·지지" in line
    assert "급등 🕒10-05 12:00 몸통 +30%" in line and "급등 후 -15%" in line
    plain = SignalEvent(symbol="BUSDT", signal="spike_bar", bar_time=t, price=1.0,
                        detail={"label": "급등봉", "body": 0.1, "vol_mult": 20.0})
    assert notify.st30_only([ev, plain]) == [ev]
