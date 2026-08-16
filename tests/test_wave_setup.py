"""파동 시그널 — 4h 돌파(ABC)와 일봉 터치(임펄스)."""

import numpy as np
import pandas as pd

from invest_signal.indicators import supertrend_full
from invest_signal.signals import wave_setup
from invest_signal.signals.wave_setup import ABC, IMPULSE, Params, detect, still_active


def _frame(closes, span=None, start="2026-01-01"):
    """종가 배열 → 4h OHLCV. span을 주면 고·저를 종가 ±span으로 벌린다."""
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, dtype=float)
    hi = c + (span if span is not None else c * 0.002)
    lo = c - (span if span is not None else c * 0.002)
    return pd.DataFrame({"Open": c, "High": hi, "Low": lo, "Close": c,
                         "Volume": np.full(len(c), 1000.0)}, index=idx)


def _falling_then_pop(n=200, pop=0.05):
    """길게 흘러내리다 마지막 봉에서 확 튀는 시리즈 — 단기만 뒤집기 좋은 모양."""
    base = list(np.linspace(100.0, 40.0, n - 1))
    return base + [base[-1] * (1 + pop)]


def test_abc_fires_when_only_fast_flips_up():
    """ⓐ 둘 다 하락이던 상태에서 단기만 상승 전환한 봉에서 발화한다."""
    df = _frame(_falling_then_pop())
    p = Params(impulse_enabled=False)
    fast = supertrend_full(df, p.fast_period, p.fast_mult)["dir"]
    slow = supertrend_full(df, p.slow_period, p.slow_mult)["dir"]
    # 전제 확인 — 직전 봉은 둘 다 하락, 마지막 봉은 단기만 상승
    assert fast.iloc[-2] < 0 and slow.iloc[-2] < 0
    assert fast.iloc[-1] > 0 and slow.iloc[-1] < 0

    evs = detect(df, "XUSDT", p)
    assert len(evs) == 1
    e = evs[0]
    assert e.signal == "wave_setup" and e.detail["stage"] == ABC
    assert e.detail["interval"] == "4h"
    assert e.bar_time == df.index[-1]
    assert e.detail["fast_line"] < e.price      # 돌파했으니 종가가 선 위


def test_abc_silent_while_both_still_down():
    """단기가 아직 안 뒤집혔으면 발화하지 않는다 — 대기 상태."""
    df = _frame(list(np.linspace(100.0, 40.0, 200)))
    assert detect(df, "XUSDT", Params(impulse_enabled=False)) == []


def test_abc_silent_when_long_term_also_flipped_up():
    """장기까지 상승으로 돌아섰으면 ⓐ의 전제(장기는 아직 하락)가 깨진다."""
    df = _frame(_falling_then_pop(pop=0.30))    # 너무 크게 튀어 장기도 뒤집힌다
    p = Params(impulse_enabled=False)
    slow = supertrend_full(df, p.slow_period, p.slow_mult)["dir"]
    assert slow.iloc[-1] > 0                    # 전제 확인
    assert detect(df, "XUSDT", p) == []


def test_abc_does_not_repeat_while_fast_stays_up():
    """뒤집힌 뒤 계속 상승이면 그 다음 봉들에서는 다시 알리지 않는다."""
    closes = _falling_then_pop()
    p = Params(impulse_enabled=False)
    df = _frame(closes + [closes[-1] * 1.01, closes[-1] * 1.02])
    fast = supertrend_full(df, p.fast_period, p.fast_mult)["dir"]
    assert fast.iloc[-1] > 0 and fast.iloc[-2] > 0   # 계속 상승 중
    assert detect(df, "XUSDT", p) == []              # 전환 봉은 grace 밖


def test_impulse_fires_on_daily_touch():
    """ⓑ 둘 다 하락인 일봉에서 캔들이 단기선을 터치하면 발화한다."""
    n = 6 * 120
    closes = list(np.linspace(100.0, 50.0, n))
    df = _frame(closes)
    p = Params(abc_enabled=False)
    daily = wave_setup._daily(df)
    fast = supertrend_full(daily, p.fast_period, p.fast_mult)
    # 마지막 날의 고가를 단기선 위로 끌어올려 '터치'를 만든다 (종가는 그대로)
    line = float(fast["line"].iloc[-1])
    day = daily.index[-1]
    df.loc[df.index.normalize() == day, "High"] = line + 1.0

    evs = detect(df, "XUSDT", p)
    assert len(evs) == 1 and evs[0].detail["stage"] == IMPULSE
    assert evs[0].detail["interval"] == "1d"
    assert evs[0].bar_time == day


def test_impulse_silent_when_candle_never_reaches_the_line():
    """선까지 못 닿은 날은 발화하지 않는다."""
    df = _frame(list(np.linspace(100.0, 50.0, 6 * 120)))
    assert detect(df, "XUSDT", Params(abc_enabled=False)) == []


def test_daily_drops_the_unfinished_last_day():
    """4h봉 6개가 안 찬 마지막 날은 일봉으로 치지 않는다."""
    df = _frame(list(np.linspace(100.0, 50.0, 6 * 10 + 2)))   # 마지막 날 2봉뿐
    daily = wave_setup._daily(df)
    assert len(daily) == 10
    assert daily.index[-1] == df.index[-3].normalize()


def test_daily_ohlc_aggregates_the_four_hour_bars():
    """일봉의 고·저·종가가 그날 4h봉들의 최대·최소·마지막과 같아야 한다."""
    df = _frame([10.0, 12.0, 9.0, 11.0, 13.0, 10.5] * 3)
    daily = wave_setup._daily(df)
    assert len(daily) == 3
    first = df.iloc[:6]
    assert daily["High"].iloc[0] == first["High"].max()
    assert daily["Low"].iloc[0] == first["Low"].min()
    assert daily["Close"].iloc[0] == first["Close"].iloc[-1]
    assert daily["Open"].iloc[0] == first["Open"].iloc[0]


def test_still_active_drops_abc_when_fast_falls_back():
    """ⓐ 돌파가 실패해 단기가 다시 하락하면 추적에서 뺀다."""
    closes = _falling_then_pop()
    p = Params(impulse_enabled=False)
    df = _frame(closes)
    e = detect(df, "XUSDT", p)[0]
    assert still_active(df, e, p)                    # 방금은 살아 있다

    later = _frame(closes + [closes[-2] * 0.5, closes[-2] * 0.4])
    assert supertrend_full(later, p.fast_period, p.fast_mult)["dir"].iloc[-1] < 0
    assert not still_active(later, e, p)


def test_still_active_drops_when_long_term_turns_up():
    """공통 전제가 사라지면(장기 상승 전환) 두 변형 모두 빠진다."""
    closes = _falling_then_pop()
    p = Params(impulse_enabled=False)
    e = detect(_frame(closes), "XUSDT", p)[0]
    surge = _frame(closes + [closes[-1] * 2, closes[-1] * 4, closes[-1] * 8])
    assert supertrend_full(surge, p.slow_period, p.slow_mult)["dir"].iloc[-1] > 0
    assert not still_active(surge, e, p)


def test_short_history_is_not_judged():
    """수퍼트렌드가 성립할 만큼 봉이 없으면 아무것도 내지 않는다."""
    assert detect(_frame(list(np.linspace(100.0, 90.0, 20))), "XUSDT", Params()) == []


def test_dedup_key_separates_the_two_variants():
    """같은 종목·같은 시각이라도 ABC와 임펄스는 서로를 막지 않는다."""
    from invest_signal.signals import SignalEvent
    t = pd.Timestamp("2026-08-01", tz="UTC")
    a = SignalEvent("XUSDT", "wave_setup", t, 1.0, {"stage": ABC})
    b = SignalEvent("XUSDT", "wave_setup", t, 1.0, {"stage": IMPULSE})
    assert a.dedup_key != b.dedup_key


def test_crypto_only_and_runs_intrabar():
    """크립토 전용이고, 인트라바 스캔에서도 돈다(매시간)."""
    assert wave_setup.CRYPTO_ONLY is True
    assert wave_setup.INTRABAR_OK is True


def test_impulse_lookback_follows_the_widened_grace():
    """스캐너가 grace_bars를 4h 기준으로 넓히면 임펄스도 같은 일수를 본다.

    안 그러면 daily_grace_bars(1)에 묶여 임펄스만 추적이 이틀 만에 끊긴다.
    """
    import dataclasses

    from invest_signal.scanner import ONGOING_LOOKBACK_BARS, TRACK_DAYS

    n = 6 * 120
    df = _frame(list(np.linspace(100.0, 50.0, n)))
    p = Params(abc_enabled=False)
    daily = wave_setup._daily(df)
    fast = supertrend_full(daily, p.fast_period, p.fast_mult)
    # TRACK_DAYS만큼 거슬러 올라간 날에 터치를 만든다 — 좁은 창이면 못 잡는다
    back = TRACK_DAYS
    day = daily.index[-1 - back + 1]
    line = float(fast["line"].loc[day])
    df.loc[df.index.normalize() == day, "High"] = line + 1.0

    assert detect(df, "XUSDT", p) == []              # 기본 창(1일)으로는 못 본다
    wide = dataclasses.replace(p, grace_bars=ONGOING_LOOKBACK_BARS)
    evs = detect(df, "XUSDT", wide)
    assert len(evs) == 1 and evs[0].bar_time == day


def test_refresh_detail_updates_the_returns_to_now():
    """추적 줄의 수익률은 트리거 시점이 아니라 현재 값이어야 한다.

    며칠 지난 셋업이면 며칠 전 수익률이 현재가 옆에 박제된다.
    """
    closes = _falling_then_pop()
    p = Params(impulse_enabled=False)
    e = detect(_frame(closes), "XUSDT", p)[0]
    at_trigger = e.detail["ret_24h"]

    later = _frame(closes + list(np.linspace(closes[-1], closes[-1] * 5, 12)))
    wave_setup.refresh_detail(later, e, p)
    assert e.detail["ret_24h"] != at_trigger
    assert e.detail["ret_24h"] > 0        # 그 사이 크게 올랐다
    # 추세선은 알림에 안 실으므로 갱신하지 않는다 — 트리거 시점 기록 그대로
    assert e.detail["fast_line"] == detect(_frame(closes), "XUSDT", p)[0].detail["fast_line"]


def test_impulse_tracking_stays_inside_the_track_window():
    """넓힌 창으로도 TRACK_DAYS보다 오래된 일봉은 잡지 않는다.

    일봉은 완성 시점이 이미 하루 지난 봉이라, 환산에서 1을 빼지 않으면
    추적 목록에 하루 많은 줄이 남는다.
    """
    import dataclasses

    from invest_signal.scanner import ONGOING_LOOKBACK_BARS, TRACK_DAYS

    df = _frame(list(np.linspace(100.0, 50.0, 6 * 120)))
    p = Params(abc_enabled=False)
    daily = wave_setup._daily(df)
    fast = supertrend_full(daily, p.fast_period, p.fast_mult)
    wide = dataclasses.replace(p, grace_bars=ONGOING_LOOKBACK_BARS)

    # 마지막 완성 일봉이 이미 하루 전이므로, 벽시계로 TRACK_DAYS일 된 봉은
    # 인덱스로 TRACK_DAYS-1칸 뒤다. 그보다 한 칸 더 뒤는 창 밖이어야 한다.
    for back, inside in ((TRACK_DAYS - 1, True), (TRACK_DAYS, False)):
        probe = df.copy()
        day = daily.index[-1 - back]
        line = float(fast["line"].loc[day])
        probe.loc[probe.index.normalize() == day, "High"] = line + 1.0
        got = [e.bar_time for e in detect(probe, "XUSDT", wide)]
        assert (day in got) is inside, (back, inside, got)


def test_impulse_can_use_the_running_day_when_asked():
    """include_live_day면 진행 중인 오늘도 일봉으로 보고 그날 터치를 잡는다.

    끄면 오늘은 미완성이라 버려지므로, 오늘의 터치는 내일 UTC 00시가
    지나야 알림이 된다 — 최대 하루가 늦는다.
    """
    import dataclasses

    n = 6 * 120 + 2                      # 마지막 날은 4h봉 2개뿐 (진행 중)
    df = _frame(list(np.linspace(100.0, 50.0, n)))
    p = Params(abc_enabled=False)

    live = wave_setup._daily(df, keep_partial=True)
    assert len(live) == len(wave_setup._daily(df)) + 1      # 오늘이 한 줄 더

    fast = supertrend_full(live, p.fast_period, p.fast_mult)
    today = live.index[-1]
    line = float(fast["line"].iloc[-1])
    df.loc[df.index.normalize() == today, "High"] = line + 1.0

    assert detect(df, "XUSDT", p) == []                     # 기본은 오늘을 안 본다
    got = detect(df, "XUSDT", dataclasses.replace(p, include_live_day=True))
    assert len(got) == 1 and got[0].bar_time == today


def test_with_fields_only_sets_known_params():
    """스캐너는 그 필드를 아는 시그널에만 값을 넘긴다."""
    import dataclasses

    from invest_signal.scanner import _with_fields
    from invest_signal.signals import uptrend_onset

    wave = _with_fields(Params(), include_live_day=True)
    assert wave.include_live_day is True

    other = uptrend_onset.Params()       # 이 필드를 모르는 시그널
    assert _with_fields(other, include_live_day=True) == other
    assert not dataclasses.replace(other, grace_bars=1).__dict__.get("include_live_day")


def test_abc_tracking_is_capped_shorter_than_the_shared_window():
    """ABC는 공용 창(TRACK_DAYS)보다 짧은 abc_track_days까지만 추적한다.

    임펄스는 같은 호출에서 공용 창을 그대로 쓴다 — 캡은 ABC 전용이다.
    """
    import dataclasses

    from invest_signal.scanner import ONGOING_LOOKBACK_BARS, TRACK_DAYS

    p = Params(impulse_enabled=False, abc_track_days=3)
    wide = dataclasses.replace(p, grace_bars=ONGOING_LOOKBACK_BARS)
    assert TRACK_DAYS > p.abc_track_days        # 캡이 실제로 좁히는 상황

    closes = _falling_then_pop()
    inside = _frame(closes + list(np.linspace(closes[-1], closes[-1] * 1.02,
                                              3 * 6)))       # 전환 후 3일
    outside = _frame(closes + list(np.linspace(closes[-1], closes[-1] * 1.02,
                                               4 * 6)))      # 전환 후 4일
    assert len(detect(inside, "XUSDT", wide)) == 1
    assert detect(outside, "XUSDT", wide) == []

    # 캡을 끄면 공용 창 안이므로 다시 잡힌다
    assert len(detect(outside, "XUSDT",
                      dataclasses.replace(wide, abc_track_days=0))) == 1


def test_abc_cap_does_not_shorten_the_normal_grace():
    """평소 스캔(grace_bars=1)에서는 캡이 아무것도 바꾸지 않는다."""
    df = _frame(_falling_then_pop())
    p = Params(impulse_enabled=False)
    assert len(detect(df, "XUSDT", p)) == 1
