"""파동 시그널 — 4h 장기선 터치(ABC)와 일봉 단기선 터치(임펄스)."""

import numpy as np
import pandas as pd

from dataclasses import replace as dataclasses_replace

from invest_signal.indicators import anchored_vwap, supertrend_full
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


def _touch(df, params=None, fast=False):
    """마지막 봉을 늘려 추세선에 닿게 만든다.

    장기선(하락)은 위에 있으니 고가를 올리고, 단기선(상승)은 아래에 있으니
    저가를 내린다. 폭을 바꾸면 ATR이 바뀌어 선도 다시 계산되는데, **전환
    봉에는 트레일링 하한이 없어** 선이 내가 옮긴 쪽으로 계속 따라온다 —
    그래서 선에 딱 맞추지 않고 5%쯤 지나쳐 잡는다.
    """
    p = params or Params()
    df = df.copy()
    period, mult = ((p.fast_period, p.fast_mult) if fast
                    else (p.slow_period, p.slow_mult))
    want_dir, col, over = (1, "Low", 0.95) if fast else (-1, "High", 1.05)
    for _ in range(5):
        st = supertrend_full(df, period, mult)
        line = float(st["line"].iloc[-1])
        if (np.sign(st["dir"].iloc[-1]) == want_dir
                and df["Low"].iloc[-1] <= line <= df["High"].iloc[-1]):
            return df
        df.iloc[-1, df.columns.get_loc(col)] = line * over
    raise AssertionError(f"{'단기선' if fast else '장기선'} 터치를 못 만들었다")


def _touching_slow(df, params=None):
    return _touch(df, params, fast=False)


def _abc_frame(pop=0.05, params=None, gap=0, start="2026-01-01", n=200):
    """ⓐ 새 정의를 만족하는 프레임 — 단기가 최근 뒤집혔고 고가가 장기선에 닿는다.

    gap을 주면 전환 봉과 터치 봉 사이를 그만큼 벌린다(전환이 오래된 경우).
    """
    closes = _falling_then_pop(n=n, pop=pop)
    # 갭은 평평하게 채운다 — 조금이라도 올리면 장기까지 상승 전환해 버려서
    # '장기는 아직 하락'이라는 전제가 깨진다.
    closes += [closes[-1]] * gap
    return _touching_slow(_frame(closes, start=start), params)


def _extend(df, closes_more):
    """프레임 뒤에 봉을 붙인다 — 터치를 만든 고가 수정을 그대로 보존한다."""
    more = _frame(closes_more, start=df.index[-1] + pd.Timedelta(hours=4))
    return pd.concat([df, more])


def test_abc_fires_when_a_recent_flip_reaches_the_slow_line():
    """ⓐ 단기가 막 뒤집혔고 캔들이 장기선(저항)에 닿은 봉에서 발화한다."""
    p = Params(impulse_enabled=False)
    df = _abc_frame(params=p)
    fast = supertrend_full(df, p.fast_period, p.fast_mult)["dir"]
    slow = supertrend_full(df, p.slow_period, p.slow_mult)
    # 전제 — 단기는 방금 뒤집혔고 장기는 아직 하락
    assert fast.iloc[-2] < 0 and fast.iloc[-1] > 0 and slow["dir"].iloc[-1] < 0

    evs = detect(df, "XUSDT", p)
    assert len(evs) == 1
    e = evs[0]
    assert e.signal == "wave_setup" and e.detail["stage"] == ABC
    assert e.detail["interval"] == "4h"
    assert e.bar_time == df.index[-1]
    assert e.price < e.detail["slow_line"]      # 터치지 돌파가 아니다


def test_abc_fires_on_the_flip_itself():
    """전환 봉은 그 자리에서 '단기선 돌파'로 알린다 — 되돌림을 기다리지 않는다."""
    df = _frame(_falling_then_pop())
    p = Params(impulse_enabled=False)
    fast = supertrend_full(df, p.fast_period, p.fast_mult)["dir"]
    slow = supertrend_full(df, p.slow_period, p.slow_mult)
    assert fast.iloc[-1] > 0 and slow["dir"].iloc[-1] < 0      # 전환은 했다
    assert df["High"].iloc[-1] < float(slow["line"].iloc[-1])  # 장기선까진 멀다

    e = detect(df, "XUSDT", p)[0]
    assert e.detail["touched"] == wave_setup.FAST_LINE
    assert e.detail["kind"] == wave_setup.BREAK
    # 전환 트리거를 끄면 침묵한다 — 어느 선에도 안 닿은 자리다
    assert detect(df, "XUSDT", dataclasses_replace(p, abc_flip=False)) == []


def test_abc_silent_when_the_flip_is_too_old():
    """전환이 flip_window_bars보다 오래됐으면 터치해도 발화하지 않는다."""
    df = _abc_frame(gap=4)                       # 전환 4봉 뒤에 터치
    assert detect(df, "XUSDT", Params(impulse_enabled=False)) != []      # 기본 창(12봉)
    assert detect(df, "XUSDT",
                  Params(impulse_enabled=False, flip_window_bars=2)) == []  # 창 밖


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
    p = Params(abc_enabled=False, vwap_condition=False)
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
    """ⓐ 반등이 죽어 단기가 다시 하락하면 추적에서 뺀다."""
    p = Params(impulse_enabled=False)
    df = _abc_frame(params=p)
    e = detect(df, "XUSDT", p)[0]
    assert still_active(df, e, p)                    # 방금은 살아 있다

    last = float(df["Close"].iloc[-1])
    later = _extend(df, [last * 0.5, last * 0.4])
    assert supertrend_full(later, p.fast_period, p.fast_mult)["dir"].iloc[-1] < 0
    assert not still_active(later, e, p)


def test_still_active_drops_when_long_term_turns_up():
    """공통 전제가 사라지면(장기 상승 전환) 두 변형 모두 빠진다."""
    p = Params(impulse_enabled=False)
    df = _abc_frame(params=p)
    e = detect(df, "XUSDT", p)[0]
    last = float(df["Close"].iloc[-1])
    surge = _extend(df, [last * 2, last * 4, last * 8])
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
    p = Params(abc_enabled=False, vwap_condition=False)
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
    p = Params(impulse_enabled=False)
    df = _abc_frame(params=p)
    e = detect(df, "XUSDT", p)[0]
    at_trigger = e.detail["ret_24h"]
    line_at_trigger = e.detail["fast_line"]

    last = float(df["Close"].iloc[-1])
    later = _extend(df, list(np.linspace(last, last * 5, 12)))
    wave_setup.refresh_detail(later, e, p)
    assert e.detail["ret_24h"] != at_trigger
    assert e.detail["ret_24h"] > 0        # 그 사이 크게 올랐다
    # 추세선은 알림에 안 실으므로 갱신하지 않는다 — 트리거 시점 기록 그대로
    assert e.detail["fast_line"] == line_at_trigger


def test_impulse_tracking_stays_inside_the_track_window():
    """넓힌 창으로도 TRACK_DAYS보다 오래된 일봉은 잡지 않는다.

    일봉은 완성 시점이 이미 하루 지난 봉이라, 환산에서 1을 빼지 않으면
    추적 목록에 하루 많은 줄이 남는다.
    """
    import dataclasses

    from invest_signal.scanner import ONGOING_LOOKBACK_BARS, TRACK_DAYS

    df = _frame(list(np.linspace(100.0, 50.0, 6 * 120)))
    p = Params(abc_enabled=False, vwap_condition=False)
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
    p = Params(abc_enabled=False, vwap_condition=False)

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

    df = _abc_frame(params=p)
    trigger = detect(df, "XUSDT", p)[0].bar_time
    last = float(df["Close"].iloc[-1])
    inside = _extend(df, list(np.linspace(last, last * 1.02, 3 * 6)))    # 터치 후 3일
    outside = _extend(df, list(np.linspace(last, last * 1.02, 4 * 6)))   # 터치 후 4일

    def bars(frame, prm):
        return [e.bar_time for e in detect(frame, "XUSDT", prm)]

    # 반등이 이어지며 나중 봉에서 또 닿을 수 있으니, 건수가 아니라
    # **그 트리거 봉이 소급 창 안에 들어오는지**로 본다.
    assert trigger in bars(inside, wide)
    assert trigger not in bars(outside, wide)
    # 캡을 끄면 공용 창 안이므로 다시 잡힌다
    assert trigger in bars(outside, dataclasses.replace(wide, abc_track_days=0))


def test_abc_cap_does_not_shorten_the_normal_grace():
    """평소 스캔(grace_bars=1)에서는 캡이 아무것도 바꾸지 않는다."""
    p = Params(impulse_enabled=False)
    assert len(detect(_abc_frame(params=p), "XUSDT", p)) == 1


def _vwap_probe():
    """VWAP 게이트만 떼어 보기 위한 최소 프레임.

    꾸준히 오르는 구간이라 MVWAP(누적 평균)은 항상 종가보다 아래에 있고,
    캔들 폭이 좁아 선에 닿지도 않는다 — 마지막 봉만 선 아래로 떨어뜨린다.
    """
    close = np.array([100.0 + 10 * i for i in range(9)] + [50.0])
    idx = pd.date_range("2026-04-01", periods=len(close), freq="4h", tz="UTC")
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999,
                       "Close": close, "Volume": np.full(len(close), 1000.0)},
                      index=idx)
    return df, wave_setup._vwap_gate(df, Params())


def test_vwap_gate_above_and_below():
    """종가가 선 위면 통과, 아래면 불통과 (터치도 없을 때)."""
    df, ok = _vwap_probe()
    assert ok(len(df) - 2) is True       # 선 위
    assert ok(len(df) - 1) is False      # 선 아래, 최근 터치 없음


def test_vwap_gate_accepts_a_touch_inside_the_window():
    """창 안에 캔들이 선을 걸쳤으면 종가가 아래여도 통과한다(any)."""
    df, _ = _vwap_probe()
    j = len(df) - 3
    df.iloc[j, df.columns.get_loc("Low")] = 10.0     # 그 봉이 선을 관통
    df.iloc[j, df.columns.get_loc("High")] = 200.0
    ok = wave_setup._vwap_gate(df, Params(vwap_mode="any", vwap_touch_bars=5))
    assert ok(len(df) - 1) is True

    narrow = wave_setup._vwap_gate(df, Params(vwap_mode="any", vwap_touch_bars=2))
    assert narrow(len(df) - 1) is False              # 창 밖이면 다시 막힌다


def test_vwap_gate_modes_differ():
    """above는 터치를 인정하지 않고, touch는 '선 위'만으로는 통과시키지 않는다."""
    df, _ = _vwap_probe()
    j = len(df) - 3
    df.iloc[j, df.columns.get_loc("Low")] = 10.0
    df.iloc[j, df.columns.get_loc("High")] = 200.0
    last = len(df) - 1
    assert wave_setup._vwap_gate(df, Params(vwap_mode="above"))(last) is False
    assert wave_setup._vwap_gate(df, Params(vwap_mode="touch"))(last) is True
    # 선 위이지만 터치는 없는 봉 — above는 통과, touch는 불통과.
    # 창이 월 첫 봉에 닿지 않는 자리를 고른다: 리셋 직후 첫 봉은 정의상
    # MVWAP = (고+저+종)/3이라 언제나 '터치'로 잡힌다.
    assert wave_setup._vwap_gate(df, Params(vwap_mode="above"))(5) is True
    assert wave_setup._vwap_gate(df, Params(vwap_mode="touch"))(5) is False


def test_vwap_gate_blocks_an_abc_far_below_the_line():
    """실제 발화 경로에서도 막힌다 — 한 달 내내 흘러내린 뒤의 반등."""
    import dataclasses

    on = Params(impulse_enabled=False)
    off = dataclasses.replace(on, vwap_condition=False)
    df = _abc_frame(params=off, start="2026-03-01", n=180)   # 한 달(30일) 안에서만

    assert len(detect(df, "XUSDT", off)) == 1        # 게이트만 빼면 잡히던 자리
    assert detect(df, "XUSDT", on) == []             # 종가는 MVWAP 한참 아래


def _quiet_frame(price=100.0, swing=0.015, vol=1000.0, bars=60):
    """4h 프레임 — swing이 클수록 ATR이, vol이 클수록 거래대금이 커진다.

    고·저가 종가 대비 ±swing이면 TR이 매 봉 2·swing·가격이라 ATR÷종가 ≈ 2·swing.
    """
    idx = pd.date_range("2026-06-01", periods=bars, freq="4h", tz="UTC")
    c = pd.Series(price, index=idx)
    return pd.DataFrame({"Open": c, "High": c * (1 + swing), "Low": c * (1 - swing),
                         "Close": c, "Volume": vol})


def test_quiet_needs_low_turnover_and_a_volatility_band():
    """거래대금이 작고 변동성이 밴드 안일 때만 조용 — 너무 조용해도 아니다."""
    P = wave_setup.Params()
    assert wave_setup.quiet(_quiet_frame(swing=0.015, vol=1000), P) is True   # ATR 3%
    # 거래대금이 크면 아니다 — 6봉 × 100 × 20000 = $12M
    assert wave_setup.quiet(_quiet_frame(swing=0.015, vol=20000), P) is False
    # 변동성이 상한 위면 아니다 (ATR ≈ 12%)
    assert wave_setup.quiet(_quiet_frame(swing=0.06, vol=1000), P) is False
    # 하한 아래도 아니다 — 안 깨지지만 안 가는 구간 (ATR ≈ 1%)
    assert wave_setup.quiet(_quiet_frame(swing=0.005, vol=1000), P) is False


def test_quiet_is_none_when_undecidable():
    """ATR이 성립 안 하는 짧은 이력이면 None — 아니라고 하지 않는다."""
    P = wave_setup.Params()
    assert wave_setup.quiet(pd.DataFrame(), P) is None
    assert wave_setup.quiet(_quiet_frame(bars=P.quiet_atr_period), P) is None


def test_quiet_is_judged_at_the_trigger_bar_not_the_last_one():
    """at으로 과거 봉을 판정한다 — 발화 시점 상태여야 한다."""
    P = wave_setup.Params()
    df = _quiet_frame(swing=0.015, vol=1000)
    df.iloc[-6:, df.columns.get_loc("Volume")] = 20000.0   # 최근 하루만 거래 폭증
    assert wave_setup.quiet(df, P) is False                # 마지막 봉 기준
    assert wave_setup.quiet(df, P, len(df) - 10) is True   # 폭증 전 봉 기준


def test_quiet_is_stamped_on_both_variants_from_the_4h_frame():
    """두 변형 모두 4h 프레임 기준으로 🍃 판정이 붙는다.

    임펄스는 일봉 사건이라 발화 봉이 그날 00시다. 그 위치를 4h 프레임에서
    그대로 찾으면 그날 **첫** 봉이 되므로, 마지막 봉으로 옮겨 판정해야
    ABC와 같은 잣대가 된다.
    """
    n = 6 * 120
    df = _frame(list(np.linspace(100.0, 50.0, n)))
    df["Volume"] = 1.0                       # 거래대금이 아주 작다 → 조용 후보
    # 변동성은 이 테스트의 관심사가 아니다 — 밴드를 열어 거래대금만 남긴다
    p = Params(abc_enabled=False, vwap_condition=False,
               quiet_atr_min=0.0, quiet_atr_max=1.0)
    daily = wave_setup._daily(df)
    fast = supertrend_full(daily, p.fast_period, p.fast_mult)
    day = daily.index[-1]
    df.loc[df.index.normalize() == day, "High"] = float(fast["line"].iloc[-1]) + 1.0
    # **전날 마지막 봉**에 거래를 몰아 준다. 거래대금 창이 6봉이라, 그날 첫
    # 봉을 기준으로 삼으면 이 봉이 창에 들어와 조용이 깨지고, 그날 마지막
    # 봉을 기준으로 삼으면 창 밖이라 조용이 유지된다 — 매핑이 갈리는 자리다.
    first = int(np.flatnonzero(df.index.normalize() == day)[0])
    df.iloc[first - 1, df.columns.get_loc("Volume")] = 10_000_000.0

    ev = detect(df, "XUSDT", p)[0]
    assert ev.detail["stage"] == IMPULSE
    assert ev.detail["quiet"] is True        # 첫 봉을 봤다면 거래대금 초과로 False


def test_abc_fires_on_a_fast_line_touch_too():
    """ⓐ는 장기선(저항)뿐 아니라 단기선(지지) 터치도 잡는다."""
    p = Params(impulse_enabled=False, vwap_condition=False)
    # 마지막 봉의 저가를 단기선까지 내려 '지지 터치'를 만든다
    df = _touch(_frame(_falling_then_pop()), p, fast=True)
    assert supertrend_full(df, p.fast_period, p.fast_mult)["dir"].iloc[-1] > 0

    # 이 봉은 전환 봉이기도 하다 — 우선순위상 '돌파'가 먼저 잡힌다
    assert detect(df, "XUSDT", p)[0].detail["kind"] == wave_setup.BREAK

    # 전환 트리거를 빼면 같은 봉이 '단기선 터치'로 잡힌다
    only_touch = dataclasses_replace(p, abc_flip=False)
    evs = detect(df, "XUSDT", only_touch)
    assert len(evs) == 1
    assert evs[0].detail["stage"] == ABC
    assert evs[0].detail["touched"] == wave_setup.FAST_LINE
    assert evs[0].detail["kind"] == wave_setup.TOUCH

    # 둘 다 끄면 침묵한다 (장기선까진 못 닿은 자리다)
    assert detect(df, "XUSDT",
                  dataclasses_replace(only_touch, abc_touch_fast=False)) == []


def test_abc_slow_touch_is_labelled():
    """장기선 터치는 그렇게 표시된다 — 두 자리를 줄에서 구분해야 한다."""
    p = Params(impulse_enabled=False)
    e = detect(_abc_frame(params=p), "XUSDT", p)[0]
    assert e.detail["touched"] == wave_setup.SLOW_LINE


# ── ⓒ 되돌림 ────────────────────────────────────────────────────────────

def _bars(closes, start="2026-01-01", rng=0.02):
    """폭이 있는 4h봉 — ATR이 살아 있어야 장기선이 되돌림 아래로 내려간다.

    _frame의 기본 폭(±0.2%)으로는 6×ATR이 너무 좁아 장기 수퍼트렌드의
    트레일링 스톱이 0.5 되돌림 **위**에 걸린다. 그러면 값이 레벨에 닿기
    전에 장기가 먼저 하락으로 꺾여 되돌림 자체가 만들어지지 않는다.
    """
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * (1 + rng), "Low": c * (1 - rng),
                         "Close": c, "Volume": np.full(len(c), 1000.0)}, index=idx)


def _add(df, closes, rng=0.02):
    return pd.concat([df, _bars(closes, df.index[-1] + pd.Timedelta(hours=4), rng)])


RETRACE_P = Params(impulse_enabled=False, abc_enabled=False, vwap_condition=False)


def _broken(params=RETRACE_P, n=180, rally=3, step=0.06, top=0.04):
    """장기가 상승 전환(=돌파)하고 고점까지 만든 프레임.

    길게 흘러내리다 step씩 튀어 장기 수퍼트렌드를 뒤집고, 거기서 rally봉을
    더 올려 고점을 세운다. 돌파봉·고점봉 위치를 같이 돌려준다.
    """
    df = _bars(list(np.linspace(100.0, 40.0, n)))
    while not supertrend_full(df, params.slow_period, params.slow_mult)["dir"].iloc[-1] > 0:
        df = _add(df, [float(df["Close"].iloc[-1]) * (1 + step)])
        assert len(df) < n + 60, "장기 상승 전환을 못 만들었다"
    break_i = len(df) - 1
    for _ in range(rally):
        df = _add(df, [float(df["Close"].iloc[-1]) * (1 + top)])
    return df, break_i, len(df) - 1


def _slide(df, bars, step=0.03):
    """고점에서 step씩 흘러내리는 봉을 bars개 붙인다."""
    top = float(df["Close"].iloc[-1])
    for k in range(1, bars + 1):
        df = _add(df, [top * (1 - step * k)])
    return df


def test_retrace_fires_when_price_first_reaches_the_level():
    """ⓒ 저가가 저점~고점의 0.5까지 처음 내려간 봉에서 발화한다."""
    p = RETRACE_P
    df, break_i, high_i = _broken(p)
    df = _slide(df, 4)
    while True:
        evs = [e for e in detect(df, "XUSDT", p) if e.detail["stage"] == wave_setup.RETRACE]
        if evs:
            break
        assert len(df) < 220, "되돌림을 못 만들었다"
        df = _add(df, [float(df["Close"].iloc[-1]) * 0.97])
    assert len(evs) == 1
    e = evs[0]
    t = int(df.index.searchsorted(e.bar_time))
    w = wave_setup.wave_at(df, t, p)
    assert (w.break_i, w.high_i) == (break_i, high_i)
    d = e.detail
    assert (d["wave_low"], d["wave_high"], d["wave_level"]) == (w.low, w.high, w.level)
    assert d["wave_level"] == w.high - 0.5 * (w.high - w.low)
    assert float(df["Low"].iloc[t]) <= d["wave_level"]      # 저가가 레벨을 찍었다
    assert d["interval"] == "4h" and d["stage"] == wave_setup.RETRACE
    assert d["break_time"] == df.index[break_i].isoformat()
    # 📐는 leader_break와 같은 규약 — (고점−종가)÷(고점−저점)
    assert d["fib"] == (w.high - e.price) / (w.high - w.low)
    assert "touched" not in d                # 선을 건드린 게 아니다


def _drift(df, bars, step=0.004):
    """완만하게 흘러내리는 봉 — 장기 수퍼트렌드를 살려 둔 채 시간을 흘린다.

    _slide(3%/봉)는 대여섯 봉이면 장기가 꺾여 감시 창이 닫힌다. 고점이
    창보다 늦게 서는 상황을 만들려면 스무 봉 넘게 살아 있어야 한다.
    """
    top = float(df["Close"].iloc[-1])
    for k in range(1, bars + 1):
        df = _add(df, [top * (1 - step * k)])
    return df


def _flip_bar(df, params=RETRACE_P, at=None):
    """돌파봉(또는 at) 이하 마지막 단기 상승 전환 봉."""
    fast = supertrend_full(df, params.fast_period, params.fast_mult)
    i = len(df) - 1 if at is None else at
    return int(wave_setup._flip_index(fast)[i])


def test_retrace_low_sits_before_the_fast_flip():
    """저점은 **단기 상승 전환 봉 앞 W봉**에서 나온다 — 사건 순서를 지킨다.

    저점 → 단기 상승 전환 → 장기 돌파 → 고점. 창을 고점에 붙이면 고점이
    늦게 설 때 창이 통째로 돌파 뒤로 밀려 랠리 중간 눌림을 저점으로 집는다.
    """
    p = RETRACE_P
    df, break_i, high_i = _broken(p)
    df = _slide(df, 4)
    t = len(df) - 1
    w = wave_setup.wave_at(df, t, p)
    win = p.retrace_window_days * wave_setup.BARS_PER_DAY
    f0 = _flip_bar(df, p, break_i)
    assert 0 <= f0 <= break_i                         # 단기 전환이 돌파보다 앞
    assert w.low_i <= f0 <= w.break_i                 # ← 회귀 방지의 핵심
    assert w.low == float(df["Low"].iloc[max(0, f0 - win + 1):f0 + 1].min())
    # 창보다 앞에 훨씬 낮은 저가를 심어도 안 뽑힌다
    early = df.copy()
    early.iloc[max(0, f0 - win) - 1, early.columns.get_loc("Low")] = 0.01
    assert wave_setup.wave_at(early, t, p).low == w.low
    # 창 안에 심으면 뽑힌다
    inside = df.copy()
    inside.iloc[f0, inside.columns.get_loc("Low")] = w.low * 0.5
    assert wave_setup.wave_at(inside, t, p).low == w.low * 0.5


def test_retrace_low_window_is_not_anchored_to_the_high():
    """고점이 W봉보다 늦게 서도 저점 창은 안 밀린다.

    옛 구현은 창을 고점에 붙여서, 고점이 늦게 서면 창이 통째로 돌파 뒤로
    가 랠리 중간 눌림을 저점이라 불렀다(HEMIUSDT 2026-08-28 건).
    """
    p = RETRACE_P
    df, break_i, _ = _broken(p)
    win = p.retrace_window_days * wave_setup.BARS_PER_DAY
    df = _drift(df, win + 2)                 # 눌림 — 여기가 옛 창의 '저점'이었다
    df = _add(df, [float(df["High"].iloc[break_i:].max()) * 1.10])   # 늦은 새 고점
    df = _drift(df, 2)
    t = len(df) - 1
    w = wave_setup.wave_at(df, t, p)
    assert w.high_i - break_i > win          # 고점이 창 길이보다 늦게 섰다
    assert w.low_i <= w.break_i              # 그래도 저점은 돌파 앞
    # 옛 정의(창을 고점에 붙임)라면 돌파 뒤 눌림을 집었을 것이다
    old_low = float(df["Low"].iloc[max(0, w.high_i - win + 1):w.high_i + 1].min())
    assert old_low > w.low


def test_retrace_high_is_a_running_max_with_no_window():
    """고점에는 창이 없다 — 오래된 고점도 만료되지 않는다.

    롤링 창을 씌우면 진짜 고점이 창 밖으로 밀리는 순간 고점이 낮아져
    📐의 분모가 거짓이 되고 화면 값이 뒷걸음질친다. 만료는
    retrace_expire_bars가 맡는다.
    """
    p = RETRACE_P
    df, break_i, _ = _broken(p)
    win = p.retrace_window_days * wave_setup.BARS_PER_DAY
    peak = float(df["High"].iloc[break_i:].max())
    df = _drift(df, win + 3)                 # 고점을 창 밖으로 밀어낸다
    t = len(df) - 1
    w = wave_setup.wave_at(df, t, p)
    assert t - w.high_i > win                # 고점이 창 밖인데도
    assert w.high == float(df["High"].iloc[break_i:t].max()) == peak
    assert w.high > float(df["High"].iloc[t - win:t].max())


def test_retrace_high_excludes_the_judging_bar():
    """판정 봉의 고가는 고점에서 뺀다 — 봉이 자라며 레벨이 따라 내려오면 안 된다."""
    p = RETRACE_P
    df, _, _ = _broken(p)
    df = _slide(df, 4)
    t = len(df) - 1
    base = wave_setup.wave_at(df, t, p)
    spiked = df.copy()
    spiked.iloc[t, spiked.columns.get_loc("High")] = base.high * 10
    after = wave_setup.wave_at(spiked, t, p)
    assert (after.high, after.high_i, after.level) == (base.high, base.high_i, base.level)


def test_retrace_needs_the_high_to_be_confirmed():
    """고점 확정 — 고점이 retrace_hold_bars만큼 안 갱신돼야 무장한다."""
    p = RETRACE_P
    df, _, high_i = _broken(p)
    # 고점 바로 다음 봉이 레벨을 찍게 만든다
    w = wave_setup.wave_at(_add(df, [float(df["Close"].iloc[-1]) * 0.99]), len(df), p)
    df = _add(df, [w.level * 0.98])
    t = len(df) - 1
    assert t - high_i == 1
    wide = dataclasses_replace(p, grace_bars=30)
    assert not [e for e in detect(df, "X", wide) if e.detail["stage"] == wave_setup.RETRACE]
    loose = dataclasses_replace(wide, retrace_hold_bars=0)
    hits = [e for e in detect(df, "X", loose) if e.detail["stage"] == wave_setup.RETRACE]
    assert [int(df.index.searchsorted(e.bar_time)) for e in hits] == [t]


def test_retrace_fires_once_per_break():
    """존을 나갔다 다시 들어와도 같은 돌파에서 두 번째 알림은 없다."""
    p = dataclasses_replace(RETRACE_P, grace_bars=30)   # 스캐너의 소급 창
    df, _, _ = _broken(RETRACE_P)
    df = _slide(df, 4)
    while not [e for e in detect(df, "X", p) if e.detail["stage"] == wave_setup.RETRACE]:
        assert len(df) < 220
        df = _add(df, [float(df["Close"].iloc[-1]) * 0.97])
    first = [e for e in detect(df, "X", p) if e.detail["stage"] == wave_setup.RETRACE]
    assert len(first) == 1
    lvl = first[0].detail["wave_level"]
    # 레벨 위로 올라갔다가 다시 내려온다
    out = _add(df, [lvl * 1.05, lvl * 1.06, lvl * 0.97])
    again = [e for e in detect(out, "X", p) if e.detail["stage"] == wave_setup.RETRACE]
    assert [e.bar_time for e in again] == [first[0].bar_time]


def test_retrace_expires_after_the_window():
    """돌파 후 retrace_expire_bars가 지난 터치는 안 잡는다."""
    p = dataclasses_replace(RETRACE_P, grace_bars=30)
    df, _, _ = _broken(RETRACE_P)
    df = _slide(df, 4)
    while not [e for e in detect(df, "X", p) if e.detail["stage"] == wave_setup.RETRACE]:
        assert len(df) < 220
        df = _add(df, [float(df["Close"].iloc[-1]) * 0.97])
    tight = dataclasses_replace(p, retrace_expire_bars=2)
    assert not [e for e in detect(df, "X", tight) if e.detail["stage"] == wave_setup.RETRACE]


def _fired(p=None):
    """되돌림이 한 건 잡힌 (프레임, 이벤트)."""
    p = p or dataclasses_replace(RETRACE_P, grace_bars=30)
    df, _, _ = _broken(RETRACE_P)
    df = _slide(df, 4)
    while True:
        evs = [e for e in detect(df, "X", p) if e.detail["stage"] == wave_setup.RETRACE]
        if evs:
            return df, evs[0]
        assert len(df) < 220
        df = _add(df, [float(df["Close"].iloc[-1]) * 0.97])


def test_retrace_survives_while_the_long_term_holds_up():
    """장기가 상승인 동안은 남는다 — ⓐ와 장기 방향 요구가 정반대다."""
    df, e = _fired()
    assert supertrend_full(df, 30, 6.0)["dir"].iloc[-1] > 0
    assert still_active(df, e, RETRACE_P)
    abc = wave_setup._event(df, "X", len(df) - 1, ABC,
                            supertrend_full(df, 22, 3.0), supertrend_full(df, 30, 6.0),
                            RETRACE_P, {})
    assert not still_active(df, abc, RETRACE_P)      # 같은 봉에 ⓐ는 못 산다


def test_retrace_dies_when_the_long_term_turns_back_down():
    """장기가 다시 하락으로 꺾이면 돌파가 실패한 것이라 좌표도 죽는다."""
    df, e = _fired()
    dead = _slide(df, 6, step=0.10)
    assert supertrend_full(dead, 30, 6.0)["dir"].iloc[-1] < 0
    assert not still_active(dead, e, RETRACE_P)


def test_retrace_dies_when_the_low_is_broken():
    """📐이 1.0을 넘으면 되돌림이 아니라 저점 이탈 — 장기가 살아 있어도 뺀다.

    프레임을 저점 아래로 끌어내리면 장기 수퍼트렌드가 먼저 꺾여 두 조건이
    섞인다. 그래서 좌표만 옮겨 📐 분기 하나만 본다.
    """
    df, e = _fired()
    hi = e.detail["wave_high"]
    assert supertrend_full(df, 30, 6.0)["dir"].iloc[-1] > 0
    assert e.detail["fib"] < 1.0 and still_active(df, e, RETRACE_P)
    close = float(df["Close"].iloc[-1])
    e.detail["wave_low"] = close + 0.01 * (hi - close)   # 종가가 저점 아래가 되게
    assert wave_setup._fib(e.detail["wave_low"], hi, close) > 1.0
    assert not still_active(df, e, RETRACE_P)


def test_retrace_fib_is_recalculated_every_scan():
    """추적 줄의 📐는 매 스캔 현재가로 다시 잰다 — 좌표는 트리거 시점 그대로."""
    df, e = _fired()
    lo, hi = e.detail["wave_low"], e.detail["wave_high"]
    before = e.detail["fib"]
    deeper = _add(df, [hi - 0.8 * (hi - lo)])
    wave_setup.refresh_detail(deeper, e, RETRACE_P)
    assert e.detail["fib"] == wave_setup._fib(lo, hi, float(deeper["Close"].iloc[-1]))
    assert e.detail["fib"] > before
    assert (e.detail["wave_low"], e.detail["wave_high"]) == (lo, hi)


def test_break_index_ignores_the_warmup_flip():
    """ATR 워밍업이 끝나며 dir이 NaN → +1이 되는 자리는 돌파가 아니다.

    'not up'을 하락으로 세면 종목마다 유령 돌파가 한 건씩 생긴다.
    """
    dirs = pd.DataFrame({"dir": [np.nan] * 3 + [1.0, 1.0, -1.0, 1.0, 1.0]})
    assert list(wave_setup._break_index(dirs)) == [-1, -1, -1, -1, -1, -1, 6, 6]


def test_retrace_can_be_turned_off():
    p = dataclasses_replace(RETRACE_P, grace_bars=30, retrace_enabled=False)
    df, _, _ = _broken(RETRACE_P)
    df = _slide(df, 8)
    assert not [e for e in detect(df, "X", p) if e.detail["stage"] == wave_setup.RETRACE]


def test_config_yaml_reaches_the_params():
    """config.yaml의 ⓒ 키가 실제로 Params까지 간다.

    기본값이 코드와 config 두 곳에 있어서 한쪽만 고치면 조용히 갈린다 —
    abc_track_days가 실제로 그랬다(코드 3 / config 10).
    """
    from invest_signal import config

    p = config.wave_params(config.load("config.yaml"))
    s = config.load("config.yaml")["signal"]["wave_setup"]
    assert p.abc_track_days == s["abc_track_days"]
    for key in ("retrace_enabled", "retrace_level", "retrace_window_days",
                "retrace_hold_bars", "retrace_expire_bars", "retrace_track_days"):
        assert getattr(p, key) == s[key], key
