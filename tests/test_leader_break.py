"""크립토 모멘텀 눌림목/이탈(15m봉) 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

import numpy as np
import pandas as pd

from invest_signal.indicators import supertrend_full
from invest_signal.signals import leader_break
from invest_signal.signals.leader_break import Params

P = Params()


def make_df(closes, start="2026-08-01"):
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": 1000.0})


def rising(bars=120, start=100.0, step=1.0):
    """우상향 — 종가가 항상 60SMA 위에 있는 구간."""
    return [start + step * i for i in range(bars)]


def test_fires_on_first_close_below_ma():
    closes = rising()
    closes.append(closes[-1] - 200)          # 60선 아래로 훅 빠지는 봉
    df = make_df(closes)
    evs = leader_break.detect(df, "TESTUSDT", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "leader_break"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["ma_period"] == P.ma and ev.detail["interval"] == "15m"
    assert ev.price < ev.detail["ma"]


def test_no_repeat_while_below_ma():
    """이탈 후 계속 60선 아래에 머물러도 첫 봉에서만 알린다."""
    closes = rising()
    closes.extend([closes[-1] - 200] * 4)    # 이탈 후 4봉 더 아래에 머묾
    df = make_df(closes)
    evs = leader_break.detect(df, "TESTUSDT", Params(grace_bars=4))
    assert len(evs) == 1                     # 재이탈이 아니라 유지 — 1건뿐
    assert evs[0].bar_time == df.index[-4]


def test_rearm_after_recovering_above_ma():
    """60선 위로 복귀했다가 다시 깨면 새 이탈로 잡는다."""
    closes = rising(bars=200)
    ma_now = float(np.mean(closes[-60:]))
    closes.append(ma_now - 50)               # 1차 이탈
    closes.append(closes[-1] + 300)          # 60선 위로 복귀
    closes.append(float(np.mean(closes[-60:])) - 50)   # 2차 이탈
    df = make_df(closes)
    evs = leader_break.detect(df, "TESTUSDT", Params(grace_bars=4))
    assert len(evs) == 2
    assert [e.bar_time for e in evs] == [df.index[-3], df.index[-1]]


def test_stays_above_ma_no_signal():
    df = make_df(rising())
    assert leader_break.detect(df, "TESTUSDT", P) == []


def test_grace_limits_how_far_back_it_looks():
    """이탈이 grace 창 밖이면 잡지 않는다 — 이미 알린 건 다시 안 뜬다."""
    closes = rising()
    closes.append(closes[-1] - 200)          # 이탈
    closes.extend([closes[-1] - 1 * i for i in range(1, 11)])   # 10봉 경과
    df = make_df(closes)
    assert leader_break.detect(df, "TESTUSDT", Params(grace_bars=4)) == []


def test_insufficient_data_returns_empty():
    assert leader_break.detect(make_df(rising(bars=P.ma - 5)), "TESTUSDT", P) == []


def test_leaders_ranks_by_gain_within_universe_and_liquidity():
    """상승률 내림차순 top_n — 유니버스 밖·거래대금 미달은 제외."""
    ticker = {
        "AUSDT": {"change_pct": 0.90, "quote_volume": 50e6, "last": 1.0},
        "BUSDT": {"change_pct": 0.70, "quote_volume": 9e6, "last": 1.0},
        "JUNKUSDT": {"change_pct": 5.00, "quote_volume": 1e4, "last": 1.0},  # 거래대금 미달
        "SPOTONLYUSDT": {"change_pct": 3.00, "quote_volume": 90e6, "last": 1.0},
        "CUSDT": {"change_pct": 0.10, "quote_volume": 7e6, "last": 1.0},
    }
    universe = {"AUSDT", "BUSDT", "JUNKUSDT", "CUSDT"}     # SPOTONLY는 스캔 대상 아님
    top = leader_break.leaders(ticker, universe, Params(top_n=2))
    assert [s for s, _ in top] == ["AUSDT", "BUSDT"]       # 잡코인·유니버스 밖 제외
    # 하한을 0으로 낮추면 거래 없는 급등 코인이 1위로 올라온다
    top0 = leader_break.leaders(ticker, universe, Params(top_n=2, min_turnover_usd=0))
    assert top0[0][0] == "JUNKUSDT"


def test_leaders_top_n_caps_result():
    ticker = {f"C{i}USDT": {"change_pct": i / 10, "quote_volume": 10e6, "last": 1.0}
              for i in range(20)}
    top = leader_break.leaders(ticker, set(ticker), Params(top_n=5))
    assert len(top) == 5
    assert [s for s, _ in top] == [f"C{i}USDT" for i in (19, 18, 17, 16, 15)]


def test_watch_list_keeps_dropped_leaders_recent_first():
    """상위권에서 밀려난 종목도 recent에 남아 있으면 계속 확인 대상."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    top = [("AUSDT", {"change_pct": 0.9}), ("BUSDT", {"change_pct": 0.5})]
    recent = {
        "AUSDT": now,                          # 지금도 상위권 — 중복으로 안 들어감
        "OLDUSDT": now - timedelta(days=1),    # 어제 밀려남
        "OLDERUSDT": now - timedelta(days=5),  # 닷새 전
        "GONEUSDT": now - timedelta(hours=2),  # 두 시간 전 — 가장 최근
    }
    universe = {"AUSDT", "BUSDT", "OLDUSDT", "OLDERUSDT", "GONEUSDT"}
    out = leader_break.watch_list(top, recent, universe, Params(top_n=2))
    assert [s for s, _ in out] == ["AUSDT", "BUSDT",        # 현재 상위권이 앞
                                   "GONEUSDT", "OLDUSDT", "OLDERUSDT"]  # 최근 등재 순
    assert [since for _, since in out[:2]] == [None, None]  # 상위권은 추적일 표시 없음
    assert out[2][1] == recent["GONEUSDT"]


def test_watch_list_caps_at_max_watch_keeping_current_top():
    """max_watch를 넘으면 현재 상위권은 지키고 잔류분만 잘라낸다."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    top = [(f"T{i}USDT", {"change_pct": 1.0}) for i in range(3)]
    recent = {f"O{i}USDT": now - timedelta(hours=i) for i in range(20)}
    universe = {s for s, _ in top} | set(recent)
    out = leader_break.watch_list(top, recent, universe, Params(top_n=3, max_watch=5))
    assert len(out) == 5
    assert [s for s, _ in out[:3]] == [f"T{i}USDT" for i in range(3)]   # 상위권 보존
    assert [s for s, _ in out[3:]] == ["O0USDT", "O1USDT"]              # 최근 순 2종


def test_watch_list_skips_symbols_no_longer_in_universe():
    """상장폐지 등으로 스캔 유니버스에서 빠진 종목은 되살리지 않는다."""
    from datetime import datetime, timezone

    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    out = leader_break.watch_list([("AUSDT", {"change_pct": 0.9})],
                                  {"DELISTEDUSDT": now}, {"AUSDT"}, Params())
    assert [s for s, _ in out] == ["AUSDT"]


def test_tracking_reports_state_on_both_sides_of_the_ma():
    """유지 중 스냅샷 — 기본은 '선 아래'인 동안만, 끄면 양쪽 다."""
    loose = Params(track_break_only=False)
    above = leader_break.tracking(make_df(rising()), loose)
    assert above["above_ma"] is True
    assert above["last_price"] > above["ma"]
    assert above["ma_period"] == P.ma and above["interval"] == "15m"

    closes = rising()
    closes.append(closes[-1] - 200)          # ma선 아래로 이탈
    below = leader_break.tracking(make_df(closes), loose)
    assert below["above_ma"] is False
    assert below["last_price"] < below["ma"]


def test_tracking_only_while_below_the_ma():
    """기본값은 이탈 조건을 추적에도 적용 — 선 위로 복귀하면 목록에서 빠진다."""
    assert leader_break.tracking(make_df(rising()), P) is None   # 선 위 → 제외
    closes = rising()
    closes.append(closes[-1] - 200)                              # 이탈
    snap = leader_break.tracking(make_df(closes), P)
    assert snap is not None and snap["above_ma"] is False


def test_tracking_needs_enough_bars():
    """ma선을 못 구할 만큼 봉이 적으면 스냅샷을 만들지 않는다."""
    loose = Params(track_break_only=False)
    assert leader_break.tracking(make_df(rising(bars=P.ma - 5)), loose) is None
    assert leader_break.tracking(make_df(rising(bars=P.ma + 5)), loose) is not None


def _df4h(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0})


def test_blocked_excludes_bullish_stack_that_fell_under_the_long_ma():
    """정배열인데 종가가 480선 아래 — 오를 만큼 오르고 꺾인 자리라 제외."""
    up = list(np.linspace(100.0, 400.0, 600))    # 우상향 → MA120>MA240>MA480 정배열
    df = _df4h(up)
    assert alignment_of(df) == "정배열"
    assert leader_break.blocked(df, P) is False     # 아직 480선 위 → 통과

    ma480 = float(pd.Series(up).rolling(480).mean().iloc[-1])
    df_down = _df4h(up + [ma480 - 20])           # 480선 아래로 밀린 봉
    assert alignment_of(df_down) == "정배열"       # 배열은 아직 정배열
    assert leader_break.blocked(df_down, P) is True


THREE = Params(exhausted_mas=(120, 240, 480))    # 혼조가 나오려면 선이 셋 이상


def _mixed_frame():
    """세 선 기준 혼조 — 오래 흘러내리다 최근에만 반등한 모양."""
    return _df4h(list(np.linspace(400.0, 100.0, 480))
                 + list(np.linspace(100.0, 300.0, 120)))


def test_two_line_alignment_has_no_mixed_state():
    """기본값(120·240)은 선이 둘이라 정배열/역배열뿐 — 혼조가 안 나온다.

    컷이 '120선이 240선 위인가' 하나로 단순해진다는 뜻이다.
    """
    df = _mixed_frame()
    assert alignment_of(df, (120, 240, 480)) == "혼조"
    assert alignment_of(df, (120, 240)) in ("정배열", "역배열")


def test_require_aligned_off_lets_even_mixed_through():
    """require_aligned를 끄면 배열을 아예 안 본다 — 혼조까지 통과."""
    df = _mixed_frame()
    assert alignment_of(df, THREE.exhausted_mas) == "혼조"
    assert leader_break.blocked(df, THREE) is True
    assert leader_break.blocked(
        df, Params(exhausted_mas=(120, 240, 480), require_aligned=False)) is False


def test_blocked_excludes_short_history_unless_allowed():
    """가장 긴 선이 안 잡히면 배열 판정 불가 — 기본은 제외.

    기본값은 240봉(40일)이 요건이다. 480을 배열 판정에서 뺐기 때문에 예전
    80일에서 내려왔다 — 상장 40~80일 종목은 배열은 잡히고 MA480이 없어
    exhausted_below 검사를 그냥 통과하므로 새로 들어온다.
    """
    new_listing = _df4h(np.linspace(100.0, 200.0, 200))
    assert alignment_of(new_listing) is None
    assert leader_break.blocked(new_listing, P) is True
    assert leader_break.blocked(
        new_listing, Params(allow_short_history=True)) is False
    # 판정이 되는데 통과 배열이 아닌 경우(혼조)는 allow_short_history와 무관하게 제외
    mixed = _mixed_frame()
    assert alignment_of(mixed, THREE.exhausted_mas) == "혼조"
    assert leader_break.blocked(
        mixed, Params(exhausted_mas=(120, 240, 480), allow_short_history=True)) is True


def test_filters_can_be_turned_off_independently():
    up = list(np.linspace(100.0, 400.0, 600))
    ma480 = float(pd.Series(up).rolling(480).mean().iloc[-1])
    fell = _df4h(up + [ma480 - 20])              # 정배열 + 480선 아래
    assert leader_break.blocked(fell, Params(exhausted_filter=False)) is False
    down = _df4h(np.linspace(400.0, 100.0, 600))  # 역배열
    assert leader_break.blocked(down, Params(require_aligned=False)) is False
    # 둘 다 끄면 아무것도 안 걸린다
    off = Params(require_aligned=False, exhausted_filter=False)
    assert leader_break.blocked(fell, off) is False and leader_break.blocked(down, off) is False


def alignment_of(df, periods=None):
    from invest_signal.indicators import alignment
    return alignment(df, periods or P.exhausted_mas)


def test_watch_list_drops_illiquid_carryovers():
    """이월분에도 거래대금 하한 적용 — 거래가 마른 종목이 슬롯을 못 차지한다."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    top = [("AUSDT", {"change_pct": 0.5, "quote_volume": 9e6})]
    recent = {"DRYUSDT": now - timedelta(hours=1),   # 이월인데 거래대금 급감
              "OKUSDT": now - timedelta(hours=2)}    # 이월이고 유동성 유지
    ticker = {"AUSDT": {"quote_volume": 9e6}, "DRYUSDT": {"quote_volume": 2e5},
              "OKUSDT": {"quote_volume": 5e6}}
    uni = {"AUSDT", "DRYUSDT", "OKUSDT"}

    got = leader_break.watch_list(top, recent, uni, P, ticker)
    assert [s for s, _ in got] == ["AUSDT", "OKUSDT"]
    # ticker를 안 주면 하한 검사를 건너뛴다 (예전 동작)
    got2 = leader_break.watch_list(top, recent, uni, P)
    assert [s for s, _ in got2] == ["AUSDT", "DRYUSDT", "OKUSDT"]
    # 현재 상위권은 leaders()에서 이미 하한을 통과했으므로 다시 안 거른다
    assert got[0][0] == "AUSDT"


def _quiet_df4h(price=100.0, swing=0.01, bars=60):
    """4h 프레임 — swing이 클수록 ATR이 커진다(고·저를 종가 ±swing만큼 벌린다).

    고·저가 종가 대비 ±swing이면 TR은 매 봉 2·swing·가격이라 ATR÷종가 ≈ 2·swing.
    """
    idx = pd.date_range("2026-06-01", periods=bars, freq="4h", tz="UTC")
    c = pd.Series(price, index=idx)
    return pd.DataFrame({"Open": c, "High": c * (1 + swing), "Low": c * (1 - swing),
                         "Close": c, "Volume": 1.0})


def test_quiet_marks_low_turnover_low_volatility():
    """거래대금·변동성이 둘 다 기준 아래일 때만 '조용'."""
    calm = _quiet_df4h(swing=0.01)              # ATR ≈ 2% < 7.3%
    assert leader_break.quiet({"quote_volume": 4e6}, calm, P) is True
    # 거래대금이 크면 조용하지 않다 — 변동성이 낮아도
    assert leader_break.quiet({"quote_volume": 80e6}, calm, P) is False
    # 변동성이 크면 조용하지 않다 — 거래대금이 작아도
    wild = _quiet_df4h(swing=0.06)              # ATR ≈ 12% > 7.3%
    assert leader_break.quiet({"quote_volume": 4e6}, wild, P) is False


def test_quiet_thresholds_are_configurable():
    """하한/상한을 바꾸면 같은 종목의 판정이 뒤집힌다."""
    calm = _quiet_df4h(swing=0.01)
    stat = {"quote_volume": 4e6}
    assert leader_break.quiet(stat, calm, Params(quiet_turnover_usd=1e6)) is False
    assert leader_break.quiet(stat, calm, Params(quiet_atr_pct=0.005)) is False


def test_quiet_is_none_when_undecidable():
    """판단할 값이 없으면 None — 조용하다고도, 아니라고도 하지 않는다."""
    calm = _quiet_df4h(swing=0.01)
    assert leader_break.quiet(None, calm, P) is None             # 티커 없음
    assert leader_break.quiet({}, calm, P) is None               # 거래대금 없음
    assert leader_break.quiet({"quote_volume": 4e6}, None, P) is None   # 4h 없음
    short = _quiet_df4h(swing=0.01, bars=P.quiet_atr_period)     # ATR 미성립
    assert leader_break.quiet({"quote_volume": 4e6}, short, P) is None


def test_bearish_stacks_pass_only_when_allowed():
    """역배열은 기본 제외 — allow_bearish를 켤 때만 통과한다."""
    down = list(np.linspace(400.0, 100.0, 600))
    df = _df4h(down)
    assert alignment_of(df) == "역배열"
    assert leader_break.blocked(df, P) is True
    assert leader_break.blocked(df, Params(allow_bearish=True)) is False


def test_blocked_still_cuts_mixed_alignment():
    """혼조는 그대로 제외 — 선을 셋 이상 볼 때의 얘기다."""
    df = _mixed_frame()
    assert alignment_of(df, THREE.exhausted_mas) == "혼조"
    assert leader_break.blocked(df, THREE) is True


def test_the_480_cut_applies_to_bullish_stacks_only():
    """480선 컷은 정배열 전용 — 역배열은 그 아래인 게 정상이라 안 건다."""
    up = list(np.linspace(100.0, 400.0, 600))
    ma480 = float(pd.Series(up).rolling(480).mean().iloc[-1])
    bull_below = _df4h(up + [ma480 - 20])
    assert alignment_of(bull_below) == "정배열"
    assert leader_break.blocked(bull_below, P) is True      # 정배열 + 480선 아래 → 컷

    down = list(np.linspace(400.0, 100.0, 600))
    bear = _df4h(down)
    ref = float(pd.Series(down).rolling(480).mean().iloc[-1])
    assert down[-1] < ref                                    # 역배열은 480선 아래가 정상
    # allow_bearish를 켠 상태에서도 480선 컷은 역배열에 걸리지 않는다
    assert leader_break.blocked(bear, Params(allow_bearish=True)) is False


def _wave_frame(down=150, up=150, back=100, top=120.0, bottom=60.0, start=100.0):
    """하락 → 상승 → 되돌림 3구간짜리 30m 프레임."""
    c = np.concatenate([np.linspace(start, bottom, down),
                        np.linspace(bottom, top, up),
                        np.linspace(top, top - (top - bottom) * 0.75, back)])
    idx = pd.date_range("2026-08-01", periods=len(c), freq="30min", tz="UTC")
    return pd.DataFrame({"Open": c, "High": c * 1.004, "Low": c * 0.996,
                         "Close": c, "Volume": 1000.0}, index=idx)


def test_last_wave_spans_the_bottom_to_the_peak():
    """저점은 하락 구간 바닥, 고점은 그 뒤 상승 구간 꼭대기."""
    df = _wave_frame()
    lo, hi = leader_break.last_wave(df, P)
    assert 59 < lo < 62          # 바닥 60 근처
    assert 118 < hi < 122        # 꼭대기 120 근처


def test_retrace_measures_from_the_current_price():
    """되돌림은 지금 가격 기준 — 파동이 닫힌 시점 값이 아니다."""
    df = _wave_frame()
    lo, hi = leader_break.last_wave(df, P)
    got = leader_break.retrace(df, P)
    expected = (hi - float(df["Close"].iloc[-1])) / (hi - lo)
    assert got == expected
    assert 0.7 < got < 0.8       # 75% 되돌린 프레임


def test_retrace_is_none_below_the_threshold():
    """기준(기본 0.618) 미만이면 표시하지 않는다."""
    shallow = _wave_frame(back=100, top=120.0, bottom=60.0)
    shallow = shallow.iloc[:-60]          # 덜 되돌린 시점에서 자른다
    v = leader_break.retrace(shallow, P)
    assert v is None or v >= P.fib_min
    assert leader_break.retrace(_wave_frame(), Params(fib_min=0.99)) is None


def test_retrace_off_by_config():
    assert leader_break.retrace(_wave_frame(), Params(fib_enabled=False)) is None


def _turn_frame(n=200, pop=0.06, rising=True):
    """4h 프레임 — 길게 오르다 눌린 뒤 마지막 봉에서 되살아나는 모양."""
    if rising:
        base = list(np.linspace(100.0, 300.0, n - 30))      # 장기 상승 만들기
        base += list(np.linspace(300.0, 250.0, 29))         # 눌림 (단기만 하락)
    else:
        base = list(np.linspace(300.0, 100.0, n - 1))
    return _df4h(base + [base[-1] * (1 + pop)])


def test_turn_up_needs_the_long_term_rising():
    """4h 장기가 상승인 채로 단기가 막 뒤집혀야 한다."""
    up = _turn_frame()
    fast = supertrend_full(up, P.turn_fast_period, P.turn_fast_mult)["dir"]
    slow = supertrend_full(up, P.turn_slow_period, P.turn_slow_mult)["dir"]
    assert slow.iloc[-1] > 0 and fast.iloc[-1] > 0 and fast.iloc[-2] < 0
    assert leader_break.turn_up(up, P) == leader_break.TURN_FLIP

    # 장기가 하락이면 이 자리가 아니다 (그건 파동 ⓐABC가 잡는 자리)
    down = _turn_frame(rising=False)
    assert supertrend_full(down, P.turn_slow_period, P.turn_slow_mult)["dir"].iloc[-1] < 0
    assert leader_break.turn_up(down, P) is False


def test_turn_up_holds_for_a_few_bars():
    """전환 봉 하나만 보면 4시간 만에 사라진다 — turn_bars 안이면 계속 표시."""
    df = _turn_frame()
    last = float(df["Close"].iloc[-1])
    later = _df4h(list(df["Close"]) + [last * 1.001, last * 1.002])
    assert leader_break.turn_up(later, P) == leader_break.TURN_FLIP   # 2봉 뒤에도 유지
    # 창을 좁히면 전환 봉이 빠진다 — 터치까지 끄면 아무것도 안 남는다
    assert leader_break.turn_up(later, Params(turn_bars=1,
                                             turn_touch=False)) is False


def test_turn_up_is_none_when_undecidable():
    assert leader_break.turn_up(None, P) is None
    assert leader_break.turn_up(_df4h(np.linspace(100.0, 110.0, 20)), P) is None
    assert leader_break.turn_up(_turn_frame(), Params(turn_enabled=False)) is None


def test_turn_up_marks_a_pullback_that_the_fast_line_caught():
    """전환 뒤 눌림이 단기선에 닿으면 터치로 표시한다."""
    df = _turn_frame()                              # 마지막 봉이 전환 봉
    # 전환 봉을 turn_bars 밖으로 밀어내고, 마지막 봉만 저가가 단기선까지
    # 내려오게 한다 — 종가는 선 위(받히고 되돌아온 모양).
    closes = list(df["Close"]) + [df["Close"].iloc[-1] * (1 + 0.01 * i)
                                  for i in range(1, 6)]
    out = _df4h(closes)

    def fast(frame):
        return supertrend_full(frame, P.turn_fast_period, P.turn_fast_mult)

    # 저가를 내리면 ATR·hl2가 같이 움직여 선도 따라 내려온다 — 넉넉히 넘긴다
    out.iloc[-1, out.columns.get_loc("Low")] = float(fast(out)["line"].iloc[-1]) * 0.97
    f = fast(out)
    line = float(f["line"].iloc[-1])
    assert f["dir"].iloc[-1] > 0                    # 단기는 상승인 채로
    assert float(out["Low"].iloc[-1]) <= line <= float(out["High"].iloc[-1])

    assert leader_break.turn_up(out, P) == leader_break.TURN_TOUCH
    # 전환은 이미 창 밖이라, 터치를 끄면 남는 게 없다
    assert leader_break.turn_up(out, Params(turn_touch=False)) is False


def test_turn_up_ignores_a_touch_while_the_fast_line_is_down():
    """단기가 하락 중이면 선에 닿아도 지지가 아니라 저항이다 — 표시 안 함."""
    down = _df4h(list(np.linspace(100.0, 300.0, 170))
                 + list(np.linspace(300.0, 250.0, 30)))
    f = supertrend_full(down, P.turn_fast_period, P.turn_fast_mult)
    assert f["dir"].iloc[-1] < 0                    # 단기는 아직 하락
    down.iloc[-1, down.columns.get_loc("High")] = float(f["line"].iloc[-1]) * 1.03
    assert leader_break.turn_up(down, P) is False


def _resist_frame(n=200):
    """1h 프레임 — 길게 내리다 마지막에 살짝 반등하는 모양(단기·장기 모두 하락)."""
    return _df4h(list(np.linspace(300.0, 100.0, n - 6))
                 + [100.0 * (1 + 0.004 * i) for i in range(1, 7)])


def test_resist_test_marks_a_touch_from_below():
    """단기선이 위에 있는데 고가가 거기 닿으면 터치."""
    df = _resist_frame()
    fast = supertrend_full(df, P.turn_fast_period, P.turn_fast_mult)
    assert fast["dir"].iloc[-1] < 0                 # 단기는 하락 — 선이 위에 있다
    line = float(fast["line"].iloc[-1])
    assert line > float(df["Close"].iloc[-1])
    df.iloc[-1, df.columns.get_loc("High")] = line * 1.03    # 넉넉히 넘겨 닿게

    f = supertrend_full(df, P.turn_fast_period, P.turn_fast_mult)
    assert f["dir"].iloc[-1] < 0                    # 종가는 그대로라 아직 하락
    assert leader_break.resist_test(df, P) == leader_break.RESIST_TOUCH


def test_resist_test_calls_a_close_above_the_line_a_break():
    """종가가 선을 넘겨 단기가 뒤집히면 돌파 — 터치보다 우선한다."""
    df = _resist_frame()
    line = float(supertrend_full(df, P.turn_fast_period,
                                 P.turn_fast_mult)["line"].iloc[-1])
    out = _df4h(list(df["Close"])[:-1] + [line * 1.05])
    assert supertrend_full(out, P.turn_fast_period,
                           P.turn_fast_mult)["dir"].iloc[-1] > 0
    assert leader_break.resist_test(out, P) == leader_break.RESIST_BREAK


def test_resist_test_can_require_the_long_term_falling():
    """1h 장기가 상승이면 기본값에선 안 붙고, 조건을 끄면 붙는다."""
    # 봉마다 ±1% 폭을 줘야 ATR이 성립한다 — 폭이 0이면 밴드가 종가에
    # 달라붙어 얕은 눌림에도 장기까지 같이 뒤집힌다.
    closes = list(np.linspace(100.0, 300.0, 170)) + list(np.linspace(300.0, 276.0, 20))
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, index=idx)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                       "Close": c, "Volume": 1.0})
    fast = supertrend_full(df, P.turn_fast_period, P.turn_fast_mult)
    slow = supertrend_full(df, P.turn_slow_period, P.turn_slow_mult)
    assert slow["dir"].iloc[-1] > 0 and fast["dir"].iloc[-1] < 0   # 장기만 상승
    df.iloc[-1, df.columns.get_loc("High")] = float(fast["line"].iloc[-1]) * 1.02

    assert leader_break.resist_test(df, P) is False                # 장기 상승이라 제외
    assert leader_break.resist_test(
        df, Params(turn1h_require_bearish=False)) == leader_break.RESIST_TOUCH


def test_resist_test_is_none_when_undecidable():
    assert leader_break.resist_test(None, P) is None
    assert leader_break.resist_test(_df4h(np.linspace(100.0, 110.0, 20)), P) is None
    assert leader_break.resist_test(_resist_frame(),
                                    Params(turn1h_enabled=False)) is None
