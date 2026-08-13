"""크립토 모멘텀 눌림목/이탈(15m봉) 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

import numpy as np
import pandas as pd

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
    """유지 중 스냅샷 — 60선 위로 복귀해도 상태만 바뀌고 계속 만들어진다."""
    above = leader_break.tracking(make_df(rising()), P)
    assert above["above_ma"] is True
    assert above["last_price"] > above["ma"]
    assert above["ma_period"] == P.ma and above["interval"] == "15m"

    closes = rising()
    closes.append(closes[-1] - 200)          # 60선 아래로 이탈
    below = leader_break.tracking(make_df(closes), P)
    assert below["above_ma"] is False        # 빠지지 않고 '아래'로 표시될 뿐
    assert below["last_price"] < below["ma"]


def test_tracking_needs_enough_bars():
    """ma선을 못 구할 만큼 봉이 적으면 스냅샷을 만들지 않는다."""
    assert leader_break.tracking(make_df(rising(bars=P.ma - 5)), P) is None
    assert leader_break.tracking(make_df(rising(bars=P.ma + 5)), P) is not None


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


def test_blocked_excludes_bearish_and_mixed_alignment():
    """정배열만 통과 — 역배열·혼조는 24h 상승률 상위여도 제외."""
    down = list(np.linspace(400.0, 100.0, 600))  # 우하향 → 역배열
    df = _df4h(down)
    assert alignment_of(df) == "역배열"
    assert leader_break.blocked(df, P) is True
    # require_aligned를 끄면 예전 동작 — 역배열은 통과한다
    assert leader_break.blocked(df, Params(require_aligned=False)) is False

    mixed = list(np.linspace(400.0, 100.0, 480)) + list(np.linspace(100.0, 300.0, 120))
    df_mixed = _df4h(mixed)
    assert alignment_of(df_mixed) == "혼조"
    assert leader_break.blocked(df_mixed, P) is True


def test_blocked_excludes_short_history_unless_allowed():
    """상장 80일 미만은 MA480이 없어 배열 판정 불가 — 기본은 제외."""
    new_listing = _df4h(np.linspace(100.0, 200.0, 300))
    assert alignment_of(new_listing) is None
    assert leader_break.blocked(new_listing, P) is True
    assert leader_break.blocked(
        new_listing, Params(allow_short_history=True)) is False
    # 판정이 되는데 정배열이 아닌 경우는 allow_short_history와 무관하게 제외
    down = _df4h(np.linspace(400.0, 100.0, 600))
    assert leader_break.blocked(down, Params(allow_short_history=True)) is True


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


def alignment_of(df):
    from invest_signal.indicators import alignment
    return alignment(df, (120, 240, 480))
