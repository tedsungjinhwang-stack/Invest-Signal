"""주도주 이탈(15m봉) 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

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
    assert ev.detail["ma_period"] == 60 and ev.detail["interval"] == "15m"
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
    assert leader_break.detect(make_df(rising(bars=30)), "TESTUSDT", P) == []


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
