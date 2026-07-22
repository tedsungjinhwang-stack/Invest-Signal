"""눌림목 시그널 검출 로직 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import pullback
from invest_signal.signals.pullback import Params

P = Params()


def make_df(closes, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c})


def base_setup(hold_bars=60):
    """완만한 하락 → 240선 상향돌파(급등) → 240선 위 유지 구조를 만든다."""
    closes = list(np.linspace(150.0, 100.0, 300))    # 완만한 하락 — 240선이 가격 위
    closes.append(140.0)                             # 급등 봉 = 240선 상향돌파
    for i in range(hold_bars):
        closes.append(140.0 + 0.1 * i)               # 240선 위 유지
    return closes


def dip_below_ma120(closes):
    """현재 시점 120선 바로 아래 종가를 계산한다."""
    m120 = make_df(closes)["Close"].rolling(120).mean().iloc[-1]
    return float(m120) - 2.0


def test_fires_when_close_dips_below_ma120_after_breakout():
    closes = base_setup()
    closes.append(dip_below_ma120(closes))           # 120선 하회 눌림
    df = make_df(closes)
    evs = pullback.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "pullback"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["cross_time"] == df.index[300].isoformat()   # 급등 봉이 돌파 봉


def test_no_breakout_no_signal():
    """240선 상향돌파 없이 120선만 하회하면 미발화 — 그냥 하락장."""
    closes = list(np.linspace(400.0, 100.0, 400))    # 지속 하락: 종가 < 120선
    df = make_df(closes)
    assert pullback.detect(df, "TEST", P) == []


def test_broken_trend_no_signal():
    """돌파 후 종가가 240선 아래로 내려간 적이 있으면(추세 이탈) 미발화."""
    closes = base_setup(hold_bars=10)
    m240 = make_df(closes)["Close"].rolling(240).mean().iloc[-1]
    closes.append(float(m240) - 5)                   # 240선 아래로 마감 → 추세 깨짐
    for _ in range(3):
        closes.append(closes[-1] + 1)                # 240선 아래 머묾
    closes.append(dip_below_ma120(closes))           # 120선 하회
    df = make_df(closes)
    assert pullback.detect(df, "TEST", P) == []


def test_consecutive_dip_bars_alert_once():
    """연속 하회 구간은 첫 봉만 — 두 번째 봉 시점엔 grace로 첫 봉을 돌려준다."""
    closes = base_setup()
    dip = dip_below_ma120(closes)
    closes.append(dip)                               # 첫 하회 봉(트리거)
    closes.append(dip - 0.5)                         # 연속 하회 봉
    df = make_df(closes)
    evs = pullback.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_breakout_window_expiry():
    """돌파가 window보다 오래됐으면 미발화."""
    closes = base_setup(hold_bars=60)                # 돌파 후 60봉 유지 → 간격 61봉
    closes.append(dip_below_ma120(closes))
    df = make_df(closes)
    assert pullback.detect(df, "TEST", Params(breakout_window_bars=40)) == []
    assert len(pullback.detect(df, "TEST", Params(breakout_window_bars=180))) == 1


def test_still_active_requires_between_ma120_and_ma240():
    closes = base_setup(hold_bars=120)           # 120·240선 간격을 충분히 벌린다
    dip = dip_below_ma120(closes)
    closes.append(dip)
    df = make_df(closes)
    ev = pullback.detect(df, "TEST", P)[0]
    closes.append(dip - 0.5)                     # 여전히 120선 아래·240선 위
    assert pullback.still_active(make_df(closes), ev, P)
    m240 = make_df(closes)["Close"].rolling(240).mean().iloc[-1]
    closes.append(float(m240) - 5)               # 240선까지 깨짐 → 추세 이탈
    assert not pullback.still_active(make_df(closes), ev, P)
