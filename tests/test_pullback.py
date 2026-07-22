"""눌림목 시그널 검출 로직 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import pullback
from invest_signal.signals.pullback import Params
from invest_signal.signals import SignalEvent

P = Params()


def make_df(closes, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c})


def uptrend_regime(bars=600):
    """장기 상승 — 240선이 480선 위(눌림목 구간), 종가는 120선 위."""
    return list(np.linspace(100.0, 400.0, bars))


def recovery_no_regime():
    """급락 후 짧은 반등 — 종가는 120선 위지만 아직 240선 < 480선."""
    closes = list(np.linspace(400.0, 100.0, 500))
    closes += list(np.linspace(100.0, 180.0, 100))
    return closes


def dip_below_ma120(closes):
    m120 = make_df(closes)["Close"].rolling(120).mean().iloc[-1]
    return float(m120) - 5.0


def test_fires_on_dip_in_regime():
    closes = uptrend_regime()
    df0 = make_df(closes)
    m240 = df0["Close"].rolling(240).mean().iloc[-1]
    m480 = df0["Close"].rolling(480).mean().iloc[-1]
    assert m240 > m480                               # 전제: 눌림목 구간
    closes.append(dip_below_ma120(closes))
    df = make_df(closes)
    evs = pullback.detect(df, "TEST", P)
    assert len(evs) == 1
    assert evs[0].signal == "pullback"
    assert evs[0].bar_time == df.index[-1]


def test_no_fire_outside_regime():
    """240선이 아직 480선 아래면(상승초입 구간) 120선 하회해도 미발화."""
    closes = recovery_no_regime()
    df0 = make_df(closes)
    m240 = df0["Close"].rolling(240).mean().iloc[-1]
    m480 = df0["Close"].rolling(480).mean().iloc[-1]
    assert m240 < m480                               # 전제: 구간 아님
    closes.append(dip_below_ma120(closes))
    df = make_df(closes)
    assert pullback.detect(df, "TEST", P) == []


def test_consecutive_dip_bars_alert_once():
    closes = uptrend_regime()
    dip = dip_below_ma120(closes)
    closes.append(dip)
    closes.append(dip - 0.5)
    df = make_df(closes)
    evs = pullback.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_still_active_requires_below_120_and_regime():
    closes = uptrend_regime()
    dip = dip_below_ma120(closes)
    closes.append(dip)
    df = make_df(closes)
    ev = pullback.detect(df, "TEST", P)[0]
    closes.append(dip - 0.5)                         # 계속 120선 아래·구간 유지
    assert pullback.still_active(make_df(closes), ev, P)
    m120 = make_df(closes)["Close"].rolling(120).mean().iloc[-1]
    closes.append(float(m120) + 10)                  # 120선 위 복귀 → 종료
    assert not pullback.still_active(make_df(closes), ev, P)


def test_still_active_false_when_regime_lost():
    """240선 < 480선인 데이터에서는 유지 판정도 거짓."""
    closes = recovery_no_regime()
    closes.append(dip_below_ma120(closes))
    df = make_df(closes)
    ev = SignalEvent(symbol="TEST", signal="pullback", bar_time=df.index[-1],
                     price=float(df["Close"].iloc[-1]), detail={})
    assert not pullback.still_active(df, ev, P)
