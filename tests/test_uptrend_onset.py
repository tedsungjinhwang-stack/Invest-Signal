"""상승초입 시그널 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

import numpy as np
import pandas as pd

from invest_signal.signals import uptrend_onset
from invest_signal.signals.uptrend_onset import Params

DECLINE = 600           # 역배열을 만들 하락 구간 봉 수


def make_df(closes, highs):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    h = pd.Series([float(x) for x in highs], index=idx)
    return pd.DataFrame({
        "Open": c, "High": np.maximum(c, h), "Low": np.minimum(c, h), "Close": c,
    })


def base_decline():
    """400→100 선형 하락 — 끝에서 120<240<480 역배열, 종가는 60선 아래."""
    closes = list(np.linspace(400.0, 100.0, DECLINE))
    highs = closes.copy()
    return closes, highs


def add_touch(closes, highs):
    """다음 봉에서 고가만 240선까지 찍는 터치 봉을 추가."""
    t = len(closes)
    closes.append(closes[-1])
    ma240 = float(np.mean((closes[t - 239:t] + [closes[t]])))
    highs.append(ma240 + 1.0)
    return t


def add_sideways(closes, highs, bars, level=None):
    """60선 위·240선 아래에서 옆으로 기는 구간(터치 없음)."""
    level = level if level is not None else closes[-1] + 20.0
    for i in range(bars):
        closes.append(level + 0.01 * i)
        highs.append(closes[-1])


def add_drop(closes, highs, price):
    closes.append(price)
    highs.append(price)


def test_fires_on_first_close_below_ma60_after_touch():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs, closes[-1] - 5)      # 하락 지속 → 종가 < 60선
    df = make_df(closes, highs)
    ev = uptrend_onset.detect(df, "TEST", Params())
    assert ev is not None
    assert ev.signal == "uptrend_onset"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["touch_time"] == df.index[-2].isoformat()


def test_no_touch_no_signal():
    closes, highs = base_decline()
    df = make_df(closes, highs)                  # 종가는 늘 60선 아래지만 터치가 없음
    assert uptrend_onset.detect(df, "TEST", Params()) is None


def test_stale_trigger_respects_grace():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs, closes[-1] - 5)      # 트리거 봉
    add_drop(closes, highs, closes[-1] - 5)      # 그 뒤 1봉 경과
    df = make_df(closes, highs)
    ev = uptrend_onset.detect(df, "TEST", Params(grace_bars=1))
    assert ev is not None and ev.bar_time == df.index[-2]   # 트리거 봉 기준
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=0)) is None

    add_drop(closes, highs, closes[-1] - 5)      # 2봉 경과 → grace=1로도 stale
    df = make_df(closes, highs)
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=1)) is None


def test_touch_expires_after_window():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_sideways(closes, highs, bars=65)         # 60봉 윈도우 초과
    add_drop(closes, highs, 90.0)                # 이제서야 60선 이탈
    df = make_df(closes, highs)
    assert uptrend_onset.detect(df, "TEST", Params(touch_window_bars=60)) is None


def test_touch_within_window_fires():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_sideways(closes, highs, bars=20)         # 윈도우 안
    add_drop(closes, highs, 90.0)
    df = make_df(closes, highs)
    ev = uptrend_onset.detect(df, "TEST", Params(touch_window_bars=60))
    assert ev is not None and ev.bar_time == df.index[-1]


def test_insufficient_data_returns_none():
    closes = list(np.linspace(200.0, 100.0, 100))
    df = make_df(closes, closes)
    assert uptrend_onset.detect(df, "TEST", Params()) is None
