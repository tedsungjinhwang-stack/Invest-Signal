"""상승초입 시그널 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

import numpy as np
import pandas as pd

from invest_signal.signals import uptrend_onset
from invest_signal.signals.uptrend_onset import Params

DECLINE = 600           # 역배열을 만들 하락 구간 봉 수

# 기본 파라미터 — 합성 데이터에 Volume이 없으면 QVWAP 조건은 자동 통과
P = Params()


def make_df(closes, highs, volumes=None, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    h = pd.Series([float(x) for x in highs], index=idx)
    data = {"Open": c, "High": np.maximum(c, h), "Low": np.minimum(c, h), "Close": c}
    if volumes is not None:
        data["Volume"] = [float(v) for v in volumes]
    return pd.DataFrame(data)


def base_decline(bars=DECLINE):
    """400→100 선형 하락 — 끝에서 120<240<480 역배열, 종가는 60선 아래."""
    closes = list(np.linspace(400.0, 100.0, bars))
    highs = closes.copy()
    return closes, highs


def add_touch(closes, highs, spike_to=None):
    """다음 봉에서 고가만 240선까지(또는 spike_to까지) 찍는 터치 봉을 추가."""
    t = len(closes)
    closes.append(closes[-1])
    ma240 = float(np.mean(closes[t - 239:t + 1]))
    highs.append(max(ma240 + 1.0, spike_to or 0.0))
    return t


def add_sideways(closes, highs, bars, level=None):
    """60선 위·240선 아래에서 옆으로 기는 구간(터치 없음)."""
    level = level if level is not None else closes[-1] + 20.0
    for i in range(bars):
        closes.append(level + 0.01 * i)
        highs.append(closes[-1])


def add_drop(closes, highs, price=None):
    price = price if price is not None else closes[-1] - 5
    closes.append(price)
    highs.append(price)


def test_fires_on_first_close_below_ma60_after_touch():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 하락 지속 → 종가 < 60선
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "uptrend_onset"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["touch_time"] == df.index[-2].isoformat()


def test_no_touch_no_signal():
    closes, highs = base_decline()
    df = make_df(closes, highs)                  # 종가는 늘 60선 아래지만 터치가 없음
    assert uptrend_onset.detect(df, "TEST", P) == []


def test_touch_without_bearish_alignment_no_signal():
    """상승장(정배열)에서는 고가가 240선 위여도 시그널이 없어야 한다."""
    closes = list(np.linspace(100.0, 400.0, DECLINE))   # 상승 → 정배열
    highs = closes.copy()
    closes.append(closes[-1] * 0.8)              # 급락 → 종가 < 60선
    highs.append(closes[-1] * 1.3)               # 고가는 240선보다 훨씬 위
    df = make_df(closes, highs)
    m60 = df["Close"].rolling(60).mean().iloc[-1]
    m240 = df["Close"].rolling(240).mean().iloc[-1]
    assert df["Close"].iloc[-1] < m60 and df["High"].iloc[-1] >= m240   # 전제 확인
    assert uptrend_onset.detect(df, "TEST", P) == []


def test_stale_trigger_respects_grace():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 트리거 봉
    add_drop(closes, highs)                      # 그 뒤 1봉 경과
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]   # 트리거 봉 기준
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=0)) == []

    add_drop(closes, highs)                      # 2봉 경과 → grace=1로도 stale
    df = make_df(closes, highs)
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=1)) == []


def test_touch_window_boundary_exact():
    """터치 후 정확히 60봉째 이탈은 발화, 61봉째는 미발화."""
    def build(sideways):
        closes, highs = base_decline()
        add_touch(closes, highs)
        add_sideways(closes, highs, bars=sideways)
        add_drop(closes, highs, 90.0)
        return make_df(closes, highs)

    p = Params(touch_window_bars=60)
    df60 = build(59)                             # 터치→트리거 간격 = 60봉
    evs = uptrend_onset.detect(df60, "TEST", p)
    assert len(evs) == 1 and evs[0].bar_time == df60.index[-1]
    df61 = build(60)                             # 간격 61봉 → 윈도우 초과
    assert uptrend_onset.detect(df61, "TEST", p) == []


def test_rearm_on_new_touch_after_fire():
    """발화 → 60선 위 복귀 → 새 240 터치 → 다시 이탈하면 재발화."""
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 1차 발화 봉
    add_sideways(closes, highs, bars=5)          # 60선 위로 복귀
    add_touch(closes, highs)                     # 새 터치
    add_drop(closes, highs, 90.0)                # 2차 이탈
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", P)
    assert len(evs) == 1
    assert evs[0].bar_time == df.index[-1]
    assert evs[0].detail["touch_time"] == df.index[-2].isoformat()


def test_missed_bar_trigger_found_even_if_it_touches():
    """grace 소급 판정: 놓친 트리거 봉 자체가 240 터치여도 잡아야 한다."""
    closes, highs = base_decline()
    add_touch(closes, highs)                     # 원래 터치
    t = len(closes)                              # 트리거 봉인데 위꼬리가 240선도 찍음
    closes.append(closes[-1] - 5)
    ma240 = float(np.mean(closes[t - 239:t + 1]))
    highs.append(ma240 + 1.0)
    add_sideways(closes, highs, bars=1)          # 다음 봉은 60선 위 복귀
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_qvwap_condition_gates_signal():
    """트리거 종가가 분기 VWAP 아래면 미발화, 위면 발화."""
    # 520봉(≈87일) — 2025-01-01 시작이라 전 구간이 1분기 안
    closes, highs = base_decline(bars=520)
    add_touch(closes, highs)
    # 트리거: 직전 저점(100)보다 위지만 60선(≈116)보다는 아래로 마감
    add_drop(closes, highs, 110.0)
    # 균등 거래량 → QVWAP ≈ 분기 전체 평균(≈250) » 트리거 종가 → 탈락
    vols_eq = [1.0] * len(closes)
    df_below = make_df(closes, highs, volumes=vols_eq)
    assert uptrend_onset.detect(df_below, "TEST", Params(qvwap_condition=True)) == []
    # 같은 데이터라도 조건을 끄면 발화
    evs = uptrend_onset.detect(df_below, "TEST", Params(qvwap_condition=False))
    assert len(evs) == 1 and evs[0].detail["above_qvwap"] is False
    # 최근 저가 구간(~100)에 거래량 몰빵 → QVWAP이 트리거 종가(110) 아래로 → 발화
    vols_late = [1.0] * len(closes)
    for i in range(len(closes) - 30, len(closes)):
        vols_late[i] = 5000.0
    df_above = make_df(closes, highs, volumes=vols_late)
    evs = uptrend_onset.detect(df_above, "TEST", Params(qvwap_condition=True))
    assert len(evs) == 1 and evs[0].detail["above_qvwap"] is True


def test_insufficient_data_returns_empty():
    closes = list(np.linspace(200.0, 100.0, 100))
    df = make_df(closes, closes)
    assert uptrend_onset.detect(df, "TEST", P) == []
