"""하락전환(CHoCH) 시그널 검출 로직 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import downtrend_reversal
from invest_signal.signals.downtrend_reversal import Params

P = Params()


def make_df(closes, lows=None, start="2025-01-01", volumes=None):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    lo = pd.Series([float(x) for x in (lows or closes)], index=idx)
    data = {"Open": c, "High": np.maximum(c, lo),
            "Low": np.minimum(c, lo), "Close": c}
    if volumes is not None:
        data["Volume"] = [float(v) for v in volumes]
    return pd.DataFrame(data)


def uptrend_with_completed_dip(dip_wick=None, recover_bars=5):
    """상승 → 60선 아래 눌림(저점 형성) → 60선 위 복귀 구조를 만든다.

    반환: (closes, lows, level) — level은 눌림 구간 최저 저가.
    """
    closes = list(np.linspace(100.0, 200.0, 200))    # 상승 — 종가가 60선 위
    lows = closes.copy()
    m60 = float(pd.Series(closes).rolling(60).mean().iloc[-1])    # ≈ 185
    dip = m60 - 10.0
    level = dip_wick if dip_wick is not None else dip - 5.0
    closes += [dip, dip - 2, dip]                     # 눌림 3봉 (종가 < 60선)
    lows += [dip, level, dip]                         # 가운데 봉 꼬리가 최저점
    for i in range(recover_bars):                     # 60선 위 복귀 → 눌림 완성
        closes.append(m60 + 15.0 + i)
        lows.append(closes[-1])
    return closes, lows, level


def test_fires_on_first_close_below_prior_dip_low():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1.0)                        # 직전저점 하향돌파 마감
    lows.append(level - 1.0)
    df = make_df(closes, lows)
    evs = downtrend_reversal.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "downtrend_reversal"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["broken_low"] == level
    # 저점이 만들어진 봉(눌림 가운데 봉)의 시각이 기록된다
    assert ev.detail["low_time"] == df.index[len(closes) - 1 - 5 - 2].isoformat()


def test_no_fire_when_close_stays_above_level():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level + 1.0)                        # 저점 위에서 마감 — 돌파 아님
    lows.append(level - 3.0)                          # 꼬리만 뚫은 것도 돌파 아님(종가 기준)
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", P) == []


def test_no_fire_on_new_lows_inside_same_dip():
    """복귀 없이 진행 중인 눌림에서 저점 갱신하는 건 CHoCH가 아니다."""
    closes, lows, _ = uptrend_with_completed_dip(recover_bars=0)  # 복귀 없음
    closes.append(closes[-1] - 30.0)                  # 같은 눌림에서 폭락
    lows.append(closes[-1])
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", P) == []


def test_fires_once_then_silent():
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1.0)                        # 돌파 봉(트리거)
    lows.append(level - 1.0)
    closes.append(level - 2.0)                        # 계속 그 아래 — 재발화 없음
    lows.append(level - 2.0)
    df = make_df(closes, lows)
    evs = downtrend_reversal.detect(df, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]
    assert downtrend_reversal.detect(df, "TEST", Params(grace_bars=0)) == []


def test_qvwap_condition_gates_choch():
    """돌파 봉 종가가 QVWAP 위면 대기 — 아래로 내려오는 봉에서 발화."""
    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1.0)      # 돌파 봉 ≈169 — 균등 거래량 QVWAP(≈151) 위 → 대기
    lows.append(level - 1.0)
    vols = [1.0] * len(closes)
    df_wait = make_df(closes, lows, volumes=vols)
    assert downtrend_reversal.detect(df_wait, "TEST", P) == []
    # 조건 끄면 저점 이탈만으로 발화
    assert len(downtrend_reversal.detect(df_wait, "TEST",
                                         Params(qvwap_condition=False))) == 1
    # 종가가 QVWAP 아래로 내려오는 봉에서 발화 — 위에서 대기하던 돌파 봉은
    # '첫 봉' 판정을 막지 않는다
    closes.append(145.0)
    lows.append(145.0)
    vols.append(1.0)
    df_fire = make_df(closes, lows, volumes=vols)
    evs = downtrend_reversal.detect(df_fire, "TEST", Params(grace_bars=1))
    assert len(evs) == 1 and evs[0].bar_time == df_fire.index[-1]


def test_qvwap_line_above_candle_fires_immediately():
    """복귀 구간(고가)에 거래량 몰빵 → QVWAP 선이 캔들 위 → 돌파 봉 즉시 발화."""
    closes, lows, level = uptrend_with_completed_dip()
    vols = [1.0] * len(closes)
    for i in range(len(closes) - 5, len(closes)):     # 복귀 5봉(≈200)에 몰빵 → QVWAP ≈ 200
        vols[i] = 100000.0
    closes.append(level - 1.0)
    lows.append(level - 1.0)
    vols.append(1.0)
    df = make_df(closes, lows, volumes=vols)
    evs = downtrend_reversal.detect(df, "TEST", P)
    assert len(evs) == 1 and evs[0].bar_time == df.index[-1]


def test_lookback_expiry():
    closes, lows, level = uptrend_with_completed_dip(recover_bars=60)
    closes.append(level - 1.0)
    lows.append(level - 1.0)
    df = make_df(closes, lows)
    assert downtrend_reversal.detect(df, "TEST", Params(lookback_bars=40)) == []
    assert len(downtrend_reversal.detect(df, "TEST", Params(lookback_bars=180))) == 1


def test_still_active_drops_when_supertrend_flips_up():
    """하락전환은 거울이다 — 수퍼트렌드가 **상승**으로 뒤집히면 추적 해제."""
    from invest_signal.indicators import supertrend
    from invest_signal.signals.downtrend_reversal import Params as DParams

    closes, lows, level = uptrend_with_completed_dip()
    closes.append(level - 1)                       # 직전저점 하향돌파
    lows.append(level - 2)
    df = make_df(closes, lows)
    p = DParams(qvwap_condition=False, supertrend_exit=False)
    ev = downtrend_reversal.detect(df, "TEST", p)[0]
    assert downtrend_reversal.still_active(df, ev, p)   # 조건 끔 — 유지

    # 급등 봉을 붙여 수퍼트렌드를 상승으로 뒤집는다
    closes.append(level * 1.6)
    lows.append(level - 2)
    up = make_df(closes, lows)
    assert supertrend(up, 22, 3.0).dropna().iloc[-1] == 1
    assert not downtrend_reversal.still_active(
        up, ev, DParams(qvwap_condition=False))        # 켜면 제거
