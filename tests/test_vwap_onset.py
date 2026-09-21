"""상승초입(분기 상단 위 · 월 상단 아래 · 15m 역배열) 테스트."""

import numpy as np
import pandas as pd

from invest_signal.signals import vwap_onset as vo
from invest_signal.signals.vwap_onset import Params, detect, tracking


def _f15(closes):
    """15m 프레임 — **9월 안**에 있어야 9월 앵커 밴드가 얹힌다."""
    idx = pd.date_range("2026-09-05", periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999,
                         "Close": c, "Volume": 1000.0}, index=idx)


def _f4h(old_price=90.0, sep_price=130.0):
    """7·8월은 old_price, 9월은 sep_price인 연속 4h 프레임.

    분기(Q3)는 세 달을 다 담으므로 분기 상단은 두 값 사이 어딘가에 서고,
    9월 앵커인 월 상단은 sep_price가 된다(9월 값이 일정해 σ=0).
    """
    idx = pd.date_range("2026-07-01", periods=500, freq="4h", tz="UTC")
    price = pd.Series(np.where(idx.month >= 9, sep_price, old_price),
                      index=idx, dtype=float)
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                         "Close": price, "Volume": 1000.0}, index=idx)


def _bands(df4h):
    """이 프레임의 (분기 상단, 월 상단) — 마지막 봉 기준."""
    q = vo.anchored_vwap_bands(df4h, "Q", 1.0)[2].iloc[-1]
    m = vo.anchored_vwap_bands(df4h, "M", 1.0)[2].iloc[-1]
    assert q < m, "픽스처가 잘못됐다 — 분기 상단이 월 상단보다 낮아야 한다"
    return float(q), float(m)


def _bear(n=1100, last=None, tail=1):
    """MA240 < MA480 < MA960이 서도록 계속 내려오는 시리즈.

    마지막 tail개 봉만 last로 덮는다 — MA는 전체 하락 추세에서 나오므로
    꼬리 몇 봉을 바꿔도 역배열은 유지된다.
    """
    arr = np.linspace(400.0, 200.0, n)
    df = _f15(arr)
    if last is not None:
        df.iloc[-tail:, df.columns.get_loc("Close")] = last
    return df


def test_fires_between_the_two_upper_bands():
    """분기 상단 위 · 월 상단 아래 — 둘 사이에 처음 들어온 봉."""
    df4h = _f4h()
    lo, hi = _bands(df4h)
    df15 = _bear(last=(lo + hi) / 2)
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["label"] == "상승초입"
    assert d["band_above"] < got[0].price < d["band_below"]
    # 구간 안 어디쯤인지가 0~1로 나온다(0이면 분기 상단, 1이면 월 상단).
    # 정확히 0.5가 아닌 건 밴드가 4h봉마다 갱신되어 15m 마지막 봉이 보는
    # 값과 4h 마지막 봉의 값이 조금 다르기 때문이다 — 계단식 전방 채움.
    assert 0 < d["band_pos"] < 1
    assert round(d["to_upper"], 4) == round(d["band_below"] / got[0].price - 1, 4)


def test_below_the_quarter_band_does_not_fire():
    """분기 상단도 못 넘었으면 아직 바닥 구간이다."""
    lo, _ = _bands(_f4h())
    assert detect(_bear(last=lo * 0.9), _f4h(), "XUSDT") == []


def test_above_the_month_band_does_not_fire():
    """월 상단까지 넘었으면 '아직 안 갔다'가 아니다 — 이미 간 것이다."""
    _, hi = _bands(_f4h())
    assert detect(_bear(last=hi * 1.1), _f4h(), "XUSDT") == []


def test_not_bearish_on_15m_does_not_fire():
    """짧은 눈금이 이미 돌았으면 '초입'이 아니다."""
    df4h = _f4h()
    lo, hi = _bands(df4h)
    up = _f15(np.linspace(200.0, 400.0, 1100))      # 정배열
    up.iloc[-1, up.columns.get_loc("Close")] = (lo + hi) / 2
    assert detect(up, df4h, "XUSDT") == []


def test_only_the_entry_bar_fires():
    """상태 조건이라 매 스캔 다시 알리면 같은 말을 반복한다 — 진입 봉만."""
    df4h = _f4h()
    lo, hi = _bands(df4h)
    df15 = _bear(last=(lo + hi) / 2, tail=3)
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1 and got[0].bar_time == df15.index[-3]


def test_tracking_holds_while_inside_but_not_right_after_entry():
    """방금 들어온 건은 신규 줄이 맡는다 — 같은 봉이 두 칸에 실리면 안 된다."""
    df4h = _f4h()
    lo, hi = _bands(df4h)
    mid = (lo + hi) / 2
    assert tracking(_bear(last=mid, tail=2), df4h, "XUSDT") is None

    held = tracking(_bear(last=mid, tail=40), df4h, "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 40
    assert held.detail["last_price"] == mid


def test_short_history_is_skipped():
    """MA960이 안 서면 판정 자체가 불가능하다."""
    assert detect(_bear(n=500), _f4h(), "XUSDT") == []


def test_missing_4h_frame_is_skipped():
    assert detect(_bear(last=100.0), None, "XUSDT") == []


def test_band_is_carried_onto_15m_bars():
    """4h 밴드를 15m 인덱스에 계단식으로 얹는다 — 4시간에 한 번 갱신된다."""
    df15, df4h = _bear(), _f4h()
    up = vo._band_on_15m(df15, df4h, "M", 1.0)
    assert up is not None and not up.isna().all()
    assert up.index.equals(df15.index)
