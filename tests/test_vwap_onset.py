"""상승초입(15m 역배열 · 월 상단 > 분기 상단 · 종가 MA960 근처) 테스트."""

import numpy as np
import pandas as pd

from invest_signal import notify
from invest_signal.signals import SignalEvent
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

    기본값(90 → 130)이면 9월 앵커인 월 상단은 130이고, 7~9월을 다 담는
    분기 상단은 그보다 낮다 — **월 상단 > 분기 상단**(②가 선다).
    거꾸로(130 → 90) 주면 ②가 안 선다.
    """
    idx = pd.date_range("2026-07-01", periods=500, freq="4h", tz="UTC")
    price = pd.Series(np.where(idx.month >= 9, sep_price, old_price),
                      index=idx, dtype=float)
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                         "Close": price, "Volume": 1000.0}, index=idx)


def _regime(df4h):
    """이 프레임의 (월 상단, 분기 상단) — 마지막 봉 기준."""
    m = vo.anchored_vwap_bands(df4h, "M", 1.0)[2].iloc[-1]
    q = vo.anchored_vwap_bands(df4h, "Q", 1.0)[2].iloc[-1]
    return float(m), float(q)


def _bear(n=1100, off=None, tail=1):
    """MA240 < MA480 < MA960이 서도록 가파르게 내려오는 시리즈.

    그대로 두면 종가가 MA960보다 ≈30% 아래라 ③이 안 선다. off를 주면
    마지막 tail개 봉을 'MA960 × (1+off)'로 덮는다 — MA는 전체 하락 추세에서
    나오므로 꼬리 몇 봉을 바꿔도 역배열은 유지된다.
    """
    arr = np.linspace(400.0, 200.0, n)
    df = _f15(arr)
    if off is not None:
        ma = df["Close"].rolling(960).mean().iloc[-1]
        df.iloc[-tail:, df.columns.get_loc("Close")] = ma * (1 + off)
    return df


def _gentle(n=1100):
    """아주 완만하게 내려오는 시리즈 — 역배열이면서 종가가 늘 MA960 근처.

    MA960이 서는 첫 봉부터 끝까지 세 조건이 다 선다.
    """
    return _f15(np.linspace(101.0, 100.0, n))


def test_fixture_regimes():
    m, q = _regime(_f4h())
    assert m > q
    m, q = _regime(_f4h(old_price=130.0, sep_price=90.0))
    assert m < q


def test_fires_when_close_is_near_ma960():
    """역배열 · 월상단>분기상단에서 종가가 960선 근처로 처음 들어온 봉."""
    df15 = _bear(off=0.0)
    got = detect(df15, _f4h(), "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["label"] == "상승초입"
    assert d["near_ma"] == 960
    assert abs(d["ma_dist"]) <= 0.02
    assert d["band_top"] > d["band_bottom"]
    assert "band_pos" not in d          # '사이 어디쯤'은 더 이상 없다
    assert d["ret_7d"] is not None      # 대신 종목 수익률을 싣는다


def test_near_from_above_also_fires():
    """선 위 +1%도 '근처'다 — 판정은 절댓값으로 본다."""
    got = detect(_bear(off=0.01), _f4h(), "XUSDT")
    assert len(got) == 1 and got[0].detail["ma_dist"] > 0


def test_far_from_ma960_does_not_fire():
    """허용폭(±2%) 밖이면 아래든 위든 안 알린다."""
    assert detect(_bear(off=-0.05), _f4h(), "XUSDT") == []
    assert detect(_bear(off=0.05), _f4h(), "XUSDT") == []
    assert detect(_bear(), _f4h(), "XUSDT") == []        # ≈ −30%


def test_tolerance_is_configurable():
    p = Params(near_pct=0.06)
    assert len(detect(_bear(off=-0.05), _f4h(), "XUSDT", p)) == 1


def test_month_band_below_quarter_band_does_not_fire():
    """②가 안 서면(월 상단 < 분기 상단) 960선 근처여도 안 알린다."""
    df4h = _f4h(old_price=130.0, sep_price=90.0)
    assert detect(_bear(off=0.0), df4h, "XUSDT") == []


def test_not_bearish_on_15m_does_not_fire():
    """짧은 눈금이 이미 돌았으면(정배열) '초입'이 아니다."""
    up = _f15(np.linspace(200.0, 400.0, 1100))
    ma = up["Close"].rolling(960).mean().iloc[-1]
    up.iloc[-1, up.columns.get_loc("Close")] = ma
    assert detect(up, _f4h(), "XUSDT") == []


def test_only_the_entry_bar_fires():
    """상태 조건이라 매 스캔 다시 알리면 같은 말을 반복한다 — 진입 봉만."""
    df15 = _bear(off=0.0, tail=3)
    got = detect(df15, _f4h(), "XUSDT")
    assert len(got) == 1 and got[0].bar_time == df15.index[-3]


def test_tracking_holds_while_inside_but_not_right_after_entry():
    """방금 들어온 건은 신규 줄이 맡는다 — 같은 봉이 두 칸에 실리면 안 된다."""
    assert tracking(_bear(off=0.0, tail=2), _f4h(), "XUSDT") is None

    # 꼬리가 길면 MA960이 그 사이 움직여 앞쪽 봉이 ±2% 밖으로 나간다 — 20봉.
    df15 = _bear(off=0.0, tail=20)
    held = tracking(df15, _f4h(), "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 20
    assert held.detail["in_capped"] is False
    assert held.detail["last_price"] == df15["Close"].iloc[-1]


def test_tracking_marks_dwell_capped_at_ma960_warmup():
    """MA960이 서는 첫 봉부터 줄곧 안이었으면 그보다 오래였을 수 있다."""
    held = tracking(_gentle(), _f4h(), "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 1100 - 959
    assert held.detail["in_capped"] is True


def test_short_history_is_skipped():
    """MA960이 안 서면 판정 자체가 불가능하다."""
    assert detect(_f15(np.linspace(101.0, 100.0, 500)), _f4h(), "XUSDT") == []


def test_missing_4h_frame_is_skipped():
    assert detect(_bear(off=0.0), None, "XUSDT") == []


def test_band_is_carried_onto_15m_bars():
    """4h 밴드를 15m 인덱스에 계단식으로 얹는다 — 4시간에 한 번 갱신된다."""
    df15, df4h = _bear(), _f4h()
    up = vo._band_on_15m(df15, df4h, "M", 1.0)
    assert up is not None and not up.isna().all()
    assert up.index.equals(df15.index)


def test_line_shows_distance_to_ma960():
    assert notify._band_tags({"ma_dist": -0.0071, "near_ma": 960}) == ["960선 -0.7%"]
    assert notify._band_tags({}) == []
