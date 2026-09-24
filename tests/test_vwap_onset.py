"""상승초입(15m 역배열 또는 정배열 · 종가 월 VWAP 상단밴드 근처 · 24h 플러스) 테스트."""

import numpy as np
import pandas as pd

from invest_signal import notify
from invest_signal.signals import vwap_onset as vo
from invest_signal.signals.vwap_onset import Params, detect, tracking


def _f15(closes, start="2026-09-05"):
    """15m 프레임 — 달 중순에 두어 월 앵커 밴드가 충분히 쌓인 뒤를 본다."""
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999,
                         "Close": c, "Volume": 1000.0}, index=idx)


def _f4h(center=130.0, swing=10.0, earlier=None):
    """center ± swing를 번갈아 찍는 4h 프레임(7월 1일부터 ≈4달).

    값이 둘뿐이고 거래량이 같으니 월 VWAP은 center, σ는 swing — 월 상단밴드가
    **center + swing**(기본 140)이다. earlier를 주면 해당 월 이전 달들은 그
    값으로 채운다(분기 밴드를 월 밴드와 따로 놀게 할 때).
    """
    idx = pd.date_range("2026-07-01", periods=700, freq="4h", tz="UTC")
    alt = np.where(np.arange(len(idx)) % 2 == 0, center - swing, center + swing)
    if earlier is not None:
        alt = np.where(idx.month < earlier[0], earlier[1], alt)
    price = pd.Series(alt, index=idx, dtype=float)
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                         "Close": price, "Volume": 1000.0}, index=idx)


def _upper(df15, df4h):
    """15m 마지막 봉이 보는 월 상단밴드."""
    return float(vo._band_on_15m(df15, df4h, "M", 1.0).upper.iloc[-1])


def _bear(n=1100, rise=150, start="2026-09-05"):
    """MA240 < MA480 < MA960 — 400에서 100까지 내려왔다가 105로 살짝 반등.

    끝이 밴드(≈140)보다 한참 아래라 그대로는 ②가 안 선다. 꼬리를 밴드
    근처로 덮어 쓰면 24h(96봉 전 ≈102 대비)가 플러스가 되어 ③도 선다.
    반등이 작아 긴 선들의 역배열은 그대로다.
    """
    return _f15(np.concatenate([np.linspace(400.0, 100.0, n - rise),
                                np.linspace(100.0, 105.0, rise)]), start)


def _bull(n=1100):
    """MA120 > MA240 > MA480 — 20에서 60으로 오른다(밴드보다 한참 아래).

    꼬리를 밴드 근처로 **올리면** 짧은 선이 더 올라가 정배열이 유지된다.
    """
    return _f15(np.linspace(20.0, 60.0, n))


def _tail(df, value, tail=1):
    df = df.copy()
    df.iloc[-tail:, df.columns.get_loc("Close")] = value
    return df


def _stack(df, periods):
    c = df["Close"]
    return [c.rolling(k).mean().iloc[-1] for k in periods]


def test_fixture_band():
    assert abs(_upper(_bear(), _f4h()) - 140.0) < 0.5


def test_bearish_near_the_month_band_fires():
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    got = detect(_tail(_bear(), up), df4h, "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["label"] == "상승초입"
    assert d["trend"] == "역배열"
    assert d["near_band"] == "M" and abs(d["band_dist"]) < 1e-9
    assert d["ret_24h"] > 0 and d["ret_7d"] is not None


def test_bullish_near_the_month_band_fires():
    """정배열도 **또는**으로 통과한다."""
    df4h = _f4h()
    up = _upper(_bull(), df4h)
    df15 = _tail(_bull(), up * 1.01)
    a, b, c = _stack(df15, (120, 240, 480))
    assert a > b > c
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["trend"] == "정배열"
    assert 0 < d["band_dist"] <= 0.02


def test_far_from_the_band_does_not_fire():
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    assert detect(_tail(_bear(), up * 1.05), df4h, "XUSDT") == []
    assert detect(_tail(_bear(), up * 0.95), df4h, "XUSDT") == []
    assert detect(_bear(), df4h, "XUSDT") == []


def test_tolerance_is_configurable():
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _tail(_bear(), up * 1.03)
    assert detect(df15, df4h, "XUSDT") == []
    assert len(detect(df15, df4h, "XUSDT", Params(near_pct=0.04))) == 1


def test_quarter_band_no_longer_matters():
    """'월 상단 > 분기 상단' 조건은 뺐다 — 분기 상단이 더 높아도 알린다."""
    df4h = _f4h(earlier=(9, 300.0))       # 7·8월이 훨씬 높아 분기 상단 > 월 상단
    q = float(vo.anchored_vwap_bands(df4h, "Q", 1.0)[2].loc[:"2026-09-16"].iloc[-1])
    up = _upper(_bear(), df4h)
    assert q > up
    assert len(detect(_tail(_bear(), up), df4h, "XUSDT")) == 1


def test_first_month_of_quarter_still_fires():
    """10월 — 월과 분기가 같은 날 시작해도 월 상단만 보니 알림이 나온다."""
    df4h = _f4h()
    df15 = _bear(start="2026-10-08")
    up = _upper(df15, df4h)
    assert len(detect(_tail(df15, up), df4h, "XUSDT")) == 1


def test_month_start_guard_skips_band_hugging_vwap():
    """월초처럼 상단밴드가 VWAP에 붙어 있으면(±2% 구간이 VWAP을 품음) 안 본다."""
    tight = _f4h(swing=1.0)               # 상단 131 · VWAP 130
    up = _upper(_bear(), tight)
    df15 = _tail(_bear(), up)
    assert detect(df15, tight, "XUSDT") == []
    assert len(detect(df15, tight, "XUSDT", Params(band_gap=False))) == 1


def test_mixed_alignment_does_not_fire():
    """역배열도 정배열도 아니면(혼조) 밴드 근처여도 안 알린다."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    arr = np.concatenate([np.linspace(50.0, 300.0, 1000),
                          np.linspace(300.0, up, 100)])
    df15 = _f15(arr)
    s120, s240, s480, s960 = _stack(df15, (120, 240, 480, 960))
    assert not (s120 > s240 > s480)
    assert not (s240 < s480 < s960)
    p = Params(min_ret_24h=None)            # 끝이 내려오는 모양이라 ③은 뺀다
    assert detect(df15, df4h, "XUSDT", p) == []
    assert tracking(df15, df4h, "XUSDT", p) is None


def test_only_the_entry_bar_fires():
    """상태 조건이라 매 스캔 다시 알리면 같은 말을 반복한다 — 진입 봉만."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _tail(_bear(), up, tail=3)
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1 and got[0].bar_time == df15.index[-3]


def test_reentry_within_rearm_window_is_tracked_not_realerted():
    """±2% 경계에서 깜빡이는 종목 — 24h 안에 다시 들어오면 새로 안 알린다."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _bear()
    col = df15.columns.get_loc("Close")
    df15.iloc[-30:-20, col] = up         # 들어왔다가
    df15.iloc[-3:, col] = up             # 나갔다(원래 값 ≈105) 다시 들어옴
    assert detect(df15, df4h, "XUSDT") == []
    held = tracking(df15, df4h, "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 30          # 처음 들어온 봉부터 잰다
    assert held.detail["in_capped"] is False

    # rearm을 끄면(1봉) 다시 들어온 봉이 새 알림이 된다
    got = detect(df15, df4h, "XUSDT", Params(rearm_bars=1))
    assert len(got) == 1 and got[0].bar_time == df15.index[-3]


def test_tracking_holds_while_inside_but_not_right_after_entry():
    """방금 들어온 건은 신규 줄이 맡는다 — 같은 봉이 두 칸에 실리면 안 된다."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    assert tracking(_tail(_bear(), up, tail=2), df4h, "XUSDT") is None

    held = tracking(_tail(_bear(), up, tail=20), df4h, "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 20
    assert held.detail["in_capped"] is False
    assert held.detail["last_price"] == up


def test_bearish_dwell_is_capped_at_ma960_warmup():
    """1,000봉이면 MA960은 마지막 41봉만 선다 — 그 앞은 역배열을 못 잰다."""
    df4h = _f4h()
    up = _upper(_bear(1000), df4h)
    held = tracking(_tail(_bear(1000), up, tail=60), df4h, "XUSDT")
    assert held is not None
    assert held.detail["trend"] == "역배열"
    assert held.detail["in_bars"] == 41
    assert held.detail["in_capped"] is True
    assert notify._dwell_tag(held.detail) == "10h+째"


def test_falling_into_the_band_does_not_fire():
    """③ 24h 동안 밀려 내려와 밴드에 걸린 종목은 '초입'이 아니다."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _tail(_bear(), up)
    df15.iloc[-97, df15.columns.get_loc("Close")] = up * 1.08   # 24h 전엔 더 위
    assert detect(df15, df4h, "XUSDT") == []
    got = detect(df15, df4h, "XUSDT", Params(min_ret_24h=-0.10))
    assert len(got) == 1 and got[0].detail["ret_24h"] < 0
    assert len(detect(df15, df4h, "XUSDT", Params(min_ret_24h=None))) == 1


def test_scanner_drops_lines_whose_ticker_24h_is_negative():
    """판정은 15m 96봉 기준이라도, 줄에 찍힐 티커 24h가 마이너스면 뺀다."""
    from invest_signal import scanner
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _tail(_bear(), up)
    cfg = {"signal": {"vwap_onset": {"enabled": True}}}
    rising = {"XUSDT": {"quote_volume": 5e6, "change_pct": 0.03}}
    falling = {"XUSDT": {"quote_volume": 5e6, "change_pct": -0.004}}
    logs = []
    ev, _ = scanner._scan_vwap_onset(cfg, {"XUSDT": df15}, {"XUSDT": df4h},
                                     rising, logs.append)
    assert len(ev) == 1 and ev[0].detail["gain_24h"] == 0.03
    ev, _ = scanner._scan_vwap_onset(cfg, {"XUSDT": df15}, {"XUSDT": df4h},
                                     falling, logs.append)
    assert ev == []
    assert "24h 하락 1건 제외" in logs[-1]


def test_short_history_is_skipped():
    """정배열(MA480)조차 안 서면 판정 자체가 불가능하다."""
    df4h = _f4h()
    up = _upper(_bull(400), df4h)
    assert detect(_tail(_bull(400), up), df4h, "XUSDT") == []


def test_missing_4h_frame_is_skipped():
    assert detect(_tail(_bear(), 140.0), None, "XUSDT") == []


def test_band_is_carried_onto_15m_bars():
    """4h 밴드를 15m 인덱스에 계단식으로 얹는다 — 4시간에 한 번 갱신된다."""
    df15, df4h = _bear(), _f4h()
    band = vo._band_on_15m(df15, df4h, "M", 1.0)
    assert band is not None and not band.upper.isna().all()
    assert band.index.equals(df15.index)
    assert (band.upper >= band.vwap).all()


def test_line_shows_alignment_and_band_distance():
    d = {"trend": "정배열", "near_band": "M", "band_dist": -0.008}
    assert notify._band_tags(d) == ["↑정배열", "월상단 -0.8%"]
    d = {"trend": "역배열", "near_band": "M", "band_dist": 0.011}
    assert notify._band_tags(d) == ["↓역배열", "월상단 +1.1%"]
    assert notify._band_tags({}) == []


def test_dwell_tag():
    assert notify._dwell_tag({"in_bars": 16}) == "4h째"
    assert notify._dwell_tag({"in_bars": 144}) == "1.5일째"
    assert notify._dwell_tag({"in_bars": 41, "in_capped": True}) == "10h+째"
