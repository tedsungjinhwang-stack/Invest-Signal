"""상승초입(15m 역배열 또는 정배열 · 종가 월 VWAP 상단밴드 근처 · 24h 플러스) 테스트."""

import numpy as np
import pandas as pd

from invest_signal import notify
from invest_signal.signals import vwap_onset as vo
from invest_signal.signals.vwap_onset import Params as _Params
from invest_signal.signals.vwap_onset import detect as _detect
from invest_signal.signals.vwap_onset import tracking as _tracking

# 아래 픽스처는 400 → 100 하락 끝이라 15m MA960(≈250)이 월 상단선(≈140)보다 위다 — 10-10에
# 넣은 '상단선 > MA960' 조건을 끄고 나머지 규칙을 본다. 그 조건은 따로 테스트한다.


def Params(**kw):
    kw.setdefault("upper_above_ma", None)
    kw.setdefault("bear_mode", "align")        # 아래 픽스처는 예전 240<480<960 역배열 기준
    return _Params(**kw)


def detect(df15, df4h, symbol, params=None):
    return _detect(df15, df4h, symbol, params or Params())


def tracking(df15, df4h, symbol, params=None):
    return _tracking(df15, df4h, symbol, params or Params())


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
    assert detect(df15, df4h, "XUSDT", Params(bull_enabled=False)) == []   # 10-11 — 역배열 자리만
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
    cfg = {"signal": {"vwap_onset": {"enabled": True, "upper_above_ma": None,
                                     "bear_mode": "align"}}}
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


def test_fire_needs_1h_ignition_and_not_extended():
    """🔥 1h MA20 대비 +4% 이상 · 7일 저점 대비 +25% 이내."""
    from types import SimpleNamespace as NS
    assert vo.fire(NS(ma1h20=0.05, from_7d_low=0.10)) is True
    assert vo.fire(NS(ma1h20=0.03, from_7d_low=0.10)) is False
    assert vo.fire(NS(ma1h20=0.05, from_7d_low=0.40)) is False      # 이미 달린 종목
    assert vo.fire(NS(ma1h20=float("nan"), from_7d_low=0.1)) is None
    assert vo.fire(NS(ma1h20=0.05, from_7d_low=0.1), Params(fire_enabled=False)) is None


def test_detail_carries_fire_flag():
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    got = detect(_tail(_bear(), up), df4h, "XUSDT")
    assert len(got) == 1 and "fire" in got[0].detail


def test_upper_band_must_be_above_ma960():
    """10-10 — 월 VWAP 상단선이 15m MA960 위여야 한다. 하락 끝(MA960 ≈ 250 > 상단 140)은 안 뜬다."""
    df4h = _f4h()
    up = _upper(_bear(), df4h)
    df15 = _tail(_bear(), up)
    assert _stack(df15, (960,))[0] > up
    assert len(detect(df15, df4h, "XUSDT")) == 1                         # 조건 끄면 뜬다
    assert _detect(df15, df4h, "XUSDT", _Params(bear_mode="align")) == []  # 960 — 안 뜬다
    # 정배열(20 → 60, MA960 ≈ 40)은 상단선(≈140)이 MA960 위라 그대로 뜬다
    bull = _tail(_bull(), _upper(_bull(), df4h) * 1.01)
    assert len(_detect(bull, df4h, "XUSDT", _Params(bear_mode="align"))) == 1


def test_bear_band_close_between_240_and_upper_with_long_st_flip():
    """10-11 최종 — 역배열 쪽: 15m MA240 < 종가 < 월 VWAP 상단선 + 15m 장기 수트(30×6) 최근 상승 전환
    (30m로도 볼 수 있다)."""
    from invest_signal.indicators import supertrend_full
    c = np.r_[np.linspace(300.0, 100.0, 900), np.linspace(100.0, 120.0, 200)]
    df15 = _f15(c)
    upper = pd.Series(125.0, index=df15.index)
    ma240 = df15.Close.rolling(240).mean().iloc[-1]
    assert ma240 < df15.Close.iloc[-1] < 125.0
    f30 = df15.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                      "Close": "last", "Volume": "sum"})
    d = supertrend_full(f30, 30, 6.0)["dir"].to_numpy()
    flips = [i for i in range(1, len(d)) if d[i] > 0 and d[i - 1] < 0]
    assert d[-1] > 0 and flips
    age = len(d) - 1 - flips[-1]
    P30 = dict(bear_st_tf="30m")
    ok, known = vo._between(df15, _Params(bear_st_flip_bars=age + 1, **P30), upper)
    assert ok.iloc[-1] and known.iloc[-1]
    assert not vo._between(df15, _Params(bear_st_flip_bars=max(1, age - 1), **P30), upper)[0].iloc[-1]
    # 기본은 15m 장기 수트(10-11 최종)
    d15 = supertrend_full(df15, 30, 6.0)["dir"].to_numpy()
    f15 = [i for i in range(1, len(d15)) if d15[i] > 0 and d15[i - 1] < 0]
    age15 = len(d15) - 1 - f15[-1]
    assert _Params().bear_st_tf == "15m"
    assert vo._between(df15, _Params(bear_st_flip_bars=age15 + 1), upper)[0].iloc[-1]
    assert not vo._between(df15, _Params(bear_st_flip_bars=max(1, age15 - 1)), upper)[0].iloc[-1]
    # 종가가 상단선 위면 사이가 아니다
    assert not vo._between(df15, _Params(bear_st_flip_bars=0),
                           pd.Series(110.0, index=df15.index))[0].iloc[-1]
    # 상단선 대신 MA960(예전 설정)도 그대로 된다
    assert vo._between(df15, _Params(bear_st_flip_bars=0, bear_upper=960))[0].iloc[-1] == bool(
        ma240 < df15.Close.iloc[-1] < df15.Close.rolling(960).mean().iloc[-1])


def test_near_disabled_lets_bear_band_rows_far_below_upper(monkeypatch):
    """10-11 — near_enabled false면 역배열 자리(240 < 종가 < 월상단)는 상단 ±2%를 안 본다."""
    c = np.r_[np.linspace(300.0, 100.0, 900), np.linspace(100.0, 120.0, 200)]
    df15 = _f15(c)
    band = pd.DataFrame({"upper": 150.0, "vwap": 100.0}, index=df15.index)   # 종가 120 = 상단 −20%
    monkeypatch.setattr(vo, "_band_on_15m", lambda *a: band)
    yes = pd.Series(True, index=df15.index)
    monkeypatch.setattr(vo, "_between", lambda *a: (yes, yes))
    df4h = _f4h()
    p = _Params(upper_above_ma=None, min_ret_24h=None)
    assert not vo._state(df15, df4h, p).ok.iloc[-1]
    assert vo._state(df15, df4h, _Params(upper_above_ma=None, min_ret_24h=None,
                                         near_enabled=False, bull_enabled=False)).ok.iloc[-1]


# ── 💥 거래량 폭증(11-25) ─────────────────────────────────────────

def _burst_frame(n=300, spikes=(), price=1.0):
    idx = pd.date_range("2026-11-20", periods=n, freq="15min", tz="UTC")
    v = np.full(n, 1_000.0)
    for t in spikes:
        v[t] = 20_000.0                       # 20배 · 봉 대금 ≈ $20k
    return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price,
                         "Volume": v}, index=idx)


def _fake_state(df, ok):
    return pd.DataFrame({"close": df["Close"], "ok": ok, "bear": ok, "dist": -0.1,
                         "upper": 1.1, "ret_24h": 0.0, "ret_7d": 0.0,
                         "ma1h20": 0.0, "from_7d_low": 0.0}, index=df.index)


def _bp(**kw):
    base = dict(burst_enabled=True, burst_vol_mult=10.0, burst_min_bar_usd=10_000,
                burst_min_turnover_usd=0)
    base.update(kw)
    return vo.Params(**base)


def test_burst_fires_inside_onset_zone(monkeypatch):
    df = _burst_frame(spikes=(298,))
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    got = vo.bursts(df, df, "XUSDT", _bp())
    assert [e.bar_time for e in got] == [df.index[298]]
    e = got[0]
    assert e.detail["stage"] == vo.BURST_STAGE and e.detail["burst_vol_mult"] >= 19
    assert e.dedup_key.endswith("|거래량폭증")
    assert "💥15m거래량 20배" in " ".join(notify._band_tags(e.detail))


def test_burst_needs_zone_bar_usd_and_enabled(monkeypatch):
    df = _burst_frame(spikes=(298,))
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, False))
    assert vo.bursts(df, df, "XUSDT", _bp()) == []
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    assert vo.bursts(df, df, "XUSDT", _bp(burst_min_bar_usd=50_000)) == []
    assert vo.bursts(df, df, "XUSDT", _bp(burst_enabled=False)) == []


def test_burst_rearm_blocks_repeat_within_a_day(monkeypatch):
    df = _burst_frame(spikes=(250, 298))      # 48봉(12h) 간격 — 두 번째는 안 나간다
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    assert vo.bursts(df, df, "XUSDT", _bp()) == []
    assert len(vo.bursts(df, df, "XUSDT", _bp(burst_rearm_bars=24))) == 1


def test_burst_blocked_by_day_turnover_does_not_eat_rearm(monkeypatch):
    # 250봉 폭증은 24h 대금 하한에 걸리고(그땐 대금이 얇다) 298봉 폭증은 넘는다 — 298봉이 나가야 한다
    df = _burst_frame(spikes=(250, 298))
    df.iloc[251:298, df.columns.get_loc("Volume")] = 3_000.0     # 그 사이 대금이 차오른다
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    df.iloc[298, df.columns.get_loc("Volume")] = 60_000.0          # 직전 평균(≈2.5k)의 20배 넘게
    got = vo.bursts(df, df, "XUSDT", _bp(burst_min_turnover_usd=150_000))
    assert [e.bar_time for e in got] == [df.index[298]]


def test_onset_grade_good_and_weak():
    n = 1000
    idx = pd.date_range("2026-11-01", periods=n, freq="15min", tz="UTC")
    c = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    flat = pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": v}, index=idx)
    assert vo.grade(flat, n - 1) == vo.WEAK          # 거래량 그대로 · 24h 고저폭 0.2%
    hot = flat.copy()
    hot.iloc[-96:, hot.columns.get_loc("Volume")] = 3000.0   # 24h 대금 ↑(7일 평균의 1.32배+)
    hot.iloc[-3, hot.columns.get_loc("Volume")] = 40000.0    # 최근 4h 안 거래량 7배+
    assert vo.grade(hot, n - 1) == vo.GOOD
    assert vo.grade(flat, 100) is None                # 이력 부족


def test_burst_grade_b_rule():
    n = 1000
    idx = pd.date_range("2026-11-01", periods=n, freq="15min", tz="UTC")
    c = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    df = pd.DataFrame({"Open": c, "High": c * 1.05, "Low": c * 0.95, "Close": c, "Volume": v}, index=idx)
    df.iloc[-96:, df.columns.get_loc("Volume")] = 2000.0      # 24h 대금 = 7일 평균의 ~2배
    assert vo.burst_grade(df, n - 1, -0.10) == vo.GOOD
    assert vo.burst_grade(df, n - 1, -0.01) is None          # 월상단에 너무 붙음
    df.iloc[-96:, df.columns.get_loc("Volume")] = 1100.0      # 대금 급증 아님
    assert vo.burst_grade(df, n - 1, -0.10) is None


def test_good_grade_sorts_first_weak_last():
    from invest_signal.signals import SignalEvent
    t = pd.Timestamp("2026-11-27", tz="UTC")
    mk = lambda s, dist, g: SignalEvent(symbol=s, signal="vwap_onset", bar_time=t, price=1.0,
                                        detail={"label": "상승초입", "ma960_dist": dist, "grade": g})
    evs = [mk("AUSDT", -0.02, None), mk("BUSDT", 0.03, vo.GOOD), mk("CUSDT", 0.01, vo.WEAK), mk("DUSDT", 0.02, None)]
    got = [e.symbol for e in sorted(evs, key=notify._new_order)]
    assert got[0] == "BUSDT" and got[-1] == "CUSDT"


def test_burst_needs_1h_long_st_up(monkeypatch):
    df = _burst_frame(spikes=(298,))
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    monkeypatch.setattr(vo, "_st1h_up", lambda d: np.zeros(len(d), bool))
    assert vo.bursts(df, df, "XUSDT", _bp(burst_need_1h_st=True)) == []
    monkeypatch.setattr(vo, "_st1h_up", lambda d: np.ones(len(d), bool))
    assert len(vo.bursts(df, df, "XUSDT", _bp(burst_need_1h_st=True))) == 1


def test_blocked_burst_does_not_eat_rearm_for_1h_filter(monkeypatch):
    df = _burst_frame(spikes=(250, 298))
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    up = np.zeros(len(df), bool); up[290:] = True          # 250봉 폭증 땐 1h 장기 수트 하락 → 298봉에서 나간다
    monkeypatch.setattr(vo, "_st1h_up", lambda d: up)
    got = vo.bursts(df, df, "XUSDT", _bp(burst_need_1h_st=True))
    assert [e.bar_time for e in got] == [df.index[298]]


def test_pullback_alert_on_first_382_retrace(monkeypatch):
    n = 300
    idx = pd.date_range("2026-11-20", periods=n, freq="15min", tz="UTC")
    c = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    c[290] = 106.0; v[290] = 20000.0          # 💥
    c[291:293] = [110.0, 110.0]                # 1파 고점 110(1파 시작 ≈ 99.9)
    c[293:] = 105.0                            # 1파의 ~49% 되돌림
    df = pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": v}, index=idx)
    monkeypatch.setattr(vo, "_state", lambda a, b, p: _fake_state(df, True))
    p = _bp(burst_pullback=0.382, grace_bars=4)
    assert vo.pullbacks(df.iloc[:292], df, "XUSDT", p) == []            # 아직 안 내려옴
    ev = vo.pullbacks(df.iloc[:294], df, "XUSDT", p)
    assert len(ev) == 1 and ev[0].bar_time == idx[293]
    d = ev[0].detail
    assert d["stage"] == vo.PULLBACK_STAGE and 0.38 <= d["retrace"] < 0.6 and set(d["levels"]) == {"38.2", "50", "61.8", "78.6"}
    assert vo.pullbacks(df.iloc[:299], df, "XUSDT", p) == []            # 지나간 건 안 낸다
    line = " ".join(notify._band_tags(d))
    assert "↩️💥되돌림" in line and "무효" in line and "61.8%" in line
    assert vo.pullbacks(df.iloc[:294], df, "XUSDT", _bp(burst_pullback=None)) == []
