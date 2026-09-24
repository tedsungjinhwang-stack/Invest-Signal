"""상승초입(15m 역배열 또는 정배열 · 월 상단 > 분기 상단 · 상단밴드 근처 · 24h 플러스) 테스트."""

import numpy as np
import pandas as pd

from invest_signal import notify
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

    9월 값이 일정하니 월 상단은 sep_price 그대로(σ=0)이고, 7~9월을 다 담는
    분기 상단은 두 값 사이에 선다. 기본값(90 → 130)이면 **월 상단 130 >
    분기 상단 ≈114** — ②가 서고 두 밴드가 12%쯤 떨어져 있다.
    """
    idx = pd.date_range("2026-07-01", periods=500, freq="4h", tz="UTC")
    price = pd.Series(np.where(idx.month >= 9, sep_price, old_price),
                      index=idx, dtype=float)
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                         "Close": price, "Volume": 1000.0}, index=idx)


def _bands(df15, df4h):
    """15m 마지막 봉이 보는 (월 상단, 분기 상단)."""
    m = vo._band_on_15m(df15, df4h, "M", 1.0).iloc[-1]
    q = vo._band_on_15m(df15, df4h, "Q", 1.0).iloc[-1]
    return float(m), float(q)


def _bear(n=1100, rise=150):
    """MA240 < MA480 < MA960 — 400에서 100까지 내려왔다가 105로 살짝 반등.

    끝이 밴드(≈114~130)보다 한참 아래라 그대로는 ③이 안 선다. 꼬리를 밴드
    근처로 덮어 쓰면 24h(96봉 전 ≈102 대비)가 플러스가 되어 ④도 선다.
    반등이 작아 긴 선들의 역배열은 그대로다.
    """
    return _f15(np.concatenate([np.linspace(400.0, 100.0, n - rise),
                                np.linspace(100.0, 105.0, rise)]))


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


def test_fixture_regimes():
    m, q = _bands(_bear(), _f4h())
    assert m > q * 1.08                 # ② 선다 · 두 밴드가 떨어져 있다
    m, q = _bands(_bear(), _f4h(old_price=130.0, sep_price=90.0))
    assert m < q                        # ② 안 선다


def test_bearish_near_the_month_band_fires():
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    got = detect(_tail(_bear(), m), df4h, "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["label"] == "상승초입"
    assert d["trend"] == "역배열"
    assert d["near_band"] == "M" and abs(d["band_dist"]) < 1e-9
    assert d["band_top"] > d["band_bottom"]
    assert d["ret_7d"] is not None


def test_bullish_near_the_quarter_band_fires():
    """정배열도 **또는**으로 통과한다 — 분기 상단 하나에만 붙어도 된다."""
    df4h = _f4h()
    _, q = _bands(_bull(), df4h)
    df15 = _tail(_bull(), q * 1.01)
    a, b, c = _stack(df15, (120, 240, 480))
    assert a > b > c
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1
    d = got[0].detail
    assert d["trend"] == "정배열"
    assert d["near_band"] == "Q" and 0 < d["band_dist"] <= 0.02


def test_between_the_bands_but_near_neither_does_not_fire():
    """두 밴드 **사이**라도 어느 쪽에도 ±2% 안이 아니면 안 알린다."""
    df4h = _f4h()
    m, q = _bands(_bear(), df4h)
    assert detect(_tail(_bear(), (m + q) / 2), df4h, "XUSDT") == []


def test_far_from_both_bands_does_not_fire():
    df4h = _f4h()
    m, q = _bands(_bear(), df4h)
    assert detect(_tail(_bear(), m * 1.05), df4h, "XUSDT") == []
    assert detect(_tail(_bear(), q * 0.95), df4h, "XUSDT") == []
    assert detect(_bear(), df4h, "XUSDT") == []


def test_tolerance_is_configurable():
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    df15 = _tail(_bear(), m * 1.03)
    assert detect(df15, df4h, "XUSDT") == []
    assert len(detect(df15, df4h, "XUSDT", Params(near_pct=0.04))) == 1


def test_near_both_requires_both_bands():
    """near_both면 두 밴드 **모두** ±2% 안이어야 한다."""
    p = Params(near_both=True)
    far4h = _f4h()                                  # 두 밴드가 12% 떨어짐
    m, _ = _bands(_bear(), far4h)
    assert detect(_tail(_bear(), m), far4h, "XUSDT", p) == []

    close4h = _f4h(old_price=125.0, sep_price=130.0)  # 두 밴드가 붙어 있음
    m, q = _bands(_bear(), close4h)
    assert m > q and m / q - 1 < 0.03
    got = detect(_tail(_bear(), (m + q) / 2), close4h, "XUSDT", p)
    assert len(got) == 1


def test_month_band_below_quarter_band_does_not_fire():
    """②가 안 서면(월 상단 < 분기 상단) 밴드 근처여도 안 알린다."""
    df4h = _f4h(old_price=130.0, sep_price=90.0)
    m, q = _bands(_bear(), df4h)
    p = Params(min_ret_24h=None)            # ④는 빼고 ②만 본다
    assert detect(_tail(_bear(), m), df4h, "XUSDT", p) == []
    assert detect(_tail(_bear(), q), df4h, "XUSDT", p) == []


def test_mixed_alignment_does_not_fire():
    """역배열도 정배열도 아니면(혼조) 밴드 근처여도 안 알린다."""
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    arr = np.concatenate([np.linspace(50.0, 200.0, 1040),
                          np.linspace(200.0, m, 60)])
    df15 = _f15(arr)
    s120, s240, s480, s960 = _stack(df15, (120, 240, 480, 960))
    assert not (s120 > s240 > s480)
    assert not (s240 < s480 < s960)
    p = Params(min_ret_24h=None)            # 끝이 내려오는 모양이라 ④는 뺀다
    assert detect(df15, df4h, "XUSDT", p) == []
    assert tracking(df15, df4h, "XUSDT", p) is None


def test_only_the_entry_bar_fires():
    """상태 조건이라 매 스캔 다시 알리면 같은 말을 반복한다 — 진입 봉만."""
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    df15 = _tail(_bear(), m, tail=3)
    got = detect(df15, df4h, "XUSDT")
    assert len(got) == 1 and got[0].bar_time == df15.index[-3]


def test_reentry_within_rearm_window_is_tracked_not_realerted():
    """±2% 경계에서 깜빡이는 종목 — 24h 안에 다시 들어오면 새로 안 알린다."""
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    df15 = _bear()
    col = df15.columns.get_loc("Close")
    df15.iloc[-30:-20, col] = m          # 들어왔다가
    df15.iloc[-3:, col] = m              # 나갔다(원래 값 ≈200) 다시 들어옴
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
    m, _ = _bands(_bear(), df4h)
    assert tracking(_tail(_bear(), m, tail=2), df4h, "XUSDT") is None

    held = tracking(_tail(_bear(), m, tail=20), df4h, "XUSDT")
    assert held is not None
    assert held.detail["in_bars"] == 20
    assert held.detail["in_capped"] is False
    assert held.detail["last_price"] == m


def test_bearish_dwell_is_capped_at_ma960_warmup():
    """1,000봉이면 MA960은 마지막 41봉만 선다 — 그 앞은 역배열을 못 잰다."""
    df4h = _f4h()
    m, _ = _bands(_bear(1000), df4h)
    held = tracking(_tail(_bear(1000), m, tail=60), df4h, "XUSDT")
    assert held is not None
    assert held.detail["trend"] == "역배열"
    assert held.detail["in_bars"] == 41
    assert held.detail["in_capped"] is True
    assert notify._dwell_tag(held.detail) == "10h+째"


def test_falling_into_the_band_does_not_fire():
    """④ 24h 동안 밀려 내려와 밴드에 걸린 종목은 '초입'이 아니다."""
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    df15 = _tail(_bear(), m)
    df15.iloc[-97, df15.columns.get_loc("Close")] = m * 1.08   # 24h 전엔 더 위
    assert detect(df15, df4h, "XUSDT") == []
    got = detect(df15, df4h, "XUSDT", Params(min_ret_24h=-0.10))
    assert len(got) == 1 and got[0].detail["ret_24h"] < 0
    assert len(detect(df15, df4h, "XUSDT", Params(min_ret_24h=None))) == 1


def test_rising_into_the_band_carries_24h():
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    got = detect(_tail(_bear(), m), df4h, "XUSDT")
    assert len(got) == 1 and got[0].detail["ret_24h"] > 0


def test_scanner_drops_lines_whose_ticker_24h_is_negative():
    """판정은 15m 96봉 기준이라도, 줄에 찍힐 티커 24h가 마이너스면 뺀다."""
    from invest_signal import scanner
    df4h = _f4h()
    m, _ = _bands(_bear(), df4h)
    df15 = _tail(_bear(), m)
    cfg = {"signal": {"vwap_onset": {"enabled": True}}}
    up = {"XUSDT": {"quote_volume": 5e6, "change_pct": 0.03}}
    down = {"XUSDT": {"quote_volume": 5e6, "change_pct": -0.004}}
    logs = []
    ev, _ = scanner._scan_vwap_onset(cfg, {"XUSDT": df15}, {"XUSDT": df4h}, up,
                                     logs.append)
    assert len(ev) == 1 and ev[0].detail["gain_24h"] == 0.03
    ev, _ = scanner._scan_vwap_onset(cfg, {"XUSDT": df15}, {"XUSDT": df4h}, down,
                                     logs.append)
    assert ev == []
    assert "24h 하락 1건 제외" in logs[-1]


def test_short_history_is_skipped():
    """정배열(MA480)조차 안 서면 판정 자체가 불가능하다."""
    df4h = _f4h()
    m, _ = _bands(_bull(400), df4h)
    assert detect(_tail(_bull(400), m), df4h, "XUSDT") == []


def test_missing_4h_frame_is_skipped():
    assert detect(_tail(_bear(), 130.0), None, "XUSDT") == []


def test_band_is_carried_onto_15m_bars():
    """4h 밴드를 15m 인덱스에 계단식으로 얹는다 — 4시간에 한 번 갱신된다."""
    df15, df4h = _bear(), _f4h()
    up = vo._band_on_15m(df15, df4h, "M", 1.0)
    assert up is not None and not up.isna().all()
    assert up.index.equals(df15.index)


def test_line_shows_alignment_and_nearest_band():
    d = {"trend": "정배열", "near_band": "M", "band_dist": -0.008}
    assert notify._band_tags(d) == ["↑정배열", "월상단 -0.8%"]
    d = {"trend": "역배열", "near_band": "Q", "band_dist": 0.011}
    assert notify._band_tags(d) == ["↓역배열", "분기상단 +1.1%"]
    assert notify._band_tags({}) == []


def test_dwell_tag():
    assert notify._dwell_tag({"in_bars": 16}) == "4h째"
    assert notify._dwell_tag({"in_bars": 144}) == "1.5일째"
    assert notify._dwell_tag({"in_bars": 41, "in_capped": True}) == "10h+째"
