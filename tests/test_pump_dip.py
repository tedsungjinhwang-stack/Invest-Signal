"""펌핑 눌림(크립토 전용) 시그널 테스트 — 합성 데이터."""

import numpy as np
import pandas as pd

from invest_signal.signals import pump_dip
from invest_signal.signals.pump_dip import Params

P = Params()


def make_df(closes, highs=None, lows=None, volumes=None, start="2025-01-06"):
    """start 기본값은 월요일 — 주간 VWAP 앵커가 인덱스 0에서 시작."""
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    h = pd.Series([float(x) for x in (highs or closes)], index=idx)
    lo = pd.Series([float(x) for x in (lows or closes)], index=idx)
    data = {"Open": c, "High": np.maximum(c, h), "Low": np.minimum(c, lo),
            "Close": c}
    if volumes is not None:
        data["Volume"] = [float(v) for v in volumes]
    return pd.DataFrame(data)


def pump_fixture():
    """베이스 100 횡보 → 5봉(하루 이내) 만에 300까지 수직 펌핑."""
    closes = [100.0] * 55 + list(np.linspace(150.0, 300.0, 5))
    vols = [1.0] * len(closes)
    return closes, vols


def test_fires_on_first_wvwap_touch_after_pump():
    closes, vols = pump_fixture()
    closes.append(290.0)                     # 눌림 봉 — 종가는 고점 부근
    lows = closes.copy()
    lows[-1] = 100.0                         # 꼬리가 주간 VWAP까지 깊게 터치
    vols.append(1.0)
    df = make_df(closes, lows=lows, volumes=vols)
    evs = pump_dip.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "pump_dip"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["pump_gain"] == 2.0     # 고점 직전 하루 저점 100 → 고점 300
    assert ev.detail["wvwap"] >= 100.0

    # 다음 봉이 또 닿아도 첫 터치 봉에서만 (grace로 두 봉 다 후보일 때)
    closes.append(280.0)
    lows.append(100.0)
    vols.append(1.0)
    df2 = make_df(closes, lows=lows, volumes=vols)
    evs2 = pump_dip.detect(df2, "TEST", Params(grace_bars=1))
    assert len(evs2) == 1 and evs2[0].bar_time == df2.index[-2]


def test_slow_rise_is_not_a_pump():
    """일주일에 걸쳐 완만히 +200% 오른 건 펌핑이 아니다 (하루 내 +30% 미달)."""
    closes = [100.0] * 20 + list(np.linspace(100.0, 300.0, 42))
    vols = [1.0] * len(closes)
    closes.append(290.0)
    lows = closes.copy()
    lows[-1] = 100.0                         # 터치는 함
    vols.append(1.0)
    df = make_df(closes, lows=lows, volumes=vols)
    assert pump_dip.detect(df, "TEST", P) == []


def test_no_fire_without_pump():
    closes = [100.0] * 60
    vols = [1.0] * 60
    closes.append(95.0)                      # 터치는 하지만 펌핑이 없었음
    lows = closes.copy()
    lows[-1] = 50.0
    vols.append(1.0)
    df = make_df(closes, lows=lows, volumes=vols)
    assert pump_dip.detect(df, "TEST", P) == []


def test_no_fire_without_volume():
    closes, _ = pump_fixture()
    closes.append(290.0)
    lows = closes.copy()
    lows[-1] = 100.0
    df = make_df(closes, lows=lows)          # Volume 없음 → WVWAP 계산 불가
    assert pump_dip.detect(df, "TEST", P) == []


def test_still_active_while_close_holds_above_wvwap():
    closes, vols = pump_fixture()
    closes.append(290.0)
    lows = closes.copy()
    lows[-1] = 100.0
    vols.append(1.0)
    df = make_df(closes, lows=lows, volumes=vols)
    ev = pump_dip.detect(df, "TEST", P)[0]
    assert pump_dip.still_active(df, ev, P)          # 종가 290 — WVWAP 위
    closes.append(50.0)                              # 주간 VWAP 아래로 붕괴
    lows.append(50.0)
    vols.append(1.0)
    assert not pump_dip.still_active(make_df(closes, lows=lows, volumes=vols), ev, P)


def test_crypto_only_flag_and_registration():
    """ETF·주식 스캔 제외 플래그 + detectors() 기본 등록 확인."""
    from invest_signal import config as cfg_mod

    assert pump_dip.CRYPTO_ONLY is True
    cfg = {"signal": {}, "crypto": {}, "etf": {}}
    mods = [m.NAME for m, _ in cfg_mod.detectors(cfg)]
    assert "pump_dip" in mods
    yf_mods = [m.NAME for m, p in cfg_mod.detectors(cfg)
               if not getattr(m, "CRYPTO_ONLY", False)]
    assert "pump_dip" not in yf_mods
