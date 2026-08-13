"""스캐너 헬퍼 테스트."""

import pandas as pd

from invest_signal.scanner import _collapse
from invest_signal.signals import SignalEvent


def _ev(symbol, signal, hour):
    return SignalEvent(symbol=symbol, signal=signal,
                       bar_time=pd.Timestamp(f"2026-07-22 {hour:02d}:00", tz="UTC"),
                       price=1.0, detail={})


def test_collapse_keeps_latest_per_symbol_signal():
    evs = [_ev("ADXUSDT", "uptrend_onset", 16), _ev("ADXUSDT", "uptrend_onset", 20),
           _ev("ADXUSDT", "pullback", 20), _ev("BTCUSDT", "uptrend_onset", 20)]
    out = _collapse(evs)
    assert len(out) == 3
    ut = [e for e in out if e.symbol == "ADXUSDT" and e.signal == "uptrend_onset"]
    assert len(ut) == 1 and ut[0].bar_time.hour == 20


def test_detect_all_choch_evicts_prior_long_holds():
    """하락전환(CHoCH) 이후의 셋업만 유지 중 리스트에 살아남는다."""
    from types import SimpleNamespace
    from invest_signal.scanner import _detect_all
    from invest_signal.signals.pullback import Params

    idx = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                      index=idx)

    def ev2(sig, label, i):
        return SignalEvent(symbol="X", signal=sig, bar_time=idx[i],
                           price=1.0, detail={"label": label})

    def mod(name, events):
        return SimpleNamespace(NAME=name,
                               detect=lambda d, s, p: list(events),
                               still_active=lambda d, e, p: True)

    pb = ev2("pullback", "눌림목", 3)
    up = ev2("uptrend_onset", "상승초입", 2)
    mss_ev = ev2("mss", "MSS", 4)
    # CHoCH(5봉)가 셋업들 이후 → 눌림목만 제거, 상승초입·MSS는 유지
    dets = [(mod("pullback", [pb]), Params()),
            (mod("uptrend_onset", [up]), Params()),
            (mod("mss", [mss_ev]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 5)]), Params())]
    _, ongoing = _detect_all({"X": df}, dets)
    assert [e.signal for e in ongoing] == ["uptrend_onset", "mss"]
    assert all(e.detail["last_price"] == 1.0 for e in ongoing)   # 현재가 주석
    # CHoCH(1봉)가 눌림목(3봉)보다 과거 → 눌림목 유지
    dets = [(mod("pullback", [pb]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 1)]), Params())]
    _, ongoing = _detect_all({"X": df}, dets)
    assert [e.signal for e in ongoing] == ["pullback"]


def test_detect_all_show_choch_displays_downtrend_events():
    """show_choch=True(ETF·주식)면 하락전환이 알림·추적에 잡히고,
    기본값(크립토)이면 제거 용도로만 쓰이고 표시되지 않는다."""
    from types import SimpleNamespace
    from invest_signal.scanner import _detect_all
    from invest_signal.signals.pullback import Params

    idx = pd.date_range("2025-01-01", periods=10, freq="4h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                      index=idx)

    def ev2(sig, label, i):
        return SignalEvent(symbol="X", signal=sig, bar_time=idx[i],
                           price=1.0, detail={"label": label})

    def mod(name, events):
        return SimpleNamespace(NAME=name,
                               detect=lambda d, s, p: list(events),
                               still_active=lambda d, e, p: True)

    dets = [(mod("pullback", [ev2("pullback", "눌림목", 3)]), Params()),
            (mod("downtrend_reversal", [ev2("downtrend_reversal", "하락전환", 5)]), Params())]

    events, ongoing = _detect_all({"X": df}, dets, show_choch=True)
    assert "downtrend_reversal" in {e.signal for e in events}
    assert [e.signal for e in ongoing] == ["downtrend_reversal"]   # 풀백은 CHoCH로 제거

    events, ongoing = _detect_all({"X": df}, dets)                 # 크립토 기본 동작
    assert "downtrend_reversal" not in {e.signal for e in events}
    assert ongoing == []


def test_filter_ranked_scopes():
    """풀백 스코프는 풀백+MSS만 거르고, 상승초입 스코프는 상승초입만 —
    서로 다른 eligible 집합(완화 기준)을 독립 적용할 수 있다."""
    from invest_signal.scanner import RANK_FILTER_SCOPE, _filter_ranked

    evs = [_ev("BIGVOL", "pullback", 0), _ev("THIN", "pullback", 0),
           _ev("BIGVOL", "mss", 4), _ev("THIN", "mss", 4),
           _ev("BIGVOL", "uptrend_onset", 8), _ev("THIN", "uptrend_onset", 8)]
    out = _filter_ranked(evs, {"BIGVOL"}, RANK_FILTER_SCOPE["pullback"])
    assert {(e.symbol, e.signal) for e in out} == {
        ("BIGVOL", "pullback"), ("BIGVOL", "mss"),
        ("BIGVOL", "uptrend_onset"), ("THIN", "uptrend_onset")}
    # 상승초입은 자체(더 넓은) eligible 집합으로 거른다 — 풀백·MSS는 건드리지 않음
    out = _filter_ranked(out, {"BIGVOL", "THIN"}, RANK_FILTER_SCOPE["uptrend_onset"])
    assert ("THIN", "uptrend_onset") in {(e.symbol, e.signal) for e in out}
    out = _filter_ranked(out, {"BIGVOL"}, RANK_FILTER_SCOPE["uptrend_onset"])
    assert {(e.symbol, e.signal) for e in out} == {
        ("BIGVOL", "pullback"), ("BIGVOL", "mss"), ("BIGVOL", "uptrend_onset")}


def test_crypto_rank_eligible_union_of_volume_and_gain():
    from invest_signal.scanner import _crypto_rank_eligible

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")

    def frame(price, vol, rising=False):
        import numpy as np
        c = pd.Series(np.linspace(price * (0.5 if rising else 1.0), price, 50), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                             "Volume": float(vol)})

    frames = {
        "BIGVOL": frame(1.0, 1_000_000),          # 거래대금 1위, 상승률 없음
        "PUMP": frame(2.0, 10, rising=True),      # 상승률 1위, 거래대금 미미
        "SLEEPY": frame(1.0, 1),                  # 둘 다 하위
    }
    eligible = _crypto_rank_eligible(frames, {"volume_top": 1, "gain_top": 1,
                                              "gain_lookback_bars": 42})
    assert "BIGVOL" in eligible          # 거래대금 통과
    assert "PUMP" in eligible            # 상승률 통과
    assert "SLEEPY" not in eligible      # 어느 쪽도 아님


def test_crypto_rank_eligible_and_mode_requires_both():
    """mode=and면 거래대금·상승률 둘 다 상위여야 통과 — 주도주만 남긴다."""
    from invest_signal.scanner import _crypto_rank_eligible
    import numpy as np

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")

    def frame(price, vol, rising=False):
        c = pd.Series(np.linspace(price * (0.5 if rising else 1.0), price, 50), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                             "Volume": float(vol)})

    frames = {
        "LEADER": frame(2.0, 10_000_000, rising=True),   # 거래대금·상승률 모두 상위
        "BIGVOL": frame(1.0, 9_000_000),                 # 거래대금만 상위 (횡보)
        "THINPUMP": frame(2.0, 10, rising=True),         # 상승률만 상위 (유동성 없음)
    }
    rcfg = {"mode": "and", "volume_top": 2, "gain_top": 2, "gain_lookback_bars": 42}
    assert _crypto_rank_eligible(frames, rcfg) == {"LEADER"}
    # 같은 설정이라도 or면 셋 다 통과 — mode가 실제로 교집합을 만든다
    assert _crypto_rank_eligible(frames, {**rcfg, "mode": "or"}) == {
        "LEADER", "BIGVOL", "THINPUMP"}


def test_crypto_rank_eligible_gain_top_pct_scales_with_universe():
    """gain_top_pct는 유니버스 크기에 비례해 상위권 크기를 정한다."""
    from invest_signal.scanner import _crypto_rank_eligible
    import numpy as np

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")
    # 20종 — 상승률이 큰 순서대로 COIN00 … COIN19
    frames = {}
    for i in range(20):
        c = pd.Series(np.linspace(1.0 - i * 0.02, 1.0, 50), index=idx)
        frames[f"COIN{i:02d}"] = pd.DataFrame(
            {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1_000_000.0})
    rcfg = {"mode": "and", "volume_top": 20, "gain_top_pct": 0.10,
            "gain_lookback_bars": 42}
    eligible = _crypto_rank_eligible(frames, rcfg)
    assert len(eligible) == 2                      # 20종의 10% = 상위 2종
    assert eligible == {"COIN19", "COIN18"}
    # gain_top(절대 수)보다 gain_top_pct가 우선한다
    assert len(_crypto_rank_eligible(frames, {**rcfg, "gain_top": 15})) == 2


def test_crypto_rank_filter_min_turnover_floor():
    """상승률 상위라도 거래대금 하한($) 미달이면 제외."""
    from invest_signal.scanner import _crypto_rank_eligible
    import numpy as np

    idx = pd.date_range("2025-01-01", periods=50, freq="4h", tz="UTC")

    def frame(price, vol, rising=False):
        c = pd.Series(np.linspace(price * (0.5 if rising else 1.0), price, 50), index=idx)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                             "Volume": float(vol)})

    frames = {"BIGVOL": frame(1.0, 10_000_000),        # 24h 거래대금 ≈ 6천만 달러
              "THINPUMP": frame(2.0, 10, rising=True)}  # 상승률 1위지만 유동성 없음
    rcfg = {"volume_top": 1, "gain_top": 1, "gain_lookback_bars": 42,
            "min_turnover_usd": 5_000_000}
    eligible = _crypto_rank_eligible(frames, rcfg)
    assert "BIGVOL" in eligible
    assert "THINPUMP" not in eligible


def test_crypto_enabled_false_drops_signal_from_crypto_only(monkeypatch):
    """crypto_enabled: false인 시그널은 크립토 스캔에서만 빠진다."""
    from invest_signal import scanner
    from invest_signal.signals import downtrend_reversal, pullback

    seen = {}

    def fake_detect_all(frames, dets, log=print, show_choch=False):
        seen["names"] = [m.NAME for m, _ in dets]
        return [], []

    monkeypatch.setattr(scanner, "_detect_all", fake_detect_all)
    monkeypatch.setattr(scanner, "_scan_leader_break",
                        lambda *a, **k: ([], [], []))
    monkeypatch.setattr(scanner.data_binance, "resolve_source",
                        lambda *a, **k: ("fapi", ["XUSDT"]))
    monkeypatch.setattr(scanner.data_binance, "fetch_all",
                        lambda *a, **k: {"XUSDT": pd.DataFrame()})

    dets = [(pullback, pullback.Params()),
            (downtrend_reversal, downtrend_reversal.Params())]
    cfg = {"crypto": {"enabled": True},
           "signal": {"pullback": {"crypto_enabled": False}}}
    scanner.scan_crypto(cfg, dets, log=lambda *a: None)
    assert seen["names"] == ["downtrend_reversal"]   # 눌림목은 크립토에서 제외
    # 같은 설정이어도 ETF·주식 경로는 눌림목을 그대로 돌린다
    assert cfg["signal"]["pullback"]["crypto_enabled"] is False


def test_crypto_only_signals_skipped_on_yfinance_path():
    """CRYPTO_ONLY 모듈은 ETF·주식 목록에서 걸러진다."""
    from invest_signal.signals import leader_break, pullback

    dets = [(pullback, pullback.Params()), (leader_break, leader_break.Params())]
    assert leader_break.CRYPTO_ONLY is True
    yf = [m.NAME for m, _ in dets if not getattr(m, "CRYPTO_ONLY", False)]
    assert yf == ["pullback"]


def test_crypto_override_applies_only_to_crypto_detectors():
    """crypto_<키> 오버라이드는 크립토 목록에만 반영되고 ETF·주식은 공용 값."""
    from invest_signal import config as cfg_mod

    cfg = {"signal": {"uptrend_onset": {"vwap_condition": False,
                                        "crypto_vwap_condition": True}}}
    crypto = dict((m.NAME, p) for m, p in cfg_mod.detectors(cfg, crypto=True))
    yf = dict((m.NAME, p) for m, p in cfg_mod.detectors(cfg))
    assert crypto["uptrend_onset"].vwap_condition is True
    assert yf["uptrend_onset"].vwap_condition is False
    # 오버라이드가 없으면 양쪽 다 공용 값을 쓴다
    cfg2 = {"signal": {"uptrend_onset": {"vwap_condition": False}}}
    for c in (True, False):
        d = dict((m.NAME, p) for m, p in cfg_mod.detectors(cfg2, crypto=c))
        assert d["uptrend_onset"].vwap_condition is False


def test_uptrend_vwap_period_defaults_to_monthly():
    """상승초입 VWAP은 월간 앵커가 기본이고, 주기는 설정에서 바꾼다."""
    from invest_signal import config as cfg_mod

    assert cfg_mod.uptrend_params({}).vwap_period == "M"
    p = cfg_mod.uptrend_params({"signal": {"uptrend_onset": {"vwap_period": "Q"}}})
    assert p.vwap_period == "Q"


def test_legacy_qvwap_keys_still_honored():
    """분기 전용이던 시절의 qvwap_condition 키가 남아 있어도 그 값을 따른다."""
    from invest_signal import config as cfg_mod

    cfg = {"signal": {"uptrend_onset": {"qvwap_condition": False,
                                        "crypto_qvwap_condition": True}}}
    assert cfg_mod.uptrend_params(cfg).vwap_condition is False
    assert cfg_mod.uptrend_params(cfg, crypto=True).vwap_condition is True
    # 새 키가 함께 있으면 새 키가 이긴다
    cfg["signal"]["uptrend_onset"]["vwap_condition"] = True
    assert cfg_mod.uptrend_params(cfg).vwap_condition is True


def test_uptrend_ma_align_read_from_config():
    """ma_align·ma_entry는 설정에서 읽고, 없으면 기본 (120,240)/20."""
    from invest_signal import config as cfg_mod

    p = cfg_mod.uptrend_params(
        {"signal": {"uptrend_onset": {"ma_align": [120, 240, 480], "ma_entry": 60}}})
    assert p.ma_align == (120, 240, 480) and p.ma_entry == 60
    d = cfg_mod.uptrend_params({})
    assert d.ma_align == (120, 240) and d.ma_entry == 20


def test_market_value_prefers_crypto_key():
    from invest_signal.config import market_value

    s = {"x": 1, "crypto_x": 2}
    assert market_value(s, "x", 0, crypto=True) == 2
    assert market_value(s, "x", 0, crypto=False) == 1
    assert market_value({}, "x", 9, crypto=True) == 9      # 둘 다 없으면 기본값
    assert market_value({"x": 1}, "x", 0, crypto=True) == 1  # 오버라이드 없으면 공용
