"""config.yaml 로드와 시그널 파라미터 매핑."""

import yaml

from .signals import (downtrend_reversal, mss, pullback, pump_early,
                      uptrend_onset, wave_setup)

DEFAULT_PATH = "config.yaml"


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("crypto", {})
    cfg.setdefault("etf", {})
    cfg.setdefault("signal", {})
    return cfg


def market_value(s: dict, key: str, default, crypto: bool = False):
    """시장별로 다르게 줄 수 있는 설정값.

    크립토 스캔이면 `crypto_<key>`가 있을 때 그 값을 우선 쓰고, 없으면
    공용 값을 쓴다. ETF·주식은 항상 공용 값. 같은 시그널을 시장마다
    다른 기준으로 돌려야 할 때 쓰는 패턴이다
    (예: 상승초입의 QVWAP 조건은 크립토에서만 켠다).
    """
    base = s.get(key, default)
    return s.get(f"crypto_{key}", base) if crypto else base


def uptrend_params(cfg: dict, crypto: bool = False) -> uptrend_onset.Params:
    s = (cfg.get("signal") or {}).get("uptrend_onset") or {}
    return uptrend_onset.Params(
        ma_entry=int(s.get("ma_entry", 20)),
        grace_bars=int(s.get("grace_bars", 1)),
        ma_align=tuple(s.get("ma_align", (120, 240, 480))),
        touch_condition=bool(s.get("touch_condition", False)),
        touch_window_bars=int(s.get("touch_window_bars", 60)),
        vwap_mode=str(s.get("vwap_mode", "any")),
        vwap_touch_bars=int(s.get("vwap_touch_bars", 5)),
        supertrend_condition=bool(s.get("supertrend_condition", True)),
        st_period=int(s.get("st_period", 22)),
        st_mult=float(s.get("st_mult", 3.0)),
        vwap_condition=bool(market_value(_vwap_aliases(s), "vwap_condition",
                                         True, crypto)),
        vwap_period=str(s.get("vwap_period", "M")),
    )


def _vwap_aliases(s: dict) -> dict:
    """분기 전용이던 시절의 `qvwap_condition` 키를 새 이름으로 옮겨 준다.

    설정 파일을 갱신하지 않은 채 배포되면 조건이 조용히 기본값(켬)으로
    돌아가 ETF·주식 알림이 몇 배로 늘어난다 — 옛 키를 그대로 존중한다.
    """
    out = dict(s)
    for old, new in (("qvwap_condition", "vwap_condition"),
                     ("crypto_qvwap_condition", "crypto_vwap_condition")):
        if new not in out and old in out:
            out[new] = out[old]
    return out


def pullback_params(cfg: dict) -> pullback.Params:
    s = (cfg.get("signal") or {}).get("pullback") or {}
    return pullback.Params(
        ma_entry=int(s.get("ma_entry", 60)),
        ma_above=int(s.get("ma_above", 480)),
        band_mult=float(s.get("band_mult", 1.0)),
        band_condition=bool(s.get("band_condition", True)),
        grace_bars=int(s.get("grace_bars", 1)),
        supertrend_exit=bool(s.get("supertrend_exit", True)),
        st_period=int(s.get("st_period", 22)),
        st_mult=float(s.get("st_mult", 3.0)),
    )


def downtrend_params(cfg: dict) -> downtrend_reversal.Params:
    s = (cfg.get("signal") or {}).get("downtrend_reversal") or {}
    return downtrend_reversal.Params(
        qvwap_condition=bool(s.get("qvwap_condition", True)),
        ma_ref=int(s.get("ma_ref", 60)),
        lookback_bars=int(s.get("lookback_bars", 180)),
        grace_bars=int(s.get("grace_bars", 1)),
        supertrend_exit=bool(s.get("supertrend_exit", True)),
        st_period=int(s.get("st_period", 22)),
        st_mult=float(s.get("st_mult", 3.0)),
    )


def pump_early_params(cfg: dict) -> pump_early.Params:
    s = (cfg.get("signal") or {}).get("pump_early") or {}
    return pump_early.Params(
        rise_bars=int(s.get("rise_bars", 1)),
        min_gain=float(s.get("min_gain", 0.05)),
        lookback_bars=int(s.get("lookback_bars", 42)),
        pump_window_bars=int(s.get("pump_window_bars", 6)),
        max_pump_gain=float(s.get("max_pump_gain", 0.30)),
        max_gain=float(s.get("max_gain", 0.30)),
        ma_ref=int(s.get("ma_ref", 480)),
        below_ma_condition=bool(s.get("below_ma_condition", True)),
        grace_bars=int(s.get("grace_bars", 1)),
    )


def wave_params(cfg: dict) -> wave_setup.Params:
    s = (cfg.get("signal") or {}).get("wave_setup") or {}
    return wave_setup.Params(
        fast_period=int(s.get("fast_period", 22)),
        fast_mult=float(s.get("fast_mult", 3.0)),
        slow_period=int(s.get("slow_period", 30)),
        slow_mult=float(s.get("slow_mult", 6.0)),
        abc_enabled=bool(s.get("abc_enabled", True)),
        impulse_enabled=bool(s.get("impulse_enabled", True)),
        grace_bars=int(s.get("grace_bars", 1)),
        daily_grace_bars=int(s.get("daily_grace_bars", 1)),
        flip_window_bars=int(s.get("flip_window_bars", 12)),
        abc_touch_slow=bool(s.get("abc_touch_slow", True)),
        abc_touch_fast=bool(s.get("abc_touch_fast", True)),
        abc_flip=bool(s.get("abc_flip", True)),
        abc_track_days=int(s.get("abc_track_days", 3)),
        slow_break_enabled=bool(s.get("slow_break_enabled", True)),
        slow_break_track_days=int(s.get("slow_break_track_days", 1)),
        retrace_enabled=bool(s.get("retrace_enabled", True)),
        retrace_level=float(s.get("retrace_level", 0.5)),
        retrace_window_days=int(s.get("retrace_window_days", 3)),
        retrace_hold_bars=int(s.get("retrace_hold_bars", 2)),
        retrace_expire_bars=int(s.get("retrace_expire_bars", 42)),
        retrace_track_days=int(s.get("retrace_track_days", 2)),
        vwap_condition=bool(s.get("vwap_condition", True)),
        vwap_period=str(s.get("vwap_period", "M")),
        vwap_mode=str(s.get("vwap_mode", "any")),
        vwap_touch_bars=int(s.get("vwap_touch_bars", 5)),
        quiet_turnover_usd=float(s.get("quiet_turnover_usd", 5_000_000)),
        quiet_atr_min=float(s.get("quiet_atr_min", 0.019)),
        quiet_atr_max=float(s.get("quiet_atr_max", 0.044)),
    )


def mss_params(cfg: dict) -> mss.Params:
    s = (cfg.get("signal") or {}).get("mss") or {}
    return mss.Params(
        pivot_k=int(s.get("pivot_k", 6)),
        grace_bars=int(s.get("grace_bars", 1)),
        gate=pullback_params(cfg),   # 진입 조건은 눌림목 설정을 그대로 공유
    )


def detectors(cfg: dict, crypto: bool = False) -> list:
    """활성화된 시그널 모듈과 파라미터 목록.

    crypto=True면 `crypto_<키>` 오버라이드가 있는 설정을 크립토 값으로
    바꿔 넣는다 — 같은 시그널을 시장별로 다르게 돌리기 위한 것이다.
    """
    s = cfg.get("signal") or {}
    out = []
    if (s.get("uptrend_onset") or {}).get("enabled", True):
        out.append((uptrend_onset, uptrend_params(cfg, crypto)))
    if (s.get("pullback") or {}).get("enabled", True):
        out.append((pullback, pullback_params(cfg)))
    if (s.get("downtrend_reversal") or {}).get("enabled", True):
        out.append((downtrend_reversal, downtrend_params(cfg)))
    if (s.get("pump_early") or {}).get("enabled", True):
        out.append((pump_early, pump_early_params(cfg)))
    if (s.get("wave_setup") or {}).get("enabled", True):
        out.append((wave_setup, wave_params(cfg)))
    if (s.get("mss") or {}).get("enabled", True):
        out.append((mss, mss_params(cfg)))
    return out
