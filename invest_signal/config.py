"""config.yaml 로드와 시그널 파라미터 매핑."""

import yaml

from .signals import (downtrend_reversal, mss, pullback, pump_dip, pump_early,
                      uptrend_onset)

DEFAULT_PATH = "config.yaml"


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("crypto", {})
    cfg.setdefault("etf", {})
    cfg.setdefault("signal", {})
    return cfg


def uptrend_params(cfg: dict) -> uptrend_onset.Params:
    s = (cfg.get("signal") or {}).get("uptrend_onset") or {}
    return uptrend_onset.Params(
        touch_window_bars=int(s.get("touch_window_bars", 60)),
        grace_bars=int(s.get("grace_bars", 1)),
        qvwap_condition=bool(s.get("qvwap_condition", True)),
    )


def pullback_params(cfg: dict) -> pullback.Params:
    s = (cfg.get("signal") or {}).get("pullback") or {}
    return pullback.Params(
        ma_entry=int(s.get("ma_entry", 120)),
        grace_bars=int(s.get("grace_bars", 1)),
        qvwap_condition=bool(s.get("qvwap_condition", True)),
    )


def downtrend_params(cfg: dict) -> downtrend_reversal.Params:
    s = (cfg.get("signal") or {}).get("downtrend_reversal") or {}
    return downtrend_reversal.Params(
        qvwap_condition=bool(s.get("qvwap_condition", True)),
        ma_ref=int(s.get("ma_ref", 60)),
        lookback_bars=int(s.get("lookback_bars", 180)),
        grace_bars=int(s.get("grace_bars", 1)),
    )


def mss_params(cfg: dict) -> mss.Params:
    s = (cfg.get("signal") or {}).get("mss") or {}
    return mss.Params(
        pivot_k=int(s.get("pivot_k", 6)),
        grace_bars=int(s.get("grace_bars", 1)),
        qvwap_condition=bool(s.get("qvwap_condition", True)),
    )


def pump_params(cfg: dict) -> pump_dip.Params:
    s = (cfg.get("signal") or {}).get("pump_dip") or {}
    return pump_dip.Params(
        peak_lookback_bars=int(s.get("peak_lookback_bars", 42)),
        pump_window_bars=int(s.get("pump_window_bars", 6)),
        min_gain=float(s.get("min_gain", 0.3)),
        touch_tolerance=float(s.get("touch_tolerance", 0.003)),
        grace_bars=int(s.get("grace_bars", 1)),
    )


def pump_early_params(cfg: dict) -> pump_early.Params:
    s = (cfg.get("signal") or {}).get("pump_early") or {}
    return pump_early.Params(
        rise_bars=int(s.get("rise_bars", 1)),
        min_gain=float(s.get("min_gain", 0.05)),
        lookback_bars=int(s.get("lookback_bars", 42)),
        max_gain=float(s.get("max_gain", 0.30)),
        grace_bars=int(s.get("grace_bars", 1)),
    )


def detectors(cfg: dict) -> list:
    """활성화된 시그널 모듈과 파라미터 목록."""
    s = cfg.get("signal") or {}
    out = []
    if (s.get("uptrend_onset") or {}).get("enabled", True):
        out.append((uptrend_onset, uptrend_params(cfg)))
    if (s.get("pullback") or {}).get("enabled", True):
        out.append((pullback, pullback_params(cfg)))
    if (s.get("pump_dip") or {}).get("enabled", True):
        out.append((pump_dip, pump_params(cfg)))
    if (s.get("pump_early") or {}).get("enabled", True):
        out.append((pump_early, pump_early_params(cfg)))
    if (s.get("downtrend_reversal") or {}).get("enabled", True):
        out.append((downtrend_reversal, downtrend_params(cfg)))
    if (s.get("mss") or {}).get("enabled", True):
        out.append((mss, mss_params(cfg)))
    return out
