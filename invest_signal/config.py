"""config.yaml 로드와 시그널 파라미터 매핑."""

import yaml

from .signals import pullback, uptrend_onset

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
        breakout_window_bars=int(s.get("breakout_window_bars", 180)),
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
    return out
