"""config.yaml 로드."""

import yaml

from .signals.uptrend_onset import Params as UptrendParams

DEFAULT_PATH = "config.yaml"


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("crypto", {})
    cfg.setdefault("etf", {})
    cfg.setdefault("signal", {})
    return cfg


def uptrend_params(cfg: dict) -> UptrendParams:
    s = (cfg.get("signal") or {}).get("uptrend_onset") or {}
    return UptrendParams(
        touch_window_bars=int(s.get("touch_window_bars", 60)),
        grace_bars=int(s.get("grace_bars", 1)),
        require_above_qvwap=bool(s.get("require_above_qvwap", False)),
    )
