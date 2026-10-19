"""📅 일봉 눌림 (10-19~) — ⚡ 모멘텀·🚀 급등봉에 들었던 종목을 50일 동안 쌓아 두고 일봉 눌림을 본다.

대상(풀)  ⚡ 감시 종목(24h 상승률 상위에 들었던 종목)과 🚀 급등봉이 터진 종목 — 마지막으로 들어온 뒤
          pool_days(50)일 동안. 상태 파일의 pool에 적는다.
사건      오늘 일봉(진행 중 포함 — 매시 스캔이 닿는 순간 잡는다)에서
  ① 📉볼밴하단  어제는 저가가 일봉 볼린저 하단선(20, 2σ) 위였는데 오늘 저가가 하단선까지 내려옴
  ② 🔁일봉단기선터치 · 💔일봉단기선이탈  일봉 단기 수퍼트렌드(22×3)가 **상승**이고 어제는 저가가 선
     위였는데 오늘 저가가 선까지 내려옴 — 지금 가격이 선 위면 터치, 아래면 이탈
     선은 **어제(마감된 봉)의 선**으로 본다 — 오늘 봉이 선을 뚫으면 수트가 뒤집혀 선이 위로 튀므로.
같은 종목·같은 사건은 하루 한 번(dedup 키 = 오늘 일봉 시각 + 사건).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import supertrend_full
from . import SignalEvent

NAME = "daily_dip"
LABEL = "일봉 눌림"
INTERVAL = "1d"
BB_LOWER = "볼밴하단"
ST_TOUCH = "일봉단기선터치"
ST_BREAK = "일봉단기선이탈"


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    pool_days: int = 50
    bb_period: int = 20
    bb_mult: float = 2.0
    st_period: int = 22
    st_mult: float = 3.0
    kline_limit: int = 120


def detect(df1d: pd.DataFrame | None, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """오늘 일봉(마지막 봉, 진행 중 포함)에서 ①·② — 사건마다 SignalEvent 하나."""
    if df1d is None or len(df1d) < max(params.bb_period, params.st_period) + 3:
        return []
    t = len(df1d) - 1
    lo = df1d["Low"].to_numpy(float)
    c = df1d["Close"].to_numpy(float)
    out = []
    mid = df1d["Close"].rolling(params.bb_period).mean()
    sd = df1d["Close"].rolling(params.bb_period).std(ddof=0)
    lower = (mid - params.bb_mult * sd).to_numpy(float)
    if not np.isnan(lower[t]) and not np.isnan(lower[t - 1]) and lo[t - 1] > lower[t - 1] \
            and lo[t] <= lower[t]:
        out.append(_ev(df1d, symbol, BB_LOWER, float(lower[t])))
    st = supertrend_full(df1d.iloc[:-1], params.st_period, params.st_mult)   # 어제까지(마감 봉)
    d, line = st["dir"].to_numpy(float), st["line"].to_numpy(float)
    if len(d) >= 2 and d[-1] > 0 and not np.isnan(line[-1]) and lo[t - 1] > line[-1] \
            and lo[t] <= line[-1]:
        out.append(_ev(df1d, symbol, ST_TOUCH if c[t] >= line[-1] else ST_BREAK, float(line[-1])))
    return out


def _ev(df1d, symbol, stage, level):
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=df1d.index[-1],
                       price=float(df1d["Close"].iloc[-1]),
                       detail={"label": LABEL, "stage": stage, "interval": INTERVAL,
                               "level": level, "low": float(df1d["Low"].iloc[-1])})
