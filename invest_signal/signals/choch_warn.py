"""하락 CHoCH 경고 — ⚡ 모멘텀 감시 종목의 5m 구조가 하락으로 꺾이는 순간 (크립토 전용).

익절·청산용 **경고**다. 매수 신호가 아니다. 예전 🔻하락전환(4h, 60선 아래 눌림의
직전 저점 이탈)을 없애고 이 칸이 그 자리를 쓴다. **하방(상승 구조가 깨지는 쪽)만** 본다.

  대상  ⚡ 크립토 모멘텀 감시 종목(24h 상승률 상위 5위 + 밀려난 뒤 10일) 중
        **4h 종가가 4h MA960(≈160일선) 위** — '3파' 전제와 같은 큰 흐름 위 코인
  사건  **5m 하락 CHoCH** — 좌우 pivot_bars봉(15봉 = 75분) 피벗으로 잡은 스윙이
        고점↑·저점↑(상승 구조)였는데, **마감봉 종가가 마지막 스윙 저점 아래**로 처음
        떨어졌다. 그 뒤 새 스윙 저점이 서서 다시 상승 구조가 될 때까지 다시 세지 않는다.
  빈도  같은 종목은 rearm_hours(24h)에 한 번만.

처음엔 15m(피벗 5봉 = 75분)였다. NMR 09-30을 되짚어 보니 15m는 19:00봉에서야 잡았고
5m(피벗 15봉 = 같은 75분 스윙)는 13:10봉에서 먼저 잡아 5m로 바꿨다(09-30 요청).
스윙 크기(75분)는 그대로 두고 봉만 잘게 봐서, 같은 구조 붕괴를 더 일찍 본다.

**마감된 봉만** 본다 — 진행 중인 봉의 저점 이탈은 마감 때 되돌려질 수 있다.
스캔이 매시 :02라 지난 한 시간 남짓의 봉(grace_bars+1개 = 75분)을 소급해 본다 —
크론이 몇 분 밀려도 빈틈이 안 생기게 15분 여유를 둔다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import SignalEvent

NAME = "choch_warn"
LABEL = "하락 CHoCH"
CRYPTO_ONLY = True
INTERVAL = "5m"
LONG_INTERVAL = "4h"
LONG_LIMIT = 1000          # 4h MA960 + 여유 — 현물 미러 요청당 상한(선물은 1500)


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    interval: str = INTERVAL    # 구조를 보는 봉
    limit: int = 1000           # 받아 올 봉 수 (5m × 1000 ≈ 3.5일)
    pivot_bars: int = 15        # 스윙 피벗 좌우 봉 수 (5m × 15 = 1시간 15분)
    ma_long: int = 960          # 4h MA — 종가가 이 선 위인 종목만
    grace_bars: int = 14        # 마감 시각이 (grace_bars+1)봉 = 75분 안인 봉까지 (스캔 1시간+여유)
    rearm_hours: int = 24       # 같은 종목은 이 시간 안에 다시 안 알린다


def bar_minutes(interval: str) -> int:
    """'5m'·'15m'·'1h' → 분."""
    n, unit = int(interval[:-1]), interval[-1]
    return n * {"m": 1, "h": 60}[unit]


def choch_bars(df: pd.DataFrame, pivot: int) -> list[tuple[int, float, float]]:
    """하락 CHoCH가 난 봉 — (봉 위치, 깬 스윙 저점, 직전 스윙 고점) 목록. 봉 단위는 무관.

    봉 t에서 확정되는 피벗은 t−pivot 봉이다(오른쪽 pivot봉을 봐야 확정). 마지막 두
    스윙 고점·저점이 둘 다 올라 있으면(상승 구조) '무장'하고, 무장 상태에서 종가가
    마지막 스윙 저점 아래로 마감하면 CHoCH — 무장을 풀고, 그 뒤 새 스윙 저점이 서서
    다시 고점↑·저점↑이 될 때까지 기다린다. 피벗은 왼쪽 엄격(>)·오른쪽 (>=)이라 같은
    값이 연달아 나와도 하나만 잡힌다.
    """
    h = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    highs: list[float] = []
    lows: list[float] = []
    armed = False
    broken = -1                 # 이미 깬 스윙 저점의 위치 — 그 뒤 새 저점이 서야 다시 무장
    out = []
    for t in range(len(df)):
        p = t - pivot
        new = False
        if p >= pivot:
            left, right = slice(p - pivot, p), slice(p + 1, p + pivot + 1)
            # 왼쪽은 엄격(>)·오른쪽은 (>=) — 같은 값이 연달아 나와도 피벗은 하나만
            if h[p] > h[left].max() and h[p] >= h[right].max():
                highs.append(h[p])
                new = True
            if lo[p] < lo[left].min() and lo[p] <= lo[right].min():
                lows.append(lo[p])
                new = True
        if (new and len(highs) >= 2 and len(lows) >= 2 and len(lows) - 1 > broken
                and highs[-1] > highs[-2] and lows[-1] > lows[-2]):
            armed = True
        if armed and c[t] < lows[-1]:
            out.append((t, float(lows[-1]), float(highs[-1])))
            armed = False
            broken = len(lows) - 1
    return out


def above_long_ma(df4h: pd.DataFrame | None, when: pd.Timestamp,
                  params: Params = Params()) -> float | None:
    """`when`까지 마감된 4h봉 기준 종가 ÷ MA960 − 1. 위가 아니거나 못 재면 None."""
    if df4h is None or len(df4h) < params.ma_long:
        return None
    closed = df4h[df4h.index + pd.Timedelta(hours=4) <= when]
    if len(closed) < params.ma_long:
        return None
    c = closed["Close"]
    ma = float(c.iloc[-params.ma_long:].mean())
    gap = float(c.iloc[-1]) / ma - 1
    return gap if gap > 0 else None


def detect(df: pd.DataFrame, symbol: str, now: pd.Timestamp,
           params: Params = Params()) -> list[SignalEvent]:
    """마감 시각이 now 전 (grace_bars+1)봉 안인 봉의 하락 CHoCH. 4h 조건은 호출 측.

    df는 params.interval 봉. 진행 중인 봉이 끼어 있어도 된다 — 여기서 뗀다.
    """
    if not params.enabled or df is None:
        return []
    d = df[df.index + pd.Timedelta(minutes=bar_minutes(params.interval)) <= now]
    if len(d) < 4 * params.pivot_bars + 2:
        return []
    step = pd.Timedelta(minutes=bar_minutes(params.interval))
    cutoff = now - step * (params.grace_bars + 1)       # 마감 시각이 이보다 뒤인 봉만
    out = []
    for t, low, high in choch_bars(d, params.pivot_bars):
        if d.index[t] + step <= cutoff:
            continue
        close = float(d["Close"].iloc[t])
        out.append(SignalEvent(
            symbol=symbol, signal=NAME, bar_time=d.index[t], price=close,
            detail={"label": LABEL, "interval": params.interval, "broken_low": low,
                    "swing_high": high,
                    "from_high": close / high - 1 if high and not np.isnan(high) else None}))
    return out[-1:]            # 한 번에 여러 봉이면 가장 최근 것만
