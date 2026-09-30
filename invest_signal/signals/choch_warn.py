"""하락 CHoCH 경고 — ⚡ 모멘텀 감시 종목의 15m 구조가 하락으로 꺾이는 순간 (크립토 전용).

익절·청산용 **경고**다. 매수 신호가 아니다. 예전 🔻하락전환(4h, 60선 아래 눌림의
직전 저점 이탈)을 없애고 이 칸이 그 자리를 쓴다.

  대상  ⚡ 크립토 모멘텀 감시 종목(24h 상승률 상위 5위 + 밀려난 뒤 10일) 중
        **4h 종가가 4h MA960(≈160일선) 위** — '3파' 전제와 같은 큰 흐름 위 코인
  사건  **15m 하락 CHoCH** — 좌우 pivot_bars봉 피벗으로 잡은 15m 스윙이 고점↑·저점↑
        (상승 구조)였는데, 15m **마감봉 종가가 마지막 스윙 저점 아래**로 처음 떨어졌다.
        그 뒤 새 상승 구조가 설 때까지는 같은 저점 이탈을 다시 세지 않는다.
  빈도  같은 종목은 rearm_hours(24h)에 한 번만.

퍼프 실측(08-29~09-28, 30일): 15m 하락 CHoCH는 흔해서 전체 퍼프(4h 960선 위 ≈355종)면
하루 ≈920건(코인당 2~3번)이다. ⚡ 감시 종목으로 좁히고 24h에 한 번만 알리면 하루 ≈40건.

**마감된 15m봉만** 본다 — 진행 중인 봉의 저점 이탈은 마감 때 되돌려질 수 있다.
스캔이 매시 :02라 지난 한 시간의 봉(grace_bars+1개)을 소급해 본다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import SignalEvent

NAME = "choch_warn"
LABEL = "하락 CHoCH"
CRYPTO_ONLY = True
INTERVAL = "15m"
LONG_INTERVAL = "4h"
LONG_LIMIT = 1000          # 4h MA960 + 여유 — 현물 미러 요청당 상한(선물은 1500)


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    pivot_bars: int = 5         # 스윙 피벗 좌우 봉 수 (15m × 5 = 1시간 15분)
    ma_long: int = 960          # 4h MA — 종가가 이 선 위인 종목만
    grace_bars: int = 4         # 마지막 grace_bars+1개 마감봉에서 난 이탈까지 (스캔 1시간)
    rearm_hours: int = 24       # 같은 종목은 이 시간 안에 다시 안 알린다


def choch_bars(df: pd.DataFrame, pivot: int) -> list[tuple[int, float, float]]:
    """15m 하락 CHoCH가 난 봉 — (봉 위치, 깬 스윙 저점, 직전 스윙 고점) 목록.

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


def detect(df15: pd.DataFrame, symbol: str, now: pd.Timestamp,
           params: Params = Params()) -> list[SignalEvent]:
    """마감 시각이 now 전 (grace_bars+1)×15분 안인 15m봉의 하락 CHoCH. 4h 조건은 호출 측.

    df15는 인트라바 프레임이어도 된다 — 진행 중인 봉은 여기서 뗀다.
    """
    if not params.enabled or df15 is None:
        return []
    d = df15[df15.index + pd.Timedelta(minutes=15) <= now]
    if len(d) < 4 * params.pivot_bars + 2:
        return []
    step = pd.Timedelta(minutes=15)
    cutoff = now - step * (params.grace_bars + 1)       # 마감 시각이 이보다 뒤인 봉만
    out = []
    for t, low, high in choch_bars(d, params.pivot_bars):
        if d.index[t] + step <= cutoff:
            continue
        close = float(d["Close"].iloc[t])
        out.append(SignalEvent(
            symbol=symbol, signal=NAME, bar_time=d.index[t], price=close,
            detail={"label": LABEL, "interval": INTERVAL, "broken_low": low,
                    "swing_high": high,
                    "from_high": close / high - 1 if high and not np.isnan(high) else None}))
    return out[-1:]            # 한 번에 여러 봉이면 가장 최근 것만
