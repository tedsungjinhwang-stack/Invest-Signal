"""MSS 시그널 (4h봉) — 직전 스윙저점 하향돌파.

하락전환(60선 눌림 저점 CHoCH)과 달리 조건 없이, 가장 최근에 확정된
스윙저점(좌우 pivot_k봉보다 낮은 저가)을 종가가 하향돌파하면 알린다:
  ① 저가가 좌우 pivot_k봉(기본 6봉 = 24시간)의 저가보다 낮아 스윙저점으로 확정되고
  ② 이후 종가가 그 저점 아래로 "처음" 마감하면 → MSS 알림.

돌파 후 봉들이 계속 그 아래 머물러도 첫 돌파 봉에서만 알리고,
가격이 저점 위로 복귀했다가 다시 깨면 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from . import SignalEvent

NAME = "mss"
LABEL = "MSS"


@dataclass(frozen=True)
class Params:
    pivot_k: int = 6            # 스윙저점 판정 — 좌우 k봉(6봉=24h)보다 낮아야 함
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' MSS들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    k = params.pivot_k
    if n < 2 * k + 3:
        return []

    close = df["Close"]
    lv = df["Low"].values

    events = []
    for t in range(max(2 * k + 2, n - 1 - params.grace_bars), n):
        # t 이전에 확정된(우측 k봉까지 t-1 안에 있는) 가장 최근 스윙저점
        piv = None
        for j in range(t - k - 1, k - 1, -1):
            if lv[j] == lv[j - k:j + k + 1].min():
                piv = j
                break
        if piv is None:
            continue
        level = float(lv[piv])
        if not (close.iloc[t] < level and close.iloc[t - 1] >= level):
            continue            # 첫 하향돌파 봉만
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "broken_low": level,
                "low_time": df.index[piv].isoformat(),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """트리거 이후 종가가 계속 깨진 직전저점 아래에 머물러 있으면 '유지 중'."""
    try:
        t = df.index.get_loc(event.bar_time)
    except KeyError:
        return False
    return bool((df["Close"].iloc[t:] < event.detail["broken_low"]).all())
