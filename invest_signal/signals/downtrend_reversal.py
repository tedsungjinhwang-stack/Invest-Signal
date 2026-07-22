"""하락전환 시그널 (4h봉) — bearish CHoCH.

상승 구조가 깨지는 순간을 잡는다:
  ① 종가가 60선 아래로 내려갔던 직전 눌림 구간이 완성되고
     (종가가 60선 위로 복귀해서 구간이 끝남)
  ② 그 구간의 최저 저가(꼬리 포함)를 직전저점으로 삼아
  ③ 이후 lookback_bars(기본 180봉 = 30일) 안에 종가가 그 직전저점
     아래로 "처음" 마감하면 → 하락전환 알림.

진행 중인 눌림 안에서 저점을 계속 갱신하는 것은 돌파로 치지 않는다 —
완성된 직전저점의 하향돌파(CHoCH)만 잡는다. 돌파 후 봉들이 계속 그
아래 머물러도 첫 돌파 봉에서만 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import sma
from . import SignalEvent

NAME = "downtrend_reversal"
LABEL = "하락전환"


@dataclass(frozen=True)
class Params:
    ma_ref: int = 60            # 눌림 판정 기준선
    lookback_bars: int = 180    # 직전저점 유효기간(봉 수). 4h×180 = 30일
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 하락전환들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    need = params.ma_ref + 1
    n = len(df)
    if n < need + 2:
        return []

    close, low = df["Close"], df["Low"]
    m = sma(close, params.ma_ref)
    first = params.ma_ref - 1    # m이 유효한 첫 인덱스

    def is_below(i: int) -> bool:
        return not pd.isna(m.iloc[i]) and close.iloc[i] < m.iloc[i]

    events = []
    for t in range(max(need, n - 1 - params.grace_bars), n):
        # 현재 진행 중인 눌림 구간(있다면)을 건너뛰고,
        # 그 앞의 '복귀 구간'을 지나 직전에 완성된 눌림 구간을 찾는다.
        i = t - 1
        while i >= first and is_below(i):
            i -= 1
        while i >= first and not is_below(i):
            i -= 1
        if i < first:
            continue
        e_end = i               # 직전 완성 눌림 구간의 마지막 봉
        if t - e_end > params.lookback_bars:
            continue
        e_start = e_end
        while e_start - 1 >= first and is_below(e_start - 1):
            e_start -= 1
        seg_low = low.iloc[e_start:e_end + 1]
        level = float(seg_low.min())
        if not (close.iloc[t] < level and close.iloc[t - 1] >= level):
            continue            # 첫 하향돌파 봉만
        low_idx = e_start + int(seg_low.values.argmin())
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "broken_low": level,
                "low_time": df.index[low_idx].isoformat(),
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
