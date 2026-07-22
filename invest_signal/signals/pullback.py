"""눌림목(풀백) 시그널 (4h봉).

상승초입 다음 국면 — 240선을 뚫고 올라간 뒤 눌릴 때 잡는다:
  ① 종가가 240선을 상향돌파(직전 봉 종가 ≤ 240선 → 해당 봉 종가 > 240선)한 뒤
     breakout_window_bars(기본 180봉 = 30일) 이내이고, 돌파 후 종가가
     240선 위를 유지하는 동안
  ② 캔들 종가가 120선을 하회하면 → 눌림목 알림.

②를 연속으로 충족하는 봉들은 첫 봉만 알리고, 120선 위로 복귀했다가
다시 하회하면 새 눌림목으로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import sma
from . import SignalEvent

NAME = "pullback"
LABEL = "눌림목"


@dataclass(frozen=True)
class Params:
    ma_entry: int = 120         # 하회 판정 기준선
    ma_break: int = 240         # 상향돌파 판정 기준선
    breakout_window_bars: int = 180  # 돌파 유효기간(봉 수). 4h×180 = 30일
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 눌림목들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    need = max(params.ma_entry, params.ma_break)
    n = len(df)
    if n < need + 2:
        return []

    close = df["Close"]
    m_entry = sma(close, params.ma_entry)
    m_break = sma(close, params.ma_break)

    last = n - 1

    def below_entry(t: int) -> bool:    # ② 120선 하회
        return not pd.isna(m_entry.iloc[t]) and close.iloc[t] < m_entry.iloc[t]

    events = []
    for t in range(max(need + 1, last - params.grace_bars), last + 1):
        if not below_entry(t):
            continue
        if below_entry(t - 1):
            continue            # 연속 하회 구간은 첫 봉만
        # t 직전 window 안의 240선 상향돌파를 찾는다 — 돌파 후 종가가
        # 240선 위를 유지한 구간을 거슬러 올라가며 돌파 봉을 확인.
        cross = None
        for i in range(t - 1, max(need, t - params.breakout_window_bars) - 1, -1):
            if pd.isna(m_break.iloc[i]) or pd.isna(m_break.iloc[i - 1]):
                break
            if close.iloc[i] <= m_break.iloc[i]:
                break           # 240선 위 유지가 깨짐(또는 돌파 이전 구간)
            if close.iloc[i - 1] <= m_break.iloc[i - 1]:
                cross = i       # i가 상향돌파 봉
                break
        if cross is None:
            continue
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "entry_ma": float(m_entry.iloc[t]),
                "entry_ma_period": params.ma_entry,
                "ma240": float(m_break.iloc[t]) if not pd.isna(m_break.iloc[t]) else None,
                "cross_time": df.index[cross].isoformat(),
            },
        ))
    return events
