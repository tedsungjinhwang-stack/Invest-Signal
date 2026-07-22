"""눌림목(풀백) 시그널 (4h봉).

상승초입 다음 국면. 240선이 480선 위로 올라온 순간부터(골든크로스) 그
종목은 '눌림목 구간'이다:
  ① 240선 > 480선 상태에서 (상승초입 구간과 상호배타)
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
    ma_fast: int = 240          # 눌림목 구간 판정 — 이 선이
    ma_slow: int = 480          # 이 선 위에 있어야 한다 (240 > 480)
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 눌림목들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    need = max(params.ma_entry, params.ma_slow)
    n = len(df)
    if n < need + 2:
        return []

    close = df["Close"]
    m_entry = sma(close, params.ma_entry)
    m_fast = sma(close, params.ma_fast)
    m_slow = sma(close, params.ma_slow)

    def below_entry(t: int) -> bool:    # ② 120선 하회
        return not pd.isna(m_entry.iloc[t]) and close.iloc[t] < m_entry.iloc[t]

    def in_regime(t: int) -> bool:      # ① 240선 > 480선 (눌림목 구간)
        return (not pd.isna(m_slow.iloc[t])
                and m_fast.iloc[t] > m_slow.iloc[t])

    events = []
    for t in range(max(need + 1, n - 1 - params.grace_bars), n):
        if not in_regime(t):
            continue
        if not below_entry(t):
            continue
        if below_entry(t - 1):
            continue            # 연속 하회 구간은 첫 봉만
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "entry_ma": float(m_entry.iloc[t]),
                "entry_ma_period": params.ma_entry,
                "ma240": float(m_fast.iloc[t]),
                "ma480": float(m_slow.iloc[t]),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """트리거 이후 종가가 계속 120선 아래이고 240>480 구간이 유지되면 '유지 중'.

    240선이 480선 아래로 되돌아가면 눌림목 구간 자체가 끝난 것이므로 뺀다.
    """
    try:
        t = df.index.get_loc(event.bar_time)
    except KeyError:
        return False
    c = df["Close"].iloc[t:]
    m_entry = sma(df["Close"], params.ma_entry).iloc[t:]
    m_fast = sma(df["Close"], params.ma_fast).iloc[t:]
    m_slow = sma(df["Close"], params.ma_slow).iloc[t:]
    return bool((c < m_entry).all() and (m_fast > m_slow).all())
