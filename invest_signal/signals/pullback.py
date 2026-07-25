"""눌림목(풀백) 시그널 (4h봉).

상승초입 다음 국면. 240선이 480선 위로 올라온 순간부터(골든크로스) 그
종목은 '눌림목 구간'이다:
  ① 240선 > 480선 상태에서 (상승초입 구간과 상호배타)
  ② 캔들 종가가 120선을 하회하면서
  ③ 종가가 분기 앵커드 VWAP(QVWAP) 위에 있으면 → 눌림목 알림.
     하회 봉이 아직 QVWAP 아래면 대기하고, 같은 하회 구간에서 종가가
     QVWAP 위가 되는 첫 봉에서 알린다.

같은 하회 구간에서는 조건을 처음 충족한 봉만 알리고, 120선 위로
복귀했다가 다시 하회하면 새 눌림목으로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import quarterly_vwap, sma
from . import SignalEvent

NAME = "pullback"
LABEL = "눌림목"
INTRABAR_OK = True   # 진행봉 현재가를 잠정 종가로 판정 — 알림에 ⏳진행봉 표시


@dataclass(frozen=True)
class Params:
    ma_entry: int = 120         # 하회 판정 기준선
    ma_fast: int = 240          # 눌림목 구간 판정 — 이 선이
    ma_slow: int = 480          # 이 선 위에 있어야 한다 (240 > 480)
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수
    qvwap_condition: bool = True  # ③ 트리거 종가 > 분기VWAP 요구 (Volume 없으면 자동 통과)


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
    qv = quarterly_vwap(df)

    def below_entry(t: int) -> bool:    # ② 120선 하회
        return not pd.isna(m_entry.iloc[t]) and close.iloc[t] < m_entry.iloc[t]

    def in_regime(t: int) -> bool:      # ① 240선 > 480선 (눌림목 구간)
        return (not pd.isna(m_slow.iloc[t])
                and m_fast.iloc[t] > m_slow.iloc[t])

    def is_trigger_state(t: int) -> bool:
        """② 120선 하회 + ③ (qvwap_condition이면) 종가가 분기 VWAP 위."""
        if not below_entry(t):
            return False
        if not params.qvwap_condition or qv is None or pd.isna(qv.iloc[t]):
            return True
        return bool(close.iloc[t] > qv.iloc[t])

    events = []
    for t in range(max(need + 1, n - 1 - params.grace_bars), n):
        if not in_regime(t):
            continue
        if not is_trigger_state(t):
            continue
        # 같은 하회 구간(연속 120선 아래)에서 이미 조건을 충족한 봉이
        # 있으면 첫 봉이 아니다 — QVWAP 아래서 대기하던 봉들은 세지 않는다
        j = t - 1
        while j >= 0 and below_entry(j) and not is_trigger_state(j):
            j -= 1
        if j >= 0 and below_entry(j):
            continue
        qv_val = qv_above = None
        if qv is not None and not pd.isna(qv.iloc[t]):
            qv_val = float(qv.iloc[t])
            qv_above = bool(close.iloc[t] > qv.iloc[t])
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
                "qvwap": qv_val,
                "above_qvwap": qv_above,
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """눌림목은 발생 후 계속 리스트에 남는다 — 120선 위로 복귀해 정배열로
    올라가는 동안에도 추적한다. 제거는 두 경로뿐:
    ① 하락전환(CHoCH) 발생 시 스캐너가 걷어냄 ② 조회 범위(7일) 경과.
    """
    return event.bar_time in df.index
