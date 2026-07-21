"""상승초입 시그널 (4h봉).

조건:
  ① 120·240·480 이동평균이 역배열(MA120 < MA240 < MA480)인 상태에서
  ② 캔들 고가가 240선에 닿음(터치)
  ③ 터치 후 touch_window_bars(기본 60봉 = 10일) 이내에
     종가가 60선 아래로 마감하는 "첫" 봉 → 그 봉에서 시그널 발생.

같은 터치에 대해 종가가 60선 아래에 계속 머물러도 첫 이탈 봉에서만
발생한다(그 뒤 봉들은 stale). 새로운 240 터치가 나오면 다시 발생할 수 있다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import quarterly_vwap, sma
from . import SignalEvent

NAME = "uptrend_onset"
LABEL = "상승초입"


@dataclass(frozen=True)
class Params:
    ma_entry: int = 60          # 이탈 판정 기준선
    ma_align: tuple = (120, 240, 480)   # 역배열 판정 3선
    ma_touch: int = 240         # 터치 판정 기준선
    touch_window_bars: int = 60  # 터치 유효기간(봉 수). 4h×60 = 10일
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수
    require_above_qvwap: bool = False   # True면 트리거 봉 종가가 분기 VWAP 위일 때만


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> SignalEvent | None:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 상승초입 1건을 찾는다.

    트리거 봉이 마지막 봉(또는 grace_bars 이내)일 때만 이벤트를 돌려준다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    need = max(*params.ma_align, params.ma_entry, params.ma_touch)
    n = len(df)
    if n < need + 2:            # 최장 MA가 유효한 봉이 최소 2개는 있어야 터치→이탈이 가능
        return None

    close, high = df["Close"], df["High"]
    m_entry = sma(close, params.ma_entry)
    a1, a2, a3 = (sma(close, k) for k in params.ma_align)
    m_touch = sma(close, params.ma_touch)

    last = n - 1
    # 트리거가 last-grace 이후 봉이려면 터치는 그보다 앞이면서 window 이내여야 한다.
    lo = max(need - 1, last - params.touch_window_bars)
    touch_idx = None
    for i in range(last - 1, lo - 1, -1):   # 가장 최근 터치를 찾는다
        if pd.isna(a3.iloc[i]) or pd.isna(m_touch.iloc[i]):
            continue
        aligned = a1.iloc[i] < a2.iloc[i] < a3.iloc[i]
        if aligned and high.iloc[i] >= m_touch.iloc[i]:
            touch_idx = i
            break
    if touch_idx is None:
        return None

    # 터치 이후 종가가 60선 아래로 마감한 "첫" 봉을 찾는다.
    for i in range(touch_idx + 1, last + 1):
        if pd.isna(m_entry.iloc[i]):
            continue
        if close.iloc[i] < m_entry.iloc[i]:
            if i < last - params.grace_bars:
                return None      # 이미 지나간 시그널 — 이번 실행에선 알리지 않음
            qv = quarterly_vwap(df)
            qv_val = None
            if qv is not None and not pd.isna(qv.iloc[i]):
                qv_val = float(qv.iloc[i])
            if params.require_above_qvwap and (qv_val is None or close.iloc[i] <= qv_val):
                return None
            return SignalEvent(
                symbol=symbol,
                signal=NAME,
                bar_time=df.index[i],
                price=float(close.iloc[i]),
                detail={
                    "label": LABEL,
                    "ma60": float(m_entry.iloc[i]),
                    "ma240": float(m_touch.iloc[i]) if not pd.isna(m_touch.iloc[i]) else None,
                    "touch_time": df.index[touch_idx].isoformat(),
                    "touch_high": float(high.iloc[touch_idx]),
                    "qvwap": qv_val,
                    "above_qvwap": (float(close.iloc[i]) > qv_val) if qv_val else None,
                },
            )
    return None
