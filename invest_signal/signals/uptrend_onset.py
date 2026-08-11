"""상승초입 시그널 (4h봉).

조건:
  ① ma_align 이동평균이 역배열(기본 MA240 < MA480)인 상태에서
  ② 캔들 고가가 240선에 닿음(터치)
  ③ 터치 후 touch_window_bars(기본 60봉 = 10일) 이내에
  ④ 종가가 60선 아래이면서 분기 앵커드 VWAP(QVWAP) "위"에 있는
     "첫" 봉 → 그 봉에서 시그널 발생. QVWAP 선이 캔들 위에 있다가
     (분기 리셋 등으로) 아래로 확 내려와 종가가 그 위로 올라선 순간을
     잡는다 — 60선을 깼는데 아직 QVWAP 아래면 대기하고, 종가가 QVWAP
     위가 되는 봉에서 알린다.

같은 터치에 대해 그 상태가 계속 유지돼도 첫 봉에서만 발생한다.
새로운 240 터치가 나오면 다시 발생할 수 있다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import quarterly_vwap, sma
from . import SignalEvent

NAME = "uptrend_onset"
LABEL = "상승초입"
INTRABAR_OK = True   # 진행봉 현재가를 잠정 종가로 판정 — 알림에 ⏳진행봉 표시


@dataclass(frozen=True)
class Params:
    ma_entry: int = 60          # 이탈 판정 기준선
    ma_align: tuple = (240, 480)        # 역배열 판정선 (짧은 것부터)
    ma_touch: int = 240         # 터치 판정 기준선
    touch_window_bars: int = 60  # 터치 유효기간(봉 수). 4h×60 = 10일
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수
    qvwap_condition: bool = True  # ④ 트리거 종가 > 분기VWAP 요구 (Volume 없으면 자동 통과)


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 상승초입들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립된 트리거 후보로 판정하므로,
    스캔을 한 번 놓쳐도 직전 봉에서 발생한 시그널을 소급해 잡는다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    need = max(*params.ma_align, params.ma_entry, params.ma_touch)
    n = len(df)
    if n < need + 2:            # 최장 MA가 유효한 봉이 최소 2개는 있어야 터치→이탈이 가능
        return []

    close, high, low = df["Close"], df["High"], df["Low"]
    m_entry = sma(close, params.ma_entry)
    aligns = [sma(close, k) for k in params.ma_align]   # 선 개수는 설정에 따라 2~N
    m_touch = sma(close, params.ma_touch)
    qv = quarterly_vwap(df)

    last = n - 1

    def bearish(i: int) -> bool:
        """ma_align 선들이 짧은 것부터 오름차순 — 역배열.

        선 개수는 설정에 따라 다르다(기본 240·480 두 선). MA를 못 구하는
        구간은 판정하지 않는다.
        """
        vals = [m.iloc[i] for m in aligns]
        if any(pd.isna(v) for v in vals):
            return False
        return all(vals[j] < vals[j + 1] for j in range(len(vals) - 1))

    def is_touch(i: int) -> bool:
        """역배열 + 터치선이 캔들 범위 안.

        고가만 보면 캔들 전체가 터치선 위로 올라탄 봉도 매봉 '터치'가 되어
        터치 시점이 계속 갱신된다 — 저가가 터치선 이하인지도 본다.
        """
        if pd.isna(m_touch.iloc[i]):
            return False
        return bearish(i) and high.iloc[i] >= m_touch.iloc[i] >= low.iloc[i]

    def below_entry(i: int) -> bool:
        return not pd.isna(m_entry.iloc[i]) and close.iloc[i] < m_entry.iloc[i]

    def is_trigger_state(i: int) -> bool:
        """60선 아래 + (qvwap_condition이면) 종가가 분기 VWAP 위인 상태.

        60선 이탈 봉이 아직 QVWAP 아래면 발화하지 않고 대기 — QVWAP 선이
        내려와 종가가 그 위가 되는 첫 봉에서 발화한다.
        """
        if not below_entry(i):
            return False
        if not params.qvwap_condition or qv is None or pd.isna(qv.iloc[i]):
            return True
        return bool(close.iloc[i] > qv.iloc[i])

    events = []
    for t in range(max(need, last - params.grace_bars), last + 1):
        if not is_trigger_state(t):
            continue
        # 가장 긴 두 선이 뒤집혔으면(짧은 쪽이 위) 이미 상승 국면 —
        # 역배열 반전을 잡는 시그널이라 대상이 아니다
        long2, long1 = aligns[-2].iloc[t], aligns[-1].iloc[t]
        if not pd.isna(long1) and not pd.isna(long2) and long2 > long1:
            continue
        touch_idx = None        # t 직전 touch_window 안에서 가장 최근 터치
        for i in range(t - 1, max(need - 1, t - params.touch_window_bars) - 1, -1):
            if is_touch(i):
                touch_idx = i
                break
        if touch_idx is None:
            continue
        if any(is_trigger_state(j) for j in range(touch_idx + 1, t)):
            continue            # t가 터치 이후 "첫" 이탈 봉이 아님
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
                "ma240": float(m_touch.iloc[t]) if not pd.isna(m_touch.iloc[t]) else None,
                "touch_time": df.index[touch_idx].isoformat(),
                "touch_high": float(high.iloc[touch_idx]),
                "qvwap": qv_val,
                "above_qvwap": qv_above,
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """상승초입은 발생 후 상승 과정을 계속 추적한다(60선 위 복귀해도 유지).

    제거 조건: 종가가 240선 위로 올라섰다가 다시 240선 아래로 마감하면
    돌파 실패로 제거. 아직 240선을 넘지 못한 동안은 계속 유지되고,
    CHoCH(하락전환)로는 제거하지 않는다. 조회 범위(7일) 경과 시 자동 제거.
    """
    try:
        t = df.index.get_loc(event.bar_time)
    except KeyError:
        return False
    c = df["Close"].iloc[t:]
    m240 = sma(df["Close"], params.ma_touch).iloc[t:]
    above = c > m240
    if not above.any():
        return True                  # 240선 재도전 전 — 계속 추적
    first_above = above.idxmax()     # 첫 240선 상향 마감 시점
    fail = c.loc[first_above:] < m240.loc[first_above:]
    return not bool(fail.any())      # 돌파 후 재이탈했으면 실패로 제거
