"""상승초입 시그널 (4h봉).

조건:
  ① ma_align 이동평균이 역배열(기본 MA120 < MA240 < MA480)인 상태에서
  ② 종가가 ma_entry(기본 20)선 아래이고
  ③ 앵커드 VWAP(기본 월간 MVWAP) 조건을 만족하고
       above — 종가가 VWAP 위 (선 위로 올라선 것을 확인)
       touch — 최근 vwap_touch_bars봉 안에 VWAP이 캔들 범위 안 (선까지 눌림)
       any   — 둘 중 하나
  ④ 수퍼트렌드(4h, ATR 22 × 3)가 상승추세인
     "첫" 봉 → 그 봉에서 시그널 발생. 조건을 아직 못 채운 이탈 봉은
     발화하지 않고 대기하다가, 조건이 성립하는 첫 봉에서 알린다.

①과 ④를 함께 걸면 "장기 구조는 아직 하락(역배열)인데 단기 추세는 이미
위로 돌아선" 구간만 남는다 — 그 상태에서의 첫 눌림이 이 시그널의 타깃이다.

touch_condition을 켜면 ① 앞에 **240선 터치**가 추가된다 — 역배열 상태에서
캔들이 240선을 걸친 봉이 touch_window_bars 안에 있어야 하고, '첫 봉'의
기준점도 그 터치가 된다. 기본은 꺼져 있다.

조건이 계속 유지돼도 첫 봉에서만 발생하고, 상태를 벗어났다 다시 들어오면
새 시그널로 다시 발생한다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import anchored_vwap, pct_over, sma, supertrend
from . import SignalEvent

NAME = "uptrend_onset"
LABEL = "상승초입"
INTRABAR_OK = True   # 진행봉 현재가를 잠정 종가로 판정 — 알림에 ⏳진행봉 표시


@dataclass(frozen=True)
class Params:
    ma_entry: int = 20          # 이탈 판정 기준선 (4h×20 = 약 3.3일)
    ma_align: tuple = (120, 240, 480)   # 역배열 판정선 (짧은 것부터)
    touch_condition: bool = False  # 240선 터치를 추가로 요구할지 (아래 설명 참고)
    ma_touch: int = 240         # 터치 판정 기준선 (touch_condition을 켤 때만 쓰인다)
    touch_window_bars: int = 60  # 터치 유효기간(봉 수). 4h×60 = 10일
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수
    vwap_condition: bool = True  # ④ VWAP 조건 사용 여부 (Volume 없으면 자동 통과)
    vwap_period: str = "M"       # 앵커드 VWAP 리셋 주기 — M(월간)/Q(분기)/W(주간)
    # ④의 판정 방식:
    #   above  — 트리거 봉 종가가 VWAP 위 (선 위로 올라선 것을 확인하고 진입)
    #   touch  — 최근 vwap_touch_bars봉 안에 VWAP이 캔들 범위 안 (선까지 눌린 자리)
    #   any    — 둘 중 하나면 통과
    vwap_mode: str = "any"
    vwap_touch_bars: int = 5     # touch/any 소급 창 — 5봉 = 20시간
    supertrend_condition: bool = True   # 수퍼트렌드가 상승추세일 때만 발화
    st_period: int = 22          # 수퍼트렌드 ATR 기간
    st_mult: float = 3.0         # 수퍼트렌드 ATR 배수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 상승초입들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립된 트리거 후보로 판정하므로,
    스캔을 한 번 놓쳐도 직전 봉에서 발생한 시그널을 소급해 잡는다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    # 실제로 쓰는 선만 최소 이력에 넣는다 — 터치 조건이 꺼져 있으면 ma_touch는
    # 계산에 안 들어가므로, 그것 때문에 신규 상장을 걸러낼 이유가 없다.
    lines = [*params.ma_align, params.ma_entry]
    if params.touch_condition:
        lines.append(params.ma_touch)
    need = max(lines)
    n = len(df)
    if n < need + 2:            # 최장 MA가 유효한 봉이 최소 2개는 있어야 판정이 가능
        return []

    close, high, low = df["Close"], df["High"], df["Low"]
    m_entry = sma(close, params.ma_entry)
    aligns = [sma(close, k) for k in params.ma_align]   # 선 개수는 설정에 따라 2~N
    m_touch = sma(close, params.ma_touch)
    vw = anchored_vwap(df, params.vwap_period)
    st = (supertrend(df, params.st_period, params.st_mult)
          if params.supertrend_condition else None)

    last = n - 1

    def bearish(i: int) -> bool:
        """ma_align 선들이 짧은 것부터 오름차순 — 역배열.

        선 개수는 설정에 따라 다르다(기본 120·240·480 세 선). MA를 못
        구하는 구간은 판정하지 않는다.
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

    def vwap_touched(i: int) -> bool:
        """최근 vwap_touch_bars봉 안에 VWAP선이 캔들 범위 안에 들어왔는지.

        240선 터치와 같은 정의(저가 ≤ 선 ≤ 고가)를 VWAP에 적용한 것이다.
        선을 '위에서 눌러 닿은' 자리를 잡으려는 조건이라, 종가가 선 위든
        아래든 상관하지 않는다.
        """
        for j in range(max(0, i - params.vwap_touch_bars + 1), i + 1):
            if pd.isna(vw.iloc[j]):
                continue
            if high.iloc[j] >= vw.iloc[j] >= low.iloc[j]:
                return True
        return False

    def vwap_ok(i: int) -> bool:
        """④ VWAP 조건 — vwap_mode에 따라 '위'/'터치'/'둘 중 하나'."""
        if not params.vwap_condition or vw is None or pd.isna(vw.iloc[i]):
            return True
        above = bool(close.iloc[i] > vw.iloc[i])
        if params.vwap_mode == "above":
            return above
        if params.vwap_mode == "touch":
            return vwap_touched(i)
        return above or vwap_touched(i)          # any

    def uptrend_st(i: int) -> bool:
        """수퍼트렌드가 상승추세(+1)인지. ATR이 아직 없는 구간은 불통과.

        역배열과 함께 걸면 '장기 구조는 아직 하락이지만 단기 추세는 이미
        위로 돌아선' 구간만 남는다 — 그 상태에서의 첫 눌림이 타깃이다.
        """
        if st is None:
            return True
        v = st.iloc[i]
        return not pd.isna(v) and v > 0

    def is_trigger_state(i: int) -> bool:
        """①역배열 + 이탈선 아래 + VWAP 조건 + 수퍼트렌드 상승을 모두 만족.

        조건을 아직 못 채운 이탈 봉은 발화하지 않고 대기 — 조건이 성립하는
        첫 봉에서 발화한다. 역배열이 깨졌다 다시 성립해도 새 셋업이 된다.
        """
        return bearish(i) and below_entry(i) and vwap_ok(i) and uptrend_st(i)

    events = []
    for t in range(max(need, last - params.grace_bars), last + 1):
        if not is_trigger_state(t):
            continue
        touch_idx = None
        if params.touch_condition:
            for i in range(t - 1, max(need - 1, t - params.touch_window_bars) - 1, -1):
                if is_touch(i):     # t 직전 touch_window 안에서 가장 최근 터치
                    touch_idx = i
                    break
            if touch_idx is None:
                continue
            if any(is_trigger_state(j) for j in range(touch_idx + 1, t)):
                continue        # t가 터치 이후 "첫" 이탈 봉이 아님
        # 터치 조건이 없으면 기준 시점이 없으므로 '상태에 막 들어선 봉'을
        # 첫 봉으로 본다 — 직전 봉이 이미 같은 상태면 이어지는 중이다
        elif is_trigger_state(t - 1):
            continue
        vw_val = vw_above = None
        if vw is not None and not pd.isna(vw.iloc[t]):
            vw_val = float(vw.iloc[t])
            vw_above = bool(close.iloc[t] > vw.iloc[t])
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
                "touch_time": (df.index[touch_idx].isoformat()
                               if touch_idx is not None else None),
                "touch_high": (float(high.iloc[touch_idx])
                               if touch_idx is not None else None),
                "vwap": vw_val,
                "above_vwap": vw_above,
                "vwap_period": params.vwap_period,
                "vwap_mode": params.vwap_mode,
                "supertrend": (None if st is None or pd.isna(st.iloc[t])
                               else int(st.iloc[t])),
                # 알림 줄에 붙일 종목 수익률 — 1h는 4h봉으로 못 구하므로
                # 스캐너가 따로 받아서 채운다(ret_1h).
                "ret_4h": pct_over(close, 1, t),
                "ret_24h": pct_over(close, 6, t),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """상승초입은 발생 후 상승 과정을 계속 추적한다(이탈선 위 복귀해도 유지).

    제거는 **수퍼트렌드가 하락으로 뒤집힐 때** 하나뿐이다. 이 시그널의
    전제가 "구조는 아직 하락인데 추세는 위로 돌아섰다"이므로, 그 추세가
    꺾이면 셋업 자체가 사라진 것이다 — 진입 조건이면서 해제 조건이다.
    그 외에는 조회 범위(3일)가 지나면 자동으로 빠진다.

    240선 돌파 실패는 더 이상 보지 않는다. 240선은 원래 진입 조건(터치)의
    기준선이었는데 그 조건을 끄면서 해제 판정에만 남아 있었고, 240선을
    넘었다 밀리는 건 상승 과정에서 흔한 되돌림이라 셋업이 끝났다는 근거로
    쓰기엔 약하다. 추세 판단은 수퍼트렌드에 맡긴다.
    CHoCH(하락전환)로도 제거하지 않는다.
    """
    if event.bar_time not in df.index:
        return False
    if params.supertrend_condition:
        st = supertrend(df, params.st_period, params.st_mult).dropna()
        if len(st) and st.iloc[-1] <= 0:
            return False             # 추세가 꺾였다 — 셋업 종료
    return True
