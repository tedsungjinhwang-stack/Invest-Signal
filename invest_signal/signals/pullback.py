"""눌림목 시그널 (4h봉).

상승 추세 안의 조정 자리를 분기 VWAP **하단 밴드**로 잡는다:
  ① 종가가 ma_above(기본 480)선 **위** — 장기 추세는 살아 있어야 한다
  ② 종가가 ma_entry(기본 60)선 **하회** — 단기 조정에 들어감
  ③ 캔들이 분기 VWAP 하단 밴드(VWAP − mult×σ) **위**에 있어야 한다
     (고가가 밴드 아래면 이미 밴드를 잃은 것이라 대상이 아님)

③ 안에서 두 단계로 나뉜다:
  - **대기**: 캔들 전체가 밴드 위 (저가 > 밴드) — 아직 눌림이 덜 왔다
  - **타점**: 캔들이 밴드에 닿음 (저가 ≤ 밴드 ≤ 고가) — 진입 자리

같은 하회 구간(연속으로 60선 아래인 구간)에서 대기는 첫 충족 봉 한 번,
타점은 첫 터치 봉 한 번만 알린다. 60선 위로 복귀했다가 다시 하회하면
새 눌림목으로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import quarterly_vwap_bands, sma, supertrend
from . import SignalEvent

NAME = "pullback"
LABEL = "눌림목"
INTRABAR_OK = True   # 진행봉 현재가를 잠정 종가로 판정 — 알림에 ⏳진행봉 표시

STAGE_WAIT = "대기"
STAGE_ENTRY = "타점"


@dataclass(frozen=True)
class Params:
    ma_entry: int = 60          # ② 이 선을 하회해야 한다 (단기 조정)
    ma_above: int = 480         # ① 종가가 이 선 위여야 한다 (장기 추세)
    band_mult: float = 1.0      # ③ 하단 밴드 = 분기VWAP − mult×표준편차
    band_condition: bool = True  # ③ 밴드 조건 사용 (Volume 없으면 자동 통과)
    grace_bars: int = 1
    supertrend_exit: bool = True   # 수퍼트렌드가 하락으로 뒤집히면 추적 해제
    st_period: int = 22            # 수퍼트렌드 ATR 기간
    st_mult: float = 3.0           # 수퍼트렌드 ATR 배수         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


class Gate:
    """눌림목 진입 조건(①②③) 판정기 — MSS 등 같은 칸 시그널이 공유한다.

    지표를 한 번만 계산해두고 봉 인덱스로 물어본다.
    """

    def __init__(self, df: pd.DataFrame, params: "Params"):
        self.df, self.params = df, params
        close = df["Close"]
        self.m_entry = sma(close, params.ma_entry)
        self.m_above = sma(close, params.ma_above)
        bands = quarterly_vwap_bands(df, params.band_mult) if params.band_condition else None
        self.vwap, self.lower = (bands[0], bands[1]) if bands is not None else (None, None)

    @property
    def need(self) -> int:
        return max(self.params.ma_entry, self.params.ma_above)

    def band_at(self, t: int) -> float | None:
        if self.lower is None or pd.isna(self.lower.iloc[t]):
            return None
        return float(self.lower.iloc[t])

    def below_entry(self, t: int) -> bool:      # ② 60선 하회
        return (not pd.isna(self.m_entry.iloc[t])
                and self.df["Close"].iloc[t] < self.m_entry.iloc[t])

    def above_long(self, t: int) -> bool:       # ① 480선 위
        return (not pd.isna(self.m_above.iloc[t])
                and self.df["Close"].iloc[t] > self.m_above.iloc[t])

    def qualifies(self, t: int) -> bool:
        """①②③ — 밴드 위(또는 밴드에 닿은) 상태."""
        if not self.below_entry(t) or not self.above_long(t):
            return False
        band = self.band_at(t)
        return band is None or float(self.df["High"].iloc[t]) >= band

    def is_entry(self, t: int) -> bool:
        """③-타점 — 캔들이 하단 밴드에 닿았다."""
        band = self.band_at(t)
        return (self.qualifies(t) and band is not None
                and float(self.df["Low"].iloc[t]) <= band)

    def stage(self, t: int) -> str | None:
        """대기/타점 — 밴드가 없으면(band_condition: false) 구분 자체가 없어 None.

        타점은 '하단 밴드 터치'로 정의되므로 밴드를 끄면 나눌 기준이 사라진다.
        None이면 알림에 단계 태그가 붙지 않고 dedup 키에도 들어가지 않는다.
        """
        if self.band_at(t) is None:
            return None
        return STAGE_ENTRY if self.is_entry(t) else STAGE_WAIT

    def first_in_dip(self, t: int, pred) -> bool:
        """같은 하회 구간에서 pred를 만족한 앞선 봉이 없으면 True."""
        j = t - 1
        while j >= 0 and self.below_entry(j) and not pred(j):
            j -= 1
        return not (j >= 0 and self.below_entry(j))


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 눌림목들을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    g = Gate(df, params)
    if n < g.need + 2:
        return []

    events = []
    for t in range(max(g.need + 1, n - 1 - params.grace_bars), n):
        if not g.qualifies(t):
            continue
        entry = g.is_entry(t)
        # 타점은 구간 내 첫 터치만, 대기는 구간 내 첫 충족만 알린다
        if not g.first_in_dip(t, g.is_entry if entry else g.qualifies):
            continue
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(df["Close"].iloc[t]),
            detail={
                "label": LABEL,
                "stage": g.stage(t),
                "entry_ma": float(g.m_entry.iloc[t]),
                "entry_ma_period": params.ma_entry,
                "long_ma": float(g.m_above.iloc[t]),
                "long_ma_period": params.ma_above,
                "band": g.band_at(t),
                "qvwap": (None if g.vwap is None or pd.isna(g.vwap.iloc[t])
                          else float(g.vwap.iloc[t])),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """발생 후에도 계속 추적한다 — 60선 위로 복귀해 올라가는 동안에도 남는다.
    제거는 다섯 경로:
    ① 하락전환(CHoCH) 발생 시 스캐너가 걷어냄 ② 조회 범위(5일) 경과
    ③ 종가가 분기 VWAP 하단 밴드 아래로 마감 ④ 종가가 480선 아래로 마감
    ⑤ 수퍼트렌드가 하락으로 뒤집힘
    (③④⑤는 셋업의 전제가 깨진 것)

    ⑤는 진입 조건이 아니라 해제 조건으로만 쓴다 — 눌림목은 "장기 추세가
    살아 있는 동안의 조정"이므로, 추세 자체가 꺾이면 조정이 아니라 전환이다.
    """
    if event.bar_time not in df.index:
        return False
    if params.supertrend_exit:
        st = supertrend(df, params.st_period, params.st_mult).dropna()
        if len(st) and st.iloc[-1] <= 0:
            return False
    close = df["Close"]
    m_above = sma(close, params.ma_above)
    if pd.isna(m_above.iloc[-1]) or close.iloc[-1] <= m_above.iloc[-1]:
        return False
    if params.band_condition:
        bands = quarterly_vwap_bands(df, params.band_mult)
        if bands is not None and not pd.isna(bands[1].iloc[-1]):
            return bool(close.iloc[-1] > bands[1].iloc[-1])
    return True
