"""세력 이탈 경고 — 현선갭(선물 ÷ 현물)이 크게 벌어진 순간 (크립토 전용, 전 종목).

익절·청산용 **경고**다. 🔻하락 CHoCH(차트 구조)와 달리 **선물·현물 가격 차**를 본다.
아이디어는 '크라임코인 고점잡기'(포리치 블로그, 10-02 사용자 공유)에서 왔다: 세력이
선물 롱으로 끌어올린 뒤, 현물 호가로 가격을 받쳐 둔 채 선물 롱을 익절한다 — 그러면
선물이 현물보다 크게 싸진다(현선갭).

  조건  최근 마감 15m봉(gap_bars개) 중 하나라도
        ① |퍼프 ÷ 인덱스 − 1| ≥ gap_min(10%), 또는
        ② |퍼프 ÷ 인덱스 − 1| ≥ gap_pump_min(5%) 이고 그 봉이 24h 전보다 pump_min(+30%) 이상
        인덱스 = 여러 거래소 현물 가격의 가중 평균(바이낸스 선물 인덱스). 보통은 ±0.5% 안이다.
  대상  전체 USDT 퍼프 — 다른 조건(⚡ 감시·4h 960선·거래대금)은 없다(10-02 요청)
  빈도  같은 종목은 rearm_hours(24h)에 한 번

퍼프 실측(08-01~09-28, ⚡ 이력 311종, 15m 종가): ①∪②는 58일에 30건(하루 0.5건, 24종) —
이후 24h 종가 중앙 −11%, 24h 안 최저 중앙 −20%(−20% 이상 50%, −30% 이상 27%). 같은 기간
24h +30%↑ 펌핑 코인 중 갭이 없던 때는 −7%·−16%(33%·13%). ①만이면 19건 −18%·−28%(68%).
갭 5%만으로는(펌핑 조건 없이) −3%·−12%라 잡음이다. 50%는 그 전에 +20% 더 쐈다 —
**숏 신호가 아니라 롱 정리 신호**다. TAKE 09-23: 21:30봉 −11% → 22:02 스캔(0.195),
폭락은 22:30부터, 바닥 0.095.
"""

from dataclasses import dataclass

import pandas as pd

from . import SignalEvent

NAME = "whale_exit"
LABEL = "세력 이탈"
CRYPTO_ONLY = True
INTERVAL = "15m"
DAY_BARS = 96               # 15m × 96 = 24h


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    gap_min: float = 0.10       # ① 이 이상이면 무조건
    gap_pump_min: float = 0.05  # ② 이 이상이고
    pump_min: float = 0.30      #    24h 전보다 +30% 이상 오른 봉이면
    gap_bars: int = 5           # 최근 마감 15m봉 몇 개를 볼지 (5 = 75분, 매시 스캔 + 여유)
    rearm_hours: int = 24


def detect(px: pd.DataFrame | None, idx: pd.DataFrame | None, symbol: str,
           now: pd.Timestamp, params: Params = Params()) -> SignalEvent | None:
    """최근 마감 15m봉 중 조건을 넘긴 봉 — 갭이 가장 큰 봉 하나. 없으면 None.

    px는 퍼프 15m(24h 전 종가를 보려면 gap_bars + 96봉 이상), idx는 인덱스 15m.
    """
    if not params.enabled or px is None or idx is None or not len(px) or not len(idx):
        return None
    step = pd.Timedelta(minutes=15)
    c = px["Close"][px.index + step <= now]
    i = idx["Close"][idx.index + step <= now]
    gap = (c / i.reindex(c.index) - 1).iloc[-params.gap_bars:]
    pump = (c / c.shift(DAY_BARS) - 1).reindex(gap.index)
    best = None
    for t, g in gap.dropna().items():
        p = pump.get(t)
        ok = abs(g) >= params.gap_min or (
            abs(g) >= params.gap_pump_min and p is not None and not pd.isna(p)
            and p >= params.pump_min)
        if ok and (best is None or abs(g) > abs(best[1])):
            best = (t, float(g), None if p is None or pd.isna(p) else float(p))
    if best is None:
        return None
    t, g, p = best
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=t, price=float(c.iloc[-1]),
                       detail={"label": LABEL, "interval": INTERVAL, "gap": g, "pump_24h": p})
