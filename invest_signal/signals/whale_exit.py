"""세력 이탈 경고 — ⚡ 모멘텀 감시 종목에서 '작전 펌핑'이 끝나는 흔적 (크립토 전용).

익절·청산용 **경고**다. 🔻하락 CHoCH(차트 구조)와 달리 **선물 쪽 수급**을 본다.
아이디어는 '크라임코인 고점잡기'(포리치 블로그, 10-02 사용자 공유)에서 왔다:
세력이 선물 롱을 쌓으며 가격을 올리고(OI↑·가격↑), 다 올린 뒤에는 현물 호가로 가격을
받쳐 둔 채 선물 롱을 익절한다. 그러면 두 흔적이 남는다.

  ① 현선갭  퍼프 가격이 인덱스(여러 거래소 현물 가중 평균)보다 gap_min(10%) 이상
           벌어진 5m 마감봉 — 현물은 받치고 선물은 던지는 중
  ② OI 되돌림  24h 안에 +50% 이상 펌핑하는 동안 OI가 +30% 이상 늘었는데, 지금 OI는
           펌핑 시작 수준(+10% 이내)으로 돌아왔고 가격은 상승분의 60% 이상을 지키는 중
           — 롱은 이미 빠졌는데 가격만 떠 있다

둘 중 하나면 알린다. 같은 종목은 rearm_hours(24h)에 한 번.

퍼프 실측(08-01~09-28, ⚡ 감시 이력 311종): ①은 58일에 19건(하루 0.3건, 14종) — 직전
24h 중앙 +52%였고 이후 24h 종가 중앙 −18%, 24h 안 최저 중앙 −28%(−20% 이상 68%,
−30% 이상 37%). 같은 +40%↑ 펌핑 코인 중 갭이 없던 때(282건)는 −8%·−20%(51%·19%)라
갭이 붙으면 낙폭이 더 깊다. 다만 42%는 그 전에 +20% 더 쏘았다(AKE +154%, LSK +298%)
— **숏 신호가 아니라 롱 정리 신호**로 쓴다. ②는 OI 이력 20일·252종에서 TAKE 09-23
한 번(이후 −62%)뿐이라 검증이 안 됐다 — 표시만 한다.

4h 960선 조건은 없다 — 작전 코인은 대개 상장 160일이 안 돼 🔻하락 CHoCH에서 빠진다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import SignalEvent

NAME = "whale_exit"
LABEL = "세력 이탈"
CRYPTO_ONLY = True
GAP_INTERVAL = "5m"
OI_PERIOD = "1h"


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    gap_min: float = 0.10       # |퍼프 ÷ 인덱스 − 1| — 이 이상 벌어진 5m 마감봉
    gap_bars: int = 15          # 최근 마감 5m봉 몇 개를 볼지 (15 = 75분, 매시 스캔 + 여유)
    oi_enabled: bool = True
    oi_window_h: int = 24       # 펌핑을 찾을 창(1h봉)
    oi_pump_min: float = 0.5    # 창 안 저점 → 고점 +50% 이상
    oi_rise_min: float = 0.3    # 그동안 OI 최고가 시작 대비 +30% 이상
    oi_back_max: float = 0.1    # 지금 OI가 시작 대비 +10% 이내
    oi_keep_min: float = 0.6    # 가격은 상승분의 60% 이상 유지
    rearm_hours: int = 24


def basis_gap(px5: pd.DataFrame | None, idx5: pd.DataFrame | None, now: pd.Timestamp,
              params: Params = Params()) -> tuple[pd.Timestamp, float] | None:
    """최근 마감 5m봉 중 |퍼프 ÷ 인덱스 − 1|이 gap_min 이상인 봉 — (봉 시각, 갭). 가장 큰 것."""
    if px5 is None or idx5 is None or not len(px5) or not len(idx5):
        return None
    step = pd.Timedelta(minutes=5)
    c = px5["Close"][px5.index + step <= now]
    i = idx5["Close"][idx5.index + step <= now]
    gap = (c / i.reindex(c.index) - 1).dropna().iloc[-params.gap_bars:]
    if gap.empty:
        return None
    t = gap.abs().idxmax()
    g = float(gap[t])
    return (t, g) if abs(g) >= params.gap_min else None


def oi_unwind(k1h: pd.DataFrame | None, oi: pd.Series | None, now: pd.Timestamp,
              params: Params = Params()) -> dict | None:
    """OI 되돌림 — 마지막 마감 1h봉 기준. 조건이 다 서면 수치 dict, 아니면 None.

    OI는 정시 스냅샷이라 1h봉에는 **마감 시각**(시작 + 1h)의 값을 붙인다.
    """
    if k1h is None or oi is None or not len(k1h) or not len(oi):
        return None
    hour = pd.Timedelta(hours=1)
    k = k1h[k1h.index + hour <= now]
    o = oi[~oi.index.duplicated()].reindex(k.index + hour).to_numpy(float)
    n = params.oi_window_h
    if len(k) < n + 1 or np.isnan(o[-1]):
        return None
    lo = k["Low"].to_numpy(float)[-(n + 1):]
    hi = k["High"].to_numpy(float)[-(n + 1):]
    o = o[-(n + 1):]
    j0 = int(np.argmin(lo))                     # 창 안 저점 봉 = 펌핑 시작
    if j0 >= n - 1 or np.isnan(o[j0]) or o[j0] <= 0:
        return None
    p0, o0 = lo[j0], o[j0]
    top, opk = hi[j0:].max(), np.nanmax(o[j0:])
    c = float(k["Close"].iloc[-1])
    keep = (c - p0) / (top - p0) if top > p0 else 0.0
    if (top / p0 - 1 >= params.oi_pump_min and opk / o0 - 1 >= params.oi_rise_min
            and o[-1] / o0 - 1 <= params.oi_back_max and keep >= params.oi_keep_min):
        return {"pump": top / p0 - 1, "oi_peak": opk / o0, "oi_now": o[-1] / o0,
                "keep": keep, "start_price": p0}
    return None


def detect(symbol: str, now: pd.Timestamp, px5: pd.DataFrame | None,
           idx5: pd.DataFrame | None, k1h: pd.DataFrame | None, oi: pd.Series | None,
           params: Params = Params()) -> SignalEvent | None:
    """① 현선갭 · ② OI 되돌림 중 하나라도 서면 이벤트 하나. 둘 다면 둘 다 싣는다."""
    if not params.enabled:
        return None
    g = basis_gap(px5, idx5, now, params)
    u = oi_unwind(k1h, oi, now, params) if params.oi_enabled else None
    if g is None and u is None:
        return None
    detail = {"label": LABEL}
    if g is not None:
        bar_time = g[0]
        detail.update({"gap": g[1], "interval": GAP_INTERVAL})
    else:
        bar_time = k1h[k1h.index + pd.Timedelta(hours=1) <= now].index[-1]
        detail["interval"] = OI_PERIOD
    if u is not None:
        detail.update({f"oi_{k}" if not k.startswith("oi") else k: v for k, v in u.items()})
    if px5 is not None and len(px5):
        price = float(px5["Close"][px5.index + pd.Timedelta(minutes=5) <= now].iloc[-1])
    else:
        price = float(k1h["Close"].iloc[-1])
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=bar_time, price=price, detail=detail)
