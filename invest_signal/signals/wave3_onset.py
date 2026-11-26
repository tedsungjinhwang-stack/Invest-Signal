"""3️⃣ 3파 초입 (1h봉, 크립토 전용, 11-26~).

1h 120 < 240 < 480 **역배열**에서 종가가 240선을 **상향 돌파**한 뒤, 48시간 안에

① 월 VWAP +5% 위로 마감했다가(멀어졌다가) 다시 내려와 VWAP ±5% 안으로 마감한 봉 — 이 봉에서 알린다
② 돌파 봉부터 그 봉까지 사이에 1h 거래량이 직전 24봉 평균의 5배 이상인 봉이 하나 이상
③ 그 사이 돌파 전 저점(돌파 봉까지 48봉 최저가)을 깨지 않았다

24h 거래대금 하한은 스캐너가 티커로 건다($2M). 돌파 하나에 한 번만 나간다(①의 첫 봉).

퍼프 521종 08-13~11-26(105일, 24h $2M+) 백테스트: 하루 ≈3.6건 · 7일 안 +20%가 −10%보다
먼저 43%(아무 1h봉 27%) · 앞/뒤 반기 57%/29%(아무 봉 41%/13%) · 7일 안 +50% 13.4%(5.5%).
피보 38.2% 되돌림 조건은 효과가 없어(30%) 뺐고, 눌림 거래량 감소도 뒤 반기에 효과가 없어 뺐다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import anchored_vwap_bands
from . import SignalEvent

NAME = "wave3_onset"
LABEL = "3파초입"
INTERVAL = "1h"
KLINE_LIMIT = 1000


@dataclass(frozen=True)
class Params:
    ma: tuple = (120, 240, 480)          # 역배열 판정선 — 돌파는 가운데 선(240)
    window: int = 48                     # 돌파 뒤 이 봉 수 안에 VWAP 복귀가 나와야 한다
    pre_low_bars: int = 48               # 돌파 전 저점 — 돌파 봉까지 이 봉 수의 최저가
    vwap_anchor: str = "M"
    vwap_away: float = 0.05              # ① 먼저 VWAP × (1+이 값) 위로 마감
    vwap_near: float = 0.05              # ① 그 뒤 |종가 ÷ VWAP − 1| ≤ 이 값
    vol_mult: float = 5.0                # ② 1h 거래량 ÷ 직전 vol_ma봉 평균
    vol_ma: int = 24
    grace_bars: int = 1                  # 스캔 1시간 — 직전 마감봉 + 한 봉 소급
    min_turnover_usd: float = 2_000_000


def _find(df: pd.DataFrame, p: Params) -> list[tuple[int, int, float, float]]:
    """(돌파 봉, 신호 봉, 돌파 뒤 고점 상승률, 최대 거래량 배수) — 오래된 것부터."""
    n = len(df)
    lo_ma, mid_ma, hi_ma = (int(x) for x in p.ma)
    if n < hi_ma + 2:
        return []
    got = anchored_vwap_bands(df, p.vwap_anchor, 1.0)
    if got is None:
        return []
    vw = got[0].to_numpy(float)
    c, h, l, v = (df[k].to_numpy(float) for k in ("Close", "High", "Low", "Volume"))
    close = df["Close"]
    m1, m2, m4 = (close.rolling(x).mean().to_numpy(float) for x in (lo_ma, mid_ma, hi_ma))
    vr = (df["Volume"] / df["Volume"].shift(1).rolling(p.vol_ma).mean()).to_numpy(float)
    out = []
    # 신호가 마지막 window+grace 봉 안에 나올 수 있는 돌파만 본다
    for b in range(max(hi_ma, n - p.window - p.grace_bars - 2), n):
        if not (c[b] > m2[b] and c[b - 1] <= m2[b - 1] and m1[b] < m2[b] < m4[b]):
            continue
        low0 = np.nanmin(l[max(0, b - p.pre_low_bars):b + 1])
        away, hi = False, h[b]
        for j in range(b + 1, min(n, b + p.window + 1)):
            if l[j] <= low0 or np.isnan(vw[j]):
                break
            hi = max(hi, h[j])
            if c[j] > vw[j] * (1 + p.vwap_away):
                away = True
                continue
            if away and abs(c[j] / vw[j] - 1) <= p.vwap_near:
                vmax = np.nanmax(vr[b:j + 1])
                if vmax >= p.vol_mult:
                    out.append((b, j, hi / c[b] - 1, float(vmax)))
                break                    # 돌파 하나에 첫 복귀 봉 한 번만
    return out


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 1h봉(오름차순, UTC)에서 마지막 grace_bars+1봉 안의 신호 봉."""
    n = len(df)
    out, seen = [], set()
    for b, j, rise, vmax in _find(df, params):
        # 240선을 두 번 뚫었다 내려와 같은 봉에서 복귀하면 한 줄만 — 먼저 뚫은 쪽으로 적는다(11-26 MOVE)
        if j < n - 1 - params.grace_bars or j in seen:
            continue
        seen.add(j)
        vw = anchored_vwap_bands(df, params.vwap_anchor, 1.0)[0]
        out.append(SignalEvent(
            symbol=symbol, signal=NAME, bar_time=df.index[j], price=float(df["Close"].iloc[j]),
            detail={"label": LABEL, "interval": INTERVAL,
                    "break_time": df.index[b].isoformat(),
                    "rise_after": float(rise), "vol_mult": vmax,
                    "vwap_dist": float(df["Close"].iloc[j] / vw.iloc[j] - 1)}))
    return out
