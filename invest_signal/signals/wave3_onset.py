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

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..indicators import anchored_vwap_bands
from . import SignalEvent

NAME = "wave3_onset"
LABEL = "3파초입"
CONFIRM_NAME = "wave3_confirm"
CONFIRM_LABEL = "3파확정"
GOOD = "🎯2.618후보"
WEAK = "⚠️1배형"
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
    # ① 또는(11-26~) — 240선(ma 가운데)에서도 같은 식: 240선 × (1+ma_away) 위로 마감했다가 ±ma_near 안으로.
    # ma_near가 None이면 VWAP만 본다.
    ma_away: float = 0.05
    ma_near: float | None = None         # 11-26 백테스트: 240선 복귀는 ±3% 25% · ±5% 32%(VWAP 43%) — 기본 끔
    vol_mult: float = 5.0                # ② 1h 거래량 ÷ 직전 vol_ma봉 평균
    vol_ma: int = 24
    grace_bars: int = 1                  # 스캔 1시간 — 직전 마감봉 + 한 봉 소급
    min_turnover_usd: float = 2_000_000
    # 3파 확정(11-27~) — 알림 뒤 1파 고점을 넘는 봉. 1파 끝은 돌파 뒤 고점에서 1파의 23.6% 이상 밀린
    # 시점의 그 고점, 2파 저점은 그 뒤 1파 고점을 넘기 전까지 최저가(엘리엇: 3파는 1파 끝을 넘는다).
    confirm_enabled: bool = True
    confirm_lookback: int = 336          # 돌파 뒤 이 봉 수(14일) 안의 확정만 본다
    w1_retrace: float = 0.236


def _find(df: pd.DataFrame, p: Params) -> list[tuple[int, int, float, float, str]]:
    """(돌파 봉, 신호 봉, 돌파 뒤 고점 상승률, 최대 거래량 배수, 복귀한 선) — 오래된 것부터."""
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
        away_v = away_m = False
        hi = h[b]
        for j in range(b + 1, min(n, b + p.window + 1)):
            if l[j] <= low0:
                break
            hi = max(hi, h[j])
            hit = None
            if not np.isnan(vw[j]):
                if away_v and abs(c[j] / vw[j] - 1) <= p.vwap_near:
                    hit = "월VWAP"
                away_v = away_v or c[j] > vw[j] * (1 + p.vwap_away)
            if hit is None and p.ma_near is not None and not np.isnan(m2[j]):
                if away_m and abs(c[j] / m2[j] - 1) <= p.ma_near:
                    hit = f"{mid_ma}선"
                away_m = away_m or c[j] > m2[j] * (1 + p.ma_away)
            if hit is not None:
                vmax = np.nanmax(vr[b:j + 1])
                if vmax >= p.vol_mult:
                    out.append((b, j, hi / c[b] - 1, float(vmax), hit))
                break                    # 돌파 하나에 첫 복귀 봉 한 번만
    return out


def alert_grade(retr: float, d480: float, g7d: float) -> str | None:
    """3파 초입 알림 시점 등급(11-27 · 521종 105일 학습/검증) — 🎯 되돌림 30% 미만 + 1h 480선 +2.7% 위
    (2.618배+ 검증 27%, 평균 10%) · ⚠️ 480선 +1.5% 미만 + 되돌림 30%+ + 7일 등락 +3.9% 미만(81% 1배 미만)."""
    if retr < 0.30 and d480 >= 0.027:
        return GOOD
    if d480 < 0.015 and retr >= 0.30 and g7d < 0.039:
        return WEAK
    return None


def confirm_grade(w1: float, fib: float, d480_w2: float, from30h: float, d240_w2: float,
                  v2_v1: float) -> str | None:
    """3파 확정 시점 등급(11-27 · 2파 완성 505건 학습/검증) — ⚠️ 1파 +12%+ · 되돌림 33% 미만 · 2파 저점이
    480선 위(1배 미만 79%/83%) · 🎯 1파 고점이 30일 고점 −14% 안 · 2파 저점이 240선(=15m 960선) +2.4% 안 ·
    2파 거래량 < 1파(2.618배+ 학습 61% → 검증 27%, 평균 34%/10%)."""
    if w1 >= 0.12 and fib < 0.33 and d480_w2 >= 0.001:
        return WEAK
    if from30h >= -0.14 and d240_w2 < 0.024 and v2_v1 < 0.92:
        return GOOD
    return None


def _waves(df: pd.DataFrame, b: int, p: Params):
    """돌파 봉 b부터 (1파 시작가, 1파 끝 위치, 2파 저점 위치, 3파 확정 위치) — 못 세면 None."""
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    n = len(df)
    w0 = float(np.nanmin(l[max(0, b - p.pre_low_bars):b + 1]))
    hi, hii, w1i = h[b], b, None
    for k in range(b + 1, min(n, b + 120)):
        if h[k] > hi:
            hi, hii = h[k], k
        if l[k] <= w0:
            return None
        if l[k] <= hi - p.w1_retrace * (hi - w0) and hi / w0 - 1 >= 0.05:
            w1i = hii
            break
    if w1i is None:
        return None
    lo, loi = np.inf, None
    for k in range(w1i + 1, min(n, b + p.confirm_lookback)):
        if l[k] <= w0:
            return None
        if l[k] < lo:
            lo, loi = l[k], k
        if h[k] > h[w1i]:
            return w0, w1i, loi, k
    return None


def confirmations(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """3️⃣ 3파 확정 — 3파 초입이 나간 돌파가 1파 고점을 넘은 봉(마지막 grace_bars+1봉 안)."""
    if not params.confirm_enabled:
        return []
    n = len(df)
    lookback = params.confirm_lookback + params.window + 2
    p2 = replace(params, grace_bars=lookback)          # 지난 3파 초입 알림까지 다시 찾는다
    c, h, v = (df[k].to_numpy(float) for k in ("Close", "High", "Volume"))
    close = df["Close"]
    m240 = close.rolling(int(params.ma[1])).mean().to_numpy(float)
    m480 = close.rolling(int(params.ma[2])).mean().to_numpy(float)
    vw = anchored_vwap_bands(df, params.vwap_anchor, 1.0)[0].to_numpy(float)
    out, seen = [], set()
    for b, j, rise, vmax, where in _find(df, p2):
        if b in seen:
            continue
        seen.add(b)
        got = _waves(df, b, params)
        if got is None:
            continue
        w0, w1i, w2i, ov = got
        if ov <= j or ov < n - 1 - params.grace_bars:
            continue                     # 3파 초입 알림 뒤의 확정만 · 지나간 확정은 안 낸다
        w1, w2 = h[w1i], df["Low"].iloc[w2i]
        leg = w1 - w0
        v1 = v[b:w1i + 1].mean()
        v2 = v[w1i + 1:w2i + 1].mean() if w2i > w1i else np.nan
        d = {"label": CONFIRM_LABEL, "interval": INTERVAL, "break_time": df.index[b].isoformat(),
             "wave1": float(w1 / w0 - 1), "retrace": float((w1 - w2) / leg),
             "w2_low": float(w2), "w2_vs_240": float(w2 / m240[w2i] - 1),
             "w2_vs_vwap": float(w2 / vw[w2i] - 1) if not np.isnan(vw[w2i]) else None,
             "targets": [float(w2 + k * leg) for k in (1.0, 1.618, 2.618)]}
        d["grade"] = confirm_grade(d["wave1"], d["retrace"], float(w2 / m480[w2i] - 1),
                                   float(w1 / np.nanmax(h[max(0, b - 720):b]) - 1) if b > 0 else -1.0,
                                   d["w2_vs_240"], float(v2 / v1) if v1 > 0 else np.nan)
        out.append(SignalEvent(symbol=symbol, signal=CONFIRM_NAME, bar_time=df.index[ov],
                               price=float(c[ov]), detail=d))
    return out


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 1h봉(오름차순, UTC)에서 마지막 grace_bars+1봉 안의 신호 봉."""
    n = len(df)
    out, seen = [], set()
    for b, j, rise, vmax, where in _find(df, params):
        # 240선을 두 번 뚫었다 내려와 같은 봉에서 복귀하면 한 줄만 — 먼저 뚫은 쪽으로 적는다(11-26 MOVE)
        if j < n - 1 - params.grace_bars or j in seen:
            continue
        seen.add(j)
        vw = anchored_vwap_bands(df, params.vwap_anchor, 1.0)[0]
        lo_, hi_ = df["Low"].to_numpy(float), df["High"].to_numpy(float)
        w0 = float(np.nanmin(lo_[max(0, b - params.pre_low_bars):b + 1]))
        h1 = float(np.nanmax(hi_[b:j + 1]))
        cj = float(df["Close"].iloc[j])
        retr = (h1 - cj) / (h1 - w0) if h1 > w0 else np.nan
        d480 = cj / float(df["Close"].rolling(int(params.ma[2])).mean().iloc[j]) - 1
        g7d = cj / float(df["Close"].iloc[max(0, j - 168)]) - 1
        out.append(SignalEvent(
            symbol=symbol, signal=NAME, bar_time=df.index[j], price=float(df["Close"].iloc[j]),
            detail={"label": LABEL, "interval": INTERVAL,
                    "break_time": df.index[b].isoformat(),
                    "rise_after": float(rise), "vol_mult": vmax, "near": where,
                    "ma_dist": float(df["Close"].iloc[j] / df["Close"].rolling(int(params.ma[1])).mean().iloc[j] - 1),
                    "vwap_dist": float(df["Close"].iloc[j] / vw.iloc[j] - 1),
                    "retrace": float(retr), "ma480_dist": float(d480), "ret_7d": float(g7d),
                    "grade": alert_grade(retr, d480, g7d)}))
    return out
