"""공용 지표 계산."""

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """단순이동평균. 데이터가 n개 미만인 구간은 NaN."""
    return close.rolling(n).mean()


def alignment(df: pd.DataFrame, periods: tuple = (120, 240, 480)) -> str | None:
    """마지막 봉의 이동평균 배열 상태 — "역배열"/"정배열"/"혼조".

    짧은 선부터 순서대로 커지면 역배열(하락 구조), 작아지면 정배열(상승 구조).
    데이터가 모자라 최장 MA가 NaN이면 None.
    """
    c = df["Close"]
    vals = [c.rolling(k).mean().iloc[-1] for k in periods]
    if any(pd.isna(v) for v in vals):
        return None
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "역배열"
    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "정배열"
    return "혼조"


def anchored_vwap(df: pd.DataFrame, period: str) -> pd.Series | None:
    """앵커드 VWAP — 기간 시작(UTC)마다 리셋. period: "Q"(분기) 또는 "M"(월).

    typical price(H+L+C)/3 × 거래량을 기간 내 누적해 계산한다.
    Volume 컬럼이 없거나 전부 0이면 None.
    """
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].fillna(0.0)
    if not (vol > 0).any():
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
    p = idx.to_period(period)
    pv = (tp * vol).groupby(p).cumsum()
    cv = vol.groupby(p).cumsum()
    return pv / cv.replace(0, np.nan)


def anchored_vwap_bands(df: pd.DataFrame, period: str, mult: float = 1.0):
    """앵커드 VWAP과 표준편차 밴드 — (vwap, lower, upper) 또는 None.

    TradingView의 Anchored VWAP 밴드(Standard Deviation 모드)와 같은 식:
      분산 = Σ(Vol×TP²)/Σ(Vol) − VWAP²,  밴드 = VWAP ± mult × √분산
    기간 시작(UTC)마다 리셋되며 Volume이 없으면 None.
    """
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].fillna(0.0)
    if not (vol > 0).any():
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
    p = idx.to_period(period)
    cv = vol.groupby(p).cumsum().replace(0, np.nan)
    vwap = (tp * vol).groupby(p).cumsum() / cv
    var = (tp ** 2 * vol).groupby(p).cumsum() / cv - vwap ** 2
    sd = np.sqrt(var.clip(lower=0))
    return vwap, vwap - mult * sd, vwap + mult * sd


def quarterly_vwap_bands(df: pd.DataFrame, mult: float = 1.0):
    """분기 앵커드 VWAP 밴드 — (vwap, lower, upper)."""
    return anchored_vwap_bands(df, "Q", mult)


def quarterly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """분기 앵커드 VWAP (1/4/7/10월 1일 리셋)."""
    return anchored_vwap(df, "Q")


def monthly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """월간 앵커드 VWAP (매월 1일 리셋)."""
    return anchored_vwap(df, "M")


def weekly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """주간 앵커드 VWAP (매주 월요일 00:00 UTC 리셋)."""
    return anchored_vwap(df, "W")


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range — max(고−저, |고−전일종가|, |저−전일종가|)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder 평활(TradingView ta.rma) — 첫 값은 n개 단순평균으로 시드.

    pandas의 ewm(adjust=False)은 첫 값 하나로 시드하는 게 달라서, 초반
    수십 봉의 값이 TradingView와 어긋난다. 시드를 맞춰 직접 돌린다.
    """
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < n:
        return pd.Series(out, index=s.index)
    seed = v[:n]
    if np.isnan(seed).any():            # 초반 NaN(첫 TR 등)은 빼고 시드
        start = int(np.argmax(~np.isnan(v)))
        if start + n > len(v):
            return pd.Series(out, index=s.index)
        seed, first = v[start:start + n], start + n - 1
    else:
        first = n - 1
    prev = float(np.mean(seed))
    out[first] = prev
    for i in range(first + 1, len(v)):
        x = v[i]
        prev = prev if np.isnan(x) else (prev * (n - 1) + x) / n
        out[i] = prev
    return pd.Series(out, index=s.index)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """평균 진폭(ATR) — Wilder 평활. TradingView ta.atr과 같은 값."""
    return rma(true_range(df), n)


def supertrend(df: pd.DataFrame, period: int = 22,
               mult: float = 3.0) -> pd.Series:
    """수퍼트렌드 방향 — +1 상승추세, −1 하락추세, ATR 미성립 구간은 NaN.

    TradingView 내장 `ta.supertrend(mult, period)`와 같은 정의다:
      기준선 = (고+저)/2, 밴드 = 기준선 ∓ mult×ATR
      상단/하단 밴드는 추세가 유지되는 동안 유리한 쪽으로만 따라 올라간다
      (트레일링 스톱). 종가가 반대편 밴드를 넘기면 방향이 뒤집힌다.
    """
    a = atr(df, period).to_numpy(dtype=float)
    hl2 = ((df["High"] + df["Low"]) / 2).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    dirs = np.full(n, np.nan)
    lower = upper = np.nan
    trend = 1                       # 파인 기본값과 동일하게 상승에서 시작
    for i in range(n):
        if np.isnan(a[i]):
            continue
        lo, up = hl2[i] - mult * a[i], hl2[i] + mult * a[i]
        # 직전 종가가 밴드 바깥이면 밴드를 끌어올린다/내린다 (트레일링)
        if not np.isnan(lower) and close[i - 1] > lower:
            lo = max(lo, lower)
        if not np.isnan(upper) and close[i - 1] < upper:
            up = min(up, upper)
        if not np.isnan(upper) and trend == -1 and close[i] > upper:
            trend = 1
        elif not np.isnan(lower) and trend == 1 and close[i] < lower:
            trend = -1
        lower, upper = lo, up
        dirs[i] = trend
    return pd.Series(dirs, index=df.index)
