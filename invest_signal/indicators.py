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
