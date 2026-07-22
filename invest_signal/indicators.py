"""공용 지표 계산."""

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """단순이동평균. 데이터가 n개 미만인 구간은 NaN."""
    return close.rolling(n).mean()


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


def quarterly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """분기 앵커드 VWAP (1/4/7/10월 1일 리셋)."""
    return anchored_vwap(df, "Q")


def monthly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """월간 앵커드 VWAP (매월 1일 리셋)."""
    return anchored_vwap(df, "M")
