"""공용 지표 계산."""

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """단순이동평균. 데이터가 n개 미만인 구간은 NaN."""
    return close.rolling(n).mean()


def quarterly_vwap(df: pd.DataFrame) -> pd.Series | None:
    """분기 시작(UTC 1/4/7/10월 1일) 앵커드 VWAP.

    typical price(H+L+C)/3 × 거래량을 분기 내 누적해 계산한다.
    Volume 컬럼이 없거나 전부 0이면 None.
    """
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].fillna(0.0)
    if not (vol > 0).any():
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
    q = idx.to_period("Q")
    pv = (tp * vol).groupby(q).cumsum()
    cv = vol.groupby(q).cumsum()
    return pv / cv.replace(0, np.nan)
