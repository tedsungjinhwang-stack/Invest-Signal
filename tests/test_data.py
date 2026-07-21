"""데이터 레이어 테스트 — 캔들 파싱과 4h 리샘플(네트워크 없음)."""

import pandas as pd

from invest_signal.data_binance import parse_klines
from invest_signal.data_etf import resample_4h

H = 3600 * 1000


def _row(open_ms, o, h, l, c, v):
    # 바이낸스 kline 형식: [openTime, o, h, l, c, v, closeTime, ...]
    return [open_ms, str(o), str(h), str(l), str(c), str(v), open_ms + 4 * H - 1]


def test_parse_klines_drops_in_progress_bar():
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % (4 * H))
    rows = [_row(t0, 1, 2, 0.5, 1.5, 100),
            _row(t0 + 4 * H, 1.5, 3, 1, 2, 200)]
    now = t0 + 4 * H + 1000            # 두 번째 봉은 아직 진행 중
    df = parse_klines(rows, now_ms=now)
    assert len(df) == 1
    assert df["Close"].iloc[0] == 1.5
    assert df.index.tz is not None

    df2 = parse_klines(rows, now_ms=t0 + 8 * H)   # 둘 다 마감
    assert len(df2) == 2 and df2["Volume"].iloc[1] == 200


def test_resample_4h_aggregates_and_converts_tz():
    idx = pd.date_range("2025-06-02 09:30", periods=6, freq="1h",
                        tz="America/New_York")
    df = pd.DataFrame({
        "Open": [10, 11, 12, 13, 14, 15],
        "High": [11, 12, 13, 14, 15, 16],
        "Low": [9, 10, 11, 12, 13, 14],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        "Volume": [1, 1, 1, 1, 1, 1],
    }, index=idx, dtype=float)
    now = pd.Timestamp("2025-06-03", tz="UTC")
    out = resample_4h(df, now=now)
    # 09:30~15:30 ET = 13:30~19:30 UTC → 12:00, 16:00 두 버킷
    assert list(out.index.hour) == [12, 16]
    b0 = out.iloc[0]                   # 13:30·14:30·15:30 UTC 세 봉 합성
    assert b0["Open"] == 10 and b0["High"] == 13 and b0["Low"] == 9
    assert b0["Close"] == 12.5 and b0["Volume"] == 3


def test_resample_4h_drops_incomplete_bucket():
    idx = pd.date_range("2025-06-02 00:00", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                       "Volume": 1.0}, index=idx)
    # 05:00 시점 — 04:00 버킷은 아직 진행 중이므로 00:00 버킷만 남아야 함
    out = resample_4h(df, now=pd.Timestamp("2025-06-02 05:00", tz="UTC"))
    assert list(out.index.hour) == [0]
