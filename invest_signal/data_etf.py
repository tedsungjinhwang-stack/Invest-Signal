"""레버리지 ETF 4h 캔들 수집 — yfinance 1h봉을 UTC 기준 4h봉으로 합성."""

import pandas as pd

# Yahoo 1h봉은 최근 730일까지만 제공. period="720d"를 쓰면 상장 845일차 같은
# 종목에서 yfinance가 상장일을 시작점으로 보내 거부당하므로(BITU·ETHU·AMDL
# 실측) 명시적 start 날짜로 요청한다.
LOOKBACK_DAYS = 700


def yahoo_symbol(code: str, market: str) -> str:
    return f"{code}.KS" if market == "KR" else code


def resample_4h(df1h: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """1h OHLC → UTC 4h봉. 아직 진행 중인 마지막 버킷은 버린다."""
    if df1h is None or len(df1h) == 0:
        return None
    d = df1h.dropna(subset=["Close"]).copy()
    if len(d) == 0:
        return None
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        d.index = d.index.tz_convert("UTC")
    agg = {
        "Open": d["Open"].resample("4h").first(),
        "High": d["High"].resample("4h").max(),
        "Low": d["Low"].resample("4h").min(),
        "Close": d["Close"].resample("4h").last(),
    }
    if "Volume" in d.columns:
        agg["Volume"] = d["Volume"].resample("4h").sum()
    out = pd.DataFrame(agg).dropna(subset=["Open", "High", "Low", "Close"])
    if len(out) == 0:
        return None
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    # 마지막 버킷의 4시간 구간이 아직 안 끝났으면 미확정 봉이므로 제외
    if out.index[-1] + pd.Timedelta(hours=4) > now:
        out = out.iloc[:-1]
    return out if len(out) else None


def extract_ohlc(data: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    """yf.download(group_by='ticker') 결과에서 심볼 하나의 OHLC를 꺼낸다."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if sym not in data.columns.get_level_values(0):
                return None
            df = data[sym]
        else:
            df = data
        cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
        df = df[cols]
    except (KeyError, TypeError):
        return None
    df = df.dropna(subset=["Close"])
    return df if len(df) else None


def fetch_all(tickers: list[dict], log=print) -> dict[str, pd.DataFrame]:
    """config의 ETF 목록 전체를 받아 {코드: 4h OHLC}를 돌려준다."""
    import yfinance as yf     # 무거운 임포트 — 테스트에서 모듈 로드만 할 땐 안 불리게 지연

    syms = [yahoo_symbol(t["code"], t["market"]) for t in tickers]
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    data = yf.download(syms, start=start, interval="1h", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        sym = yahoo_symbol(t["code"], t["market"])
        df1h = extract_ohlc(data, sym)
        d4 = resample_4h(df1h) if df1h is not None else None
        if d4 is None or len(d4) == 0:
            log(f"[etf] {t['code']} 데이터 없음/부족 — 건너뜀")
            continue
        out[t["code"]] = d4
    return out
