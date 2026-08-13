"""레버리지 ETF 4h 캔들 수집 — yfinance 1h봉을 UTC 기준 4h봉으로 합성."""

import pandas as pd

# Yahoo 1h봉은 최근 730일까지만 제공. period="720d"를 쓰면 상장 845일차 같은
# 종목에서 yfinance가 상장일을 시작점으로 보내 거부당하므로(BITU·ETHU·AMDL
# 실측) 명시적 start 날짜로 요청한다.
LOOKBACK_DAYS = 700


def yahoo_symbol(code: str, market: str) -> str:
    """야후 심볼 — KR(코스피)은 .KS, KQ(코스닥)은 .KQ, 미국은 그대로."""
    if market == "KR":
        return f"{code}.KS"
    if market == "KQ":
        return f"{code}.KQ"
    return code


def fill_missing_kr_names(tickers: list[dict], log=print) -> None:
    """이름이 빈 한국 종목의 종목명을 야후에서 보충한다 (제자리 수정).

    퀀트포트폴리오 리포트가 이름을 못 찾으면 '코드(코드)'로 내보내는
    경우가 있어(예: 010120(010120)), 알림에 코드만 찍히는 것을 막는다.
    조회 실패는 무시 — 코드 표시로 폴백.
    """
    import yfinance as yf
    for t in tickers:
        if t.get("name") or not str(t.get("code", ""))[:1].isdigit():
            continue
        for suffix in (".KS", ".KQ"):
            try:
                info = yf.Ticker(f"{t['code']}{suffix}").get_info()
                name = (info.get("shortName") or info.get("longName") or "").strip()
            except Exception:       # noqa: BLE001 — 이름 보충 실패가 스캔을 막지 않게
                continue
            if name:
                t["name"] = name
                log(f"[stocks] {t['code']} 종목명 보충: {name}")
                break


def resample_4h(df1h: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """1h OHLC → UTC 4h봉. 미확정 데이터가 섞일 수 있는 마지막 버킷은 버린다.

    미국 종목의 Yahoo 1h봉은 :30 기준(13:30, 14:30… UTC)이라 4h 버킷 경계를
    30분 넘어간다. 진행 중인 1h봉을 먼저 제거하고, 그 봉이 속했던 버킷과
    4시간이 다 지나지 않은 버킷을 끝에서 잘라내 확정된 봉만 남긴다.
    """
    if df1h is None or len(df1h) == 0:
        return None
    d = df1h.dropna(subset=["Close"]).copy()
    if len(d) == 0:
        return None
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        d.index = d.index.tz_convert("UTC")
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    bar_end = d.index + pd.Timedelta(hours=1)
    live_buckets = set(d.index[bar_end > now].floor("4h"))   # 진행 중 1h봉이 속한 버킷
    d = d[bar_end <= now]
    if len(d) == 0:
        return None
    agg = {
        "Open": d["Open"].resample("4h").first(),
        "High": d["High"].resample("4h").max(),
        "Low": d["Low"].resample("4h").min(),
        "Close": d["Close"].resample("4h").last(),
    }
    if "Volume" in d.columns:
        agg["Volume"] = d["Volume"].resample("4h").sum()
    out = pd.DataFrame(agg).dropna(subset=["Open", "High", "Low", "Close"])
    while len(out) and (out.index[-1] + pd.Timedelta(hours=4) > now
                        or out.index[-1] in live_buckets):
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


def _download(syms: list[str]):
    import yfinance as yf     # 무거운 임포트 — 테스트에서 모듈 로드만 할 땐 안 불리게 지연

    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return yf.download(syms, start=start, interval="1h", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)


def fetch_all(tickers: list[dict], log=print) -> tuple[dict, dict]:
    """티커 목록 전체를 받아 ({코드: 4h OHLC}, {코드: 1h OHLC})를 돌려준다.

    원본이 1h봉이라 4h는 리샘플 결과다. 1h 프레임은 알림에 붙일 1시간
    수익률을 계산하는 데만 쓰므로 시그널 판정에는 들어가지 않는다.

    KR(.KS)로 조회했는데 데이터가 없는 6자리 코드는 코스닥(.KQ)으로
    재시도한다 — 퀀트포트폴리오 리포트에는 코스피/코스닥 구분이 없어서.
    """
    data = _download([yahoo_symbol(t["code"], t["market"]) for t in tickers])
    out: dict[str, pd.DataFrame] = {}
    hourly: dict[str, pd.DataFrame] = {}
    retry_kq = []
    for t in tickers:
        sym = yahoo_symbol(t["code"], t["market"])
        df1h = extract_ohlc(data, sym)
        d4 = resample_4h(df1h) if df1h is not None else None
        if d4 is not None and len(d4):
            out[t["code"]] = d4
            hourly[t["code"]] = df1h
        elif t["market"] == "KR":
            retry_kq.append(t)
        else:
            log(f"[etf] {t['code']} 데이터 없음/부족 — 건너뜀")
    if retry_kq:
        data2 = _download([f"{t['code']}.KQ" for t in retry_kq])
        for t in retry_kq:
            df1h = extract_ohlc(data2, f"{t['code']}.KQ")
            d4 = resample_4h(df1h) if df1h is not None else None
            if d4 is not None and len(d4):
                out[t["code"]] = d4
                hourly[t["code"]] = df1h
            else:
                log(f"[etf] {t['code']} 데이터 없음(KS/KQ 모두) — 건너뜀")
    return out, hourly
