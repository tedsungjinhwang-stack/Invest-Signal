"""바이낸스 USDT 무기한 선물 4h 캔들 수집.

주의: fapi.binance.com은 미국 IP(GitHub Actions 러너 포함)에서 HTTP 451로
차단된다. 그 경우 BINANCE_FAPI_BASE 환경변수로 프록시/미러 주소를 지정하거나
크립토 스캔을 미국 외 환경에서 돌려야 한다(README 참고).
"""

import os
import time

import pandas as pd
import requests

DEFAULT_BASE = "https://fapi.binance.com"
KLINE_INTERVAL = "4h"
BAR_MS = 4 * 3600 * 1000

# 분당 요청 가중치 한도(2400)에 여유를 두고 이 값을 넘으면 잠시 쉰다.
WEIGHT_SOFT_LIMIT = 1800


class GeoBlockedError(RuntimeError):
    """미국 등 제한 지역 IP에서 바이낸스가 451을 돌려줄 때."""


def _base() -> str:
    return os.environ.get("BINANCE_FAPI_BASE", DEFAULT_BASE).rstrip("/")


def _get(session: requests.Session, path: str, params: dict | None = None,
         tries: int = 3) -> requests.Response:
    url = f"{_base()}{path}"
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            rsp = session.get(url, params=params, timeout=15)
        except requests.RequestException:
            if attempt == tries:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if rsp.status_code == 451:
            raise GeoBlockedError(
                "바이낸스 선물 API가 이 IP(제한 지역)를 차단했습니다. "
                "BINANCE_FAPI_BASE로 프록시를 지정하거나 미국 외 환경에서 실행하세요.")
        if rsp.status_code in (418, 429):        # 레이트리밋 — 잠시 쉬고 재시도
            wait = int(rsp.headers.get("Retry-After", "30"))
            time.sleep(min(wait, 120))
            continue
        if rsp.status_code >= 500 and attempt < tries:
            time.sleep(delay)
            delay *= 2
            continue
        rsp.raise_for_status()
        _throttle(rsp)
        return rsp
    raise RuntimeError(f"바이낸스 요청 실패: {path}")


def _throttle(rsp: requests.Response) -> None:
    used = rsp.headers.get("X-MBX-USED-WEIGHT-1M")
    if used and used.isdigit() and int(used) >= WEIGHT_SOFT_LIMIT:
        time.sleep(25)


def usdt_perp_symbols(session: requests.Session,
                      exclude: set[str] | None = None) -> list[str]:
    """거래 중인 전체 USDT 무기한 심볼 목록."""
    info = _get(session, "/fapi/v1/exchangeInfo").json()
    exclude = exclude or set()
    out = []
    for s in info.get("symbols", []):
        if (s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING"
                and s.get("quoteAsset") == "USDT"
                and s["symbol"] not in exclude):
            out.append(s["symbol"])
    return sorted(out)


def parse_klines(rows: list, now_ms: int | None = None) -> pd.DataFrame:
    """kline 배열 → OHLCV DataFrame(UTC 인덱스). 진행 중인 마지막 봉은 버린다."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    keep = [r for r in rows if int(r[6]) <= now_ms]     # r[6] = closeTime(ms)
    if not keep:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(
        {
            "Open": [float(r[1]) for r in keep],
            "High": [float(r[2]) for r in keep],
            "Low": [float(r[3]) for r in keep],
            "Close": [float(r[4]) for r in keep],
            "Volume": [float(r[5]) for r in keep],
        },
        index=pd.to_datetime([int(r[0]) for r in keep], unit="ms", utc=True),
    )
    df.index.name = "OpenTime"
    return df


def klines_4h(session: requests.Session, symbol: str, limit: int = 600) -> pd.DataFrame:
    rows = _get(session, "/fapi/v1/klines",
                {"symbol": symbol, "interval": KLINE_INTERVAL, "limit": limit}).json()
    return parse_klines(rows)


def fetch_all(symbols: list[str], limit: int = 600,
              pause: float = 0.12, log=print) -> dict[str, pd.DataFrame]:
    """전 심볼 4h 캔들 수집. 개별 실패는 건너뛰고, 451은 즉시 중단."""
    out: dict[str, pd.DataFrame] = {}
    failed = []
    with requests.Session() as session:
        for i, sym in enumerate(symbols):
            try:
                out[sym] = klines_4h(session, sym, limit)
            except GeoBlockedError:
                raise
            except Exception as e:                     # noqa: BLE001 — 종목별 실패는 스캔 전체를 막지 않는다
                failed.append(sym)
                log(f"[binance] {sym} 수집 실패: {e}")
            if pause:
                time.sleep(pause)
            if (i + 1) % 100 == 0:
                log(f"[binance] {i + 1}/{len(symbols)} 수집")
    if failed:
        log(f"[binance] 실패 {len(failed)}종: {', '.join(failed[:10])}"
            + (" ..." if len(failed) > 10 else ""))
    return out
