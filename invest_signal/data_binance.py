"""바이낸스 USDT 4h 캔들 수집 — 선물(fapi) 우선, 미국 IP에선 현물 미러 폴백.

fapi.binance.com·api.binance.com은 미국 IP(GitHub Actions 러너 포함)에서
HTTP 451로 차단된다. 공식 현물 데이터 미러 data-api.binance.vision은
미국에서도 열려 있어서, source=auto(기본)면 fapi가 막혔을 때 현물 USDT
페어로 자동 전환한다(현물·퍼프 4h 가격은 MA 시그널 용도로는 사실상 동일).
진짜 퍼프 데이터가 필요하면 BINANCE_FAPI_BASE로 프록시를 지정하거나
미국 외 환경에서 실행하면 된다.
"""

import os
import time

import pandas as pd
import requests

FAPI_BASE = "https://fapi.binance.com"
SPOT_MIRROR_BASE = "https://data-api.binance.vision"
KLINE_INTERVAL = "4h"

# 분당 요청 가중치 한도(선물 2400)에 여유를 두고 이 값을 넘으면 잠시 쉰다.
WEIGHT_SOFT_LIMIT = 1800

# 현물 유니버스에서 뺄 스테이블코인(스테이블/스테이블 페어는 시그널 무의미)
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "EURI", "AEUR",
                "USDE", "USD1", "XUSD", "BUSD", "PAXG"}


class GeoBlockedError(RuntimeError):
    """미국 등 제한 지역 IP에서 바이낸스가 451을 돌려줄 때."""


def fapi_base() -> str:
    return os.environ.get("BINANCE_FAPI_BASE", FAPI_BASE).rstrip("/")


def _get(session: requests.Session, base: str, path: str,
         params: dict | None = None, tries: int = 3) -> requests.Response:
    url = f"{base}{path}"
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            rsp = session.get(url, params=params, timeout=20)
        except requests.RequestException:
            if attempt == tries:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if rsp.status_code == 451:
            raise GeoBlockedError(
                "바이낸스 API가 이 IP(제한 지역)를 차단했습니다(HTTP 451).")
        if rsp.status_code in (418, 429):        # 레이트리밋 — 쉬고 재시도
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
    """거래 중인 전체 USDT 무기한 심볼 (fapi)."""
    info = _get(session, fapi_base(), "/fapi/v1/exchangeInfo").json()
    exclude = exclude or set()
    out = [s["symbol"] for s in info.get("symbols", [])
           if s.get("contractType") == "PERPETUAL"
           and s.get("status") == "TRADING"
           and s.get("quoteAsset") == "USDT"
           and s["symbol"] not in exclude]
    return sorted(out)


def usdt_spot_symbols(session: requests.Session,
                      exclude: set[str] | None = None) -> list[str]:
    """거래 중인 전체 USDT 현물 페어 (vision 미러) — 스테이블 베이스 제외."""
    info = _get(session, SPOT_MIRROR_BASE, "/api/v3/exchangeInfo").json()
    exclude = exclude or set()
    out = [s["symbol"] for s in info.get("symbols", [])
           if s.get("status") == "TRADING"
           and s.get("quoteAsset") == "USDT"
           and s.get("isSpotTradingAllowed", True)
           and s.get("baseAsset") not in STABLE_BASES
           and s["symbol"] not in exclude]
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


def resolve_source(session: requests.Session, requested: str,
                   exclude: set[str], log=print) -> tuple[str, list[str]]:
    """설정된 source(auto|fapi|spot_mirror)를 실제 소스+심볼 목록으로 확정."""
    if requested in ("auto", "fapi"):
        try:
            syms = usdt_perp_symbols(session, exclude)
            return "fapi", syms
        except GeoBlockedError:
            if requested == "fapi":
                raise
            log("[binance] fapi 451 차단 — 현물 미러(data-api.binance.vision)로 폴백")
    return "spot_mirror", usdt_spot_symbols(session, exclude)


def klines_4h(session: requests.Session, symbol: str, source: str,
              limit: int = 600) -> pd.DataFrame:
    if source == "fapi":
        base, path = fapi_base(), "/fapi/v1/klines"
    else:
        base, path = SPOT_MIRROR_BASE, "/api/v3/klines"
    rows = _get(session, base, path,
                {"symbol": symbol, "interval": KLINE_INTERVAL, "limit": limit}).json()
    return parse_klines(rows)


def fetch_all(session: requests.Session, symbols: list[str], source: str,
              limit: int = 600, pause: float = 0.1,
              log=print) -> dict[str, pd.DataFrame]:
    """전 심볼 4h 캔들 수집. 개별 실패는 건너뛰고, 451은 즉시 중단."""
    out: dict[str, pd.DataFrame] = {}
    failed = []
    for i, sym in enumerate(symbols):
        try:
            out[sym] = klines_4h(session, sym, source, limit)
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
