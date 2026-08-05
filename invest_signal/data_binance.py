"""바이낸스 USDT 4h 캔들 수집 — 선물(fapi) 우선, 미국 IP에선 현물 미러 폴백.

fapi.binance.com·api.binance.com은 미국 IP(GitHub Actions 러너 포함)에서
HTTP 451로 차단된다. 공식 현물 데이터 미러 data-api.binance.vision은
미국에서도 열려 있어서, source=auto(기본)면 fapi가 막혔을 때 현물 USDT
페어로 자동 전환한다(현물·퍼프 4h 가격은 MA 시그널 용도로는 사실상 동일).
진짜 퍼프 데이터가 필요하면 BINANCE_FAPI_BASE로 프록시를 지정하거나
미국 외 환경에서 실행하면 된다.
"""

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

FAPI_BASE = "https://fapi.binance.com"
SPOT_MIRROR_BASE = "https://data-api.binance.vision"
# 바이낸스 공식 과거 데이터 저장소(S3) — 미국 IP에서도 접근 가능.
# USDT-M 선물 심볼 폴더 목록으로 '퍼프 상장 여부'를 판별하는 데 쓴다.
DATA_VISION_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UM_KLINES_PREFIX = "data/futures/um/monthly/klines/"
KLINE_INTERVAL = "4h"

# 분당 요청 가중치 한도(선물 2400)에 여유를 두고 이 값을 넘으면 잠시 쉰다.
WEIGHT_SOFT_LIMIT = 1800

# 현물 유니버스에서 뺄 스테이블코인(스테이블/스테이블 페어는 시그널 무의미)
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "EURI", "AEUR",
                "USDE", "USD1", "XUSD", "BUSD"}


class GeoBlockedError(RuntimeError):
    """미국 등 제한 지역 IP에서 바이낸스가 451을 돌려줄 때."""


def _safe(exc: Exception) -> str:
    """예외 메시지에서 URL 자격증명(user:pass@)을 지운 문자열.

    프록시 연결 실패 예외는 프록시 URL을 통째로 담고 있어서 그대로 찍으면
    BINANCE_PROXY의 아이디·비밀번호가 로그에 남는다. 퍼블릭 레포는 Actions
    로그도 공개되므로(시크릿 자동 마스킹은 값이 정확히 일치할 때만 걸린다)
    로그로 내보내기 전에 여기서 지운다.
    """
    return re.sub(r"(?<=//)[^/\s@]+:[^/\s@]+@", "***:***@", str(exc))


def fapi_base() -> str:
    # GitHub Actions는 미설정 시크릿을 빈 문자열로 넘기므로 빈 값도 기본값 취급
    return (os.environ.get("BINANCE_FAPI_BASE") or FAPI_BASE).rstrip("/")


def _proxies_for(base: str) -> dict | None:
    """fapi 요청만 BINANCE_PROXY(비미국 프록시)로 보낸다.

    현물 미러·S3는 미국에서도 열려 있으므로 직결 유지 — 프록시가 죽어도
    폴백 경로는 영향을 받지 않는다. 형식: http://user:pass@ip:port 또는
    socks5h://user:pass@ip:port.
    """
    p = os.environ.get("BINANCE_PROXY") or None
    if p and base == fapi_base():
        return {"http": p, "https": p}
    return None


def _get(session: requests.Session, base: str, path: str,
         params: dict | None = None, tries: int = 3) -> requests.Response:
    url = f"{base}{path}"
    delay = 1.0
    proxies = _proxies_for(base)
    for attempt in range(1, tries + 1):
        try:
            rsp = session.get(url, params=params, timeout=20, proxies=proxies)
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


def _parse_s3_listing(text: str) -> tuple[list[str], bool, str | None]:
    """S3 ListObjects XML → (심볼 목록, 다음 페이지 여부, NextMarker)."""
    syms = re.findall(
        r"<Prefix>" + re.escape(UM_KLINES_PREFIX) + r"([^/<]+)/</Prefix>", text)
    truncated = "<IsTruncated>true</IsTruncated>" in text
    m = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
    return syms, truncated, m.group(1) if m else None


def um_futures_symbols(session: requests.Session) -> set[str]:
    """USDT-M 선물로 상장된 적 있는 심볼 집합 (공식 데이터 저장소 S3 목록).

    fapi가 막힌 미국 IP에서도 접근된다. 상장폐지된 퍼프도 포함되므로
    'ADX처럼 현물 전용인 코인'을 걸러내는 용도로만 쓴다.
    """
    out: set[str] = set()
    marker = None
    for _ in range(20):                 # 안전 상한 — 실제로는 1~2페이지
        params = {"prefix": UM_KLINES_PREFIX, "delimiter": "/"}
        if marker:
            params["marker"] = marker
        rsp = session.get(DATA_VISION_S3, params=params, timeout=20)
        rsp.raise_for_status()
        syms, truncated, marker = _parse_s3_listing(rsp.text)
        out.update(s for s in syms if "_" not in s)   # BTCUSDT_210625 분기물 제외
        if not truncated or not syms:
            break
        if marker is None:              # 델리미터 응답엔 NextMarker가 있지만 방어
            marker = UM_KLINES_PREFIX + syms[-1] + "/"
    if not out:
        raise RuntimeError("퍼프 심볼 목록이 비어 있음")
    return out


def parse_klines(rows: list, now_ms: int | None = None,
                 include_live: bool = False) -> pd.DataFrame:
    """kline 배열 → OHLCV DataFrame(UTC 인덱스).

    기본은 진행 중인 마지막 봉을 버린다(마감 확정 봉만 판정).
    include_live=True면 진행 중 봉도 포함 — 펌핑 인트라바 터치 판정용.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    keep = rows if include_live else [r for r in rows if int(r[6]) <= now_ms]   # r[6] = closeTime(ms)
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
    """설정된 source(auto|fapi|spot_mirror)를 실제 소스+심볼 목록으로 확정.

    auto 모드는 451뿐 아니라 어떤 fapi 실패(타임아웃·연결 리셋 등)에도
    현물 미러로 폴백한다 — 지역 차단이 늘 깔끔한 451로 오지는 않는다.
    """
    if requested in ("auto", "fapi"):
        try:
            syms = usdt_perp_symbols(session, exclude)
            if os.environ.get("BINANCE_PROXY"):
                log("[binance] BINANCE_PROXY 경유로 fapi(선물) 접속 성공")
            return "fapi", syms
        except GeoBlockedError:
            if requested == "fapi":
                raise
            log("[binance] fapi 451 차단 — 현물 미러(data-api.binance.vision)로 폴백")
        except Exception as e:              # noqa: BLE001
            if requested == "fapi":
                raise
            log(f"[binance] fapi 실패({_safe(e)}) — 현물 미러로 폴백")
    syms = usdt_spot_symbols(session, exclude)
    try:
        # 퍼프 상장 이력 없는 현물 전용 코인(ADX 등)은 스캔에서 제외
        perps = um_futures_symbols(session)
        before = len(syms)
        syms = [s for s in syms if s in perps]
        log(f"[binance] 퍼프 미상장 제외: 현물 {before}종 → {len(syms)}종")
    except Exception as e:                  # noqa: BLE001 — 목록 조회 실패가 스캔을 막지 않게
        log(f"[binance] 퍼프 심볼 목록 조회 실패({_safe(e)}) — 현물 전체 스캔")
    return "spot_mirror", syms


def klines_4h(session: requests.Session, symbol: str, source: str,
              limit: int = 600, include_live: bool = False) -> pd.DataFrame:
    if source == "fapi":
        base, path = fapi_base(), "/fapi/v1/klines"
    else:
        base, path = SPOT_MIRROR_BASE, "/api/v3/klines"
    rows = _get(session, base, path,
                {"symbol": symbol, "interval": KLINE_INTERVAL, "limit": limit}).json()
    return parse_klines(rows, include_live=include_live)


def fetch_all(session: requests.Session, symbols: list[str], source: str,
              limit: int = 600, pause: float = 0.1,
              log=print, workers: int = 6,
              include_live: bool = False) -> dict[str, pd.DataFrame]:
    """전 심볼 4h 캔들 병렬 수집. 개별 실패는 건너뛰고, 451은 즉시 중단.

    requests.Session은 스레드 간 공유가 안전하지 않아 워커 스레드마다
    세션을 하나씩 만든다(인자로 받은 session은 단일 워커일 때만 사용).
    페이싱: 워커당 요청 뒤 pause씩 쉬므로 워커 6 × klines 가중치 5 기준
    분당 ~1500으로 fapi 한도(2400)에 여유가 있고, 초과 조짐이 보이면
    _throttle이 해당 워커를 재운다.
    """
    out: dict[str, pd.DataFrame] = {}
    failed = []
    workers = max(1, int(workers))
    tls = threading.local()
    sessions, sess_lock = [], threading.Lock()

    def get_session() -> requests.Session:
        if workers == 1:
            return session
        if getattr(tls, "s", None) is None:
            tls.s = requests.Session()
            with sess_lock:
                sessions.append(tls.s)
        return tls.s

    def one(sym: str) -> pd.DataFrame:
        df = klines_4h(get_session(), sym, source, limit, include_live=include_live)
        if pause:
            time.sleep(pause)
        return df

    geo_err = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, sym): sym for sym in symbols}
            done = 0
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    out[sym] = fut.result()
                except GeoBlockedError as e:
                    geo_err = e
                    ex.shutdown(cancel_futures=True)
                    break
                except Exception as e:             # noqa: BLE001 — 종목별 실패는 스캔 전체를 막지 않는다
                    failed.append(sym)
                    log(f"[binance] {sym} 수집 실패: {_safe(e)}")
                done += 1
                if done % 100 == 0:
                    log(f"[binance] {done}/{len(symbols)} 수집")
    finally:
        for s in sessions:
            s.close()
    if geo_err is not None:
        raise geo_err
    if failed:
        log(f"[binance] 실패 {len(failed)}종: {', '.join(failed[:10])}"
            + (" ..." if len(failed) > 10 else ""))
    return out
