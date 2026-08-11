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

    # include_live=True면 진행 중 봉도 포함 (인트라바 판정용)
    df3 = parse_klines(rows, now_ms=now, include_live=True)
    assert len(df3) == 2


def test_fetch_all_parallel_collects_and_skips_failures(monkeypatch):
    """병렬 수집 — 전 종목 수집, 개별 실패는 건너뛰고, 451은 전파."""
    from invest_signal import data_binance as db

    def fake_klines(session, sym, source, limit, include_live=False):
        if sym == "BADUSDT":
            raise RuntimeError("boom")
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(db, "klines_4h", fake_klines)
    syms = [f"C{i:03d}USDT" for i in range(25)] + ["BADUSDT"]
    out = db.fetch_all(None, syms, "spot_mirror", pause=0, log=lambda *a: None,
                       workers=6)
    assert len(out) == 25 and "BADUSDT" not in out

    def geo_klines(session, sym, source, limit, include_live=False):
        raise db.GeoBlockedError("451")

    monkeypatch.setattr(db, "klines_4h", geo_klines)
    import pytest
    with pytest.raises(db.GeoBlockedError):
        db.fetch_all(None, ["AUSDT", "BUSDT"], "fapi", pause=0,
                     log=lambda *a: None, workers=4)


def test_binance_proxy_applies_to_fapi_only(monkeypatch):
    """BINANCE_PROXY는 fapi 요청에만 붙는다 — 미러/S3 폴백 경로는 직결."""
    from invest_signal.data_binance import FAPI_BASE, SPOT_MIRROR_BASE, _proxies_for

    monkeypatch.delenv("BINANCE_PROXY", raising=False)
    assert _proxies_for(FAPI_BASE) is None
    monkeypatch.setenv("BINANCE_PROXY", "")          # Actions의 미설정 시크릿 = 빈 문자열
    assert _proxies_for(FAPI_BASE) is None
    monkeypatch.setenv("BINANCE_PROXY", "http://u:p@1.2.3.4:8080")
    assert _proxies_for(FAPI_BASE) == {"http": "http://u:p@1.2.3.4:8080",
                                       "https": "http://u:p@1.2.3.4:8080"}
    assert _proxies_for(SPOT_MIRROR_BASE) is None    # 미러는 프록시 안 탐


def test_safe_scrubs_proxy_credentials_from_exception_text():
    """프록시 실패 예외를 로그로 내보내기 전에 user:pass@를 지운다.

    퍼블릭 레포는 Actions 로그도 공개되므로 자격증명이 새면 안 된다.
    """
    from invest_signal.data_binance import _safe

    msg = _safe(Exception("Unable to connect to proxy "
                          "socks5h://myuser:sup3rs3cret@203.0.113.9:1080"))
    assert "myuser" not in msg and "sup3rs3cret" not in msg
    assert "***:***@203.0.113.9:1080" in msg      # 호스트·포트는 남아 진단은 가능
    # 자격증명이 없는 평범한 메시지는 그대로 통과
    assert _safe(Exception("Read timed out")) == "Read timed out"


def test_fill_missing_kr_names(monkeypatch):
    """이름 빈 한국 종목만 야후에서 보충 — 실패·미국 종목은 그대로."""
    import yfinance

    from invest_signal import data_etf

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def get_info(self):
            if self.sym == "010120.KS":
                return {"shortName": "LS ELECTRIC"}
            raise RuntimeError("no data")

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    ticks = [{"code": "010120", "market": "KR", "name": ""},
             {"code": "005930", "market": "KR", "name": "삼성전자"},
             {"code": "319660", "market": "KQ", "name": ""},   # .KS/.KQ 둘 다 실패 → 코드 유지
             {"code": "OVV", "market": "US", "name": ""}]
    data_etf.fill_missing_kr_names(ticks, log=lambda *a: None)
    assert ticks[0]["name"] == "LS ELECTRIC"
    assert ticks[1]["name"] == "삼성전자"     # 이미 있으면 안 건드림
    assert ticks[2]["name"] == ""             # 조회 실패는 무시(코드 표시 폴백)
    assert ticks[3]["name"] == ""             # 미국은 티커로 충분 — 보충 대상 아님


def test_parse_s3_listing_symbols_and_pagination():
    """S3 목록 XML에서 퍼프 심볼을 뽑고 페이지네이션 마커를 읽는다."""
    from invest_signal.data_binance import UM_KLINES_PREFIX, _parse_s3_listing

    p = UM_KLINES_PREFIX
    xml = f"""<ListBucketResult>
      <IsTruncated>true</IsTruncated>
      <NextMarker>{p}ETHUSDT/</NextMarker>
      <CommonPrefixes><Prefix>{p}ADAUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>{p}BTCUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>{p}BTCUSDT_210625/</Prefix></CommonPrefixes>
    </ListBucketResult>"""
    syms, truncated, marker = _parse_s3_listing(xml)
    assert syms == ["ADAUSDT", "BTCUSDT", "BTCUSDT_210625"]
    assert truncated and marker == f"{p}ETHUSDT/"

    xml2 = f"""<ListBucketResult>
      <IsTruncated>false</IsTruncated>
      <CommonPrefixes><Prefix>{p}ETHUSDT/</Prefix></CommonPrefixes>
    </ListBucketResult>"""
    syms2, truncated2, marker2 = _parse_s3_listing(xml2)
    assert syms2 == ["ETHUSDT"] and not truncated2 and marker2 is None


def test_spot_fallback_drops_symbols_without_perp(monkeypatch):
    """현물 미러 폴백 시 퍼프 미상장 심볼(ADX 등)은 스캔 대상에서 빠진다."""
    from invest_signal import data_binance as db

    monkeypatch.setattr(db, "usdt_spot_symbols",
                        lambda s, e: ["ADXUSDT", "BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(db, "um_futures_symbols",
                        lambda s: {"BTCUSDT", "ETHUSDT", "SOLUSDT"})
    source, syms = db.resolve_source(None, "spot_mirror", set(), log=lambda *a: None)
    assert source == "spot_mirror"
    assert syms == ["BTCUSDT", "ETHUSDT"]      # ADXUSDT 제외

    # 목록 조회 실패는 스캔을 막지 않고 현물 전체로 진행
    def boom(s):
        raise RuntimeError("s3 down")
    monkeypatch.setattr(db, "um_futures_symbols", boom)
    _, syms = db.resolve_source(None, "spot_mirror", set(), log=lambda *a: None)
    assert syms == ["ADXUSDT", "BTCUSDT", "ETHUSDT"]


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


def test_resample_4h_excludes_bucket_fed_by_in_progress_us_bar():
    """미국 :30 기준 1h봉 — 진행 중인 15:30봉이 속한 12:00 버킷은 확정 전엔 제외."""
    idx = pd.date_range("2025-06-02 13:30", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"Open": [10.0, 11, 12], "High": [11.0, 12, 13],
                       "Low": [9.0, 10, 11], "Close": [10.5, 11.5, 12.5],
                       "Volume": [1.0, 1, 1]}, index=idx)
    # 16:07 — 15:30봉이 아직 진행 중 → 12:00 버킷 전체가 미확정 → 아무것도 없음
    assert resample_4h(df, now=pd.Timestamp("2025-06-02 16:07", tz="UTC")) is None
    # 16:37 — 15:30봉 마감 → 12:00 버킷 확정, 종가는 15:30봉 종가
    out = resample_4h(df, now=pd.Timestamp("2025-06-02 16:37", tz="UTC"))
    assert list(out.index.hour) == [12] and out["Close"].iloc[0] == 12.5
