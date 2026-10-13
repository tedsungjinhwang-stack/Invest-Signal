"""캔들 캐시(10-13) — 이어 받은 결과가 처음부터 받은 결과와 같아야 한다."""

import numpy as np
import pandas as pd
import pytest

from invest_signal import data_binance as db

STEP = 900_000                      # 15m


class FakeServer:
    """시각 now_ms에 진행 중인 봉까지 내주는 가짜 바이낸스 — 요청한 limit개(마지막이 진행 중 봉)."""

    def __init__(self, start_ms, n_total):
        rng = np.random.default_rng(0)
        self.rows = []
        p = 1.0
        for i in range(n_total):
            o = p
            c = o * (1 + rng.normal(0, 0.01))
            h, l = max(o, c) * 1.003, min(o, c) * 0.997
            t = start_ms + i * STEP
            self.rows.append([t, f"{o:.8f}", f"{h:.8f}", f"{l:.8f}", f"{c:.8f}",
                              f"{rng.uniform(10, 100):.3f}", t + STEP - 1, "0", 0, "0", "0", "0"])
            p = c
        self.now = None
        self.limits = []

    def get(self, session, base, path, params=None, tries=3):
        lim = params["limit"]
        self.limits.append(lim)
        live = [r for r in self.rows if r[0] <= self.now]
        out = [list(r) for r in live[-lim:]]
        # 진행 중 봉은 지금까지의 값만(마감 전) — 종가가 바뀌어 있을 수 있다
        if out and out[-1][6] > self.now:
            out[-1][4] = out[-1][1]

        class R:
            def json(self_inner):
                return out
        return R()


@pytest.fixture
def server(monkeypatch):
    start = 1_700_000_000_000 - (1_700_000_000_000 % STEP)
    srv = FakeServer(start, 3000)
    monkeypatch.setattr(db, "_get", srv.get)
    clock = {"t": 0}
    monkeypatch.setattr(db.time, "time", lambda: clock["t"] / 1000)
    db._cache.clear()
    db._cache_on = False
    yield srv, start, clock
    db._cache.clear()
    db._cache_on = False


def _fresh(srv, limit, live):
    on = db._cache_on
    db._cache_on = False
    try:
        return db.klines(None, "XUSDT", "fapi", "15m", limit, include_live=live)
    finally:
        db._cache_on = on


def test_tail_fetch_matches_full_fetch(server, tmp_path):
    srv, start, clock = server
    db._cache_on = True
    t = start + 1200 * STEP + 2 * 60_000          # 1200번째 봉 시작 2분 뒤
    for k in range(30):                           # 매시간 스캔 30번 — 1h = 4봉
        now = t + k * 4 * STEP + (k % 3) * 37_000
        srv.now = clock["t"] = now
        for limit, live in ((200, False), (1000, False), (1000, True), (200, True)):
            got = db.klines(None, "XUSDT", "fapi", "15m", limit, include_live=live)
            want = _fresh(srv, limit, live)
            pd.testing.assert_frame_equal(got, want)
    assert db.cache_stats["tail"] > 0
    # 이어 받기 요청은 작다(가중치 1)
    assert max(srv.limits[-20:]) < 100 or 1000 in srv.limits[-20:]
    # 파일로 저장 → 다시 불러와도 같은 결과
    p = str(tmp_path / "k.pkl.gz")
    assert db.save_cache(p) == 1
    db._cache.clear()
    assert db.load_cache(p) == 1
    srv.now = clock["t"] = now + 5 * STEP
    pd.testing.assert_frame_equal(db.klines(None, "XUSDT", "fapi", "15m", 1000),
                                  _fresh(srv, 1000, False))


def test_long_gap_or_short_cache_refetches_full(server):
    srv, start, clock = server
    db._cache_on = True
    srv.now = clock["t"] = start + 1500 * STEP + 1000
    db.klines(None, "XUSDT", "fapi", "15m", 200)
    n = len(srv.limits)
    db.klines(None, "XUSDT", "fapi", "15m", 1000)     # 캐시(199봉)보다 길게 — 처음부터
    assert srv.limits[n:] == [1000]
    srv.now = clock["t"] = srv.now + 200 * STEP        # 50시간 빔 — 처음부터
    n = len(srv.limits)
    got = db.klines(None, "XUSDT", "fapi", "15m", 1000)
    assert srv.limits[n:] == [1000]
    pd.testing.assert_frame_equal(got, _fresh(srv, 1000, False))
    srv.now = clock["t"] = srv.now + 4 * STEP          # 한 시간 — 이어 받기
    n = len(srv.limits)
    got = db.klines(None, "XUSDT", "fapi", "15m", 1000)
    assert srv.limits[n:] == [7]                       # 새 마감 4 + 진행 1 + 겹침 2
    pd.testing.assert_frame_equal(got, _fresh(srv, 1000, False))


def test_missing_cache_file_is_fine(tmp_path):
    db._cache.clear()
    assert db.load_cache(str(tmp_path / "nope.pkl.gz")) == 0
    db._cache_on = False
