"""상태(중복 방지)와 알림 포맷 테스트."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from invest_signal.notify import format_events, split_chunks
from invest_signal.signals import SignalEvent
from invest_signal.state import AlertState


def _event(symbol, detail=None):
    return SignalEvent(symbol=symbol, signal="uptrend_onset",
                       bar_time=pd.Timestamp("2026-07-21 04:00", tz="UTC"),
                       price=123.45, detail=detail or {"label": "상승초입", "entry_ma": 130.0})


def test_state_dedup_roundtrip(tmp_path):
    p = str(tmp_path / "state.json")
    st = AlertState(p)
    key = _event("BTCUSDT").dedup_key
    assert st.is_new(key)
    st.mark(key)
    st.save()
    st2 = AlertState(p)
    assert not st2.is_new(key)


def test_state_prunes_old_entries(tmp_path):
    p = str(tmp_path / "state.json")
    st = AlertState(p)
    old = datetime.now(timezone.utc) - timedelta(days=40)
    st.mark("OLD|sig|t", when=old)
    st.mark("NEW|sig|t")
    st.save()
    st2 = AlertState(p)
    assert st2.is_new("OLD|sig|t")
    assert not st2.is_new("NEW|sig|t")


def test_state_survives_corrupt_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{broken json", encoding="utf-8")
    st = AlertState(str(p))
    assert st.is_new("ANY")


def test_format_events_sections_and_links():
    msg = format_events(
        [_event("BTCUSDT", {"label": "상승초입", "entry_ma": 130.0, "above_qvwap": True,
                            "touch_time": "2026-07-19T12:00:00+00:00"})],
        [_event("122630"), _event("SOXL")],
        {"122630": "KODEX 레버리지", "SOXL": "반도체 3x"})
    assert "크립토" in msg and "레버리지 ETF" in msg
    assert "binance.com/en/futures/BTCUSDT" in msg
    assert "KRX-122630" in msg                 # 한국 종목은 KRX 차트 링크
    assert "QVWAP↑" in msg
    assert "KODEX 레버리지" in msg


def test_split_chunks_preserves_content():
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    chunks = split_chunks(text, size=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text


def test_format_events_ongoing_section():
    ongoing = [_event("ETHUSDT", {"label": "상승초입"}),
               _event("SOLUSDT", {"label": "MSS", "broken_low": 70.0})]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=ongoing)
    assert "유지 중" in msg
    assert "상승초입: ETHUSDT" in msg
    assert "MSS: SOLUSDT" in msg
    # 유지 목록이 없으면 섹션도 없다
    assert "유지 중" not in format_events([_event("BTCUSDT")], [], {})


def test_ongoing_list_caps_per_label():
    evs = [_event(f"C{i:02d}USDT", {"label": "눌림목"}) for i in range(20)]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=evs)
    assert "외 5종" in msg
