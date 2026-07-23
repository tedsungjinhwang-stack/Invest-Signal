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
    assert "크립토" in msg and "ETF" in msg
    assert "binance.com/en/futures/BTCUSDT" in msg
    assert "KRX-122630" in msg                 # 한국 종목은 KRX 차트 링크
    assert "QVWAP" not in msg              # QVWAP은 조건으로만 쓰고 표시하지 않음
    assert "KODEX 레버리지" in msg


def test_split_chunks_preserves_content():
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    chunks = split_chunks(text, size=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text


def test_format_events_ongoing_section():
    ongoing = [_event("ETHUSDT", {"label": "상승초입", "last_price": 3420.5}),
               _event("SOLUSDT", {"label": "MSS", "broken_low": 70.0,
                                  "last_price": 68.2})]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=ongoing)
    assert "↳ ETH  3,420.5" in msg            # 종목마다 한 줄 + 현재가
    assert "↳ SOL  68.2" in msg
    assert "blockquote" not in msg            # 접힘·들여쓰기 없이 전부 표시
    assert "풀백" in msg and "·MSS" in msg    # MSS는 풀백 칸에 태그로 병합
    # 추적 목록이 없으면 ↳ 줄도 없다
    assert "↳" not in format_events([_event("BTCUSDT")], [], {})


def test_ongoing_list_caps_per_label():
    evs = [_event(f"C{i:02d}USDT", {"label": "풀백"}) for i in range(20)]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=evs)
    assert "↳ 외 5종" in msg
    assert msg.count("↳") == 16               # 15종 각 한 줄 + '외 N종' 줄


def test_downtrend_section_for_etf_and_stocks():
    """하락전환은 🔻 칸에 ETF·주식만 — 직전저점 태그 포함."""
    ch = SignalEvent(symbol="SOXL", signal="downtrend_reversal",
                     bar_time=pd.Timestamp("2026-07-21 04:00", tz="UTC"),
                     price=20.5, detail={"label": "하락전환", "broken_low": 21.0})
    msg = format_events([], [ch], {"SOXL": "반도체 3x"})
    assert "🔻 <b>하락전환</b>" in msg
    assert "직전저점 21 이탈" in msg
    # 추적(ongoing)도 같은 칸 아래 ↳ 줄
    msg2 = format_events([_event("BTCUSDT")], [], {}, ongoing_etf=[ch])
    assert "하락전환" in msg2 and "↳ SOXL" in msg2


def test_align_tag_shown_in_lines_and_ongoing():
    new = _event("BTCUSDT", {"label": "상승초입", "align": "역배열"})
    hold = _event("ETHUSDT", {"label": "풀백", "align": "정배열", "last_price": 3400.0})
    mixed = _event("SOLUSDT", {"label": "풀백", "align": "혼조", "last_price": 70.0})
    mss = _event("XRPUSDT", {"label": "MSS", "align": "정배열", "last_price": 3.0})
    msg = format_events([new], [], {}, ongoing_crypto=[hold, mixed, mss])
    assert "역배열" in msg
    assert "·정배열" in msg
    assert "혼조" not in msg                          # 혼조는 표시하지 않는다
    assert "↳ XRP  3 · " in msg and "MSS" in msg
    xrp_line = next(l for l in msg.split("\n") if l.startswith("↳ XRP"))
    assert "정배열" not in xrp_line                   # MSS 항목은 MSS만
