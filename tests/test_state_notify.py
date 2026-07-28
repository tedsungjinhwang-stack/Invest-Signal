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
    assert ">KODEX 레버리지</a>" in msg    # 한국 종목은 코드 없이 종목명만 표시


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
    assert "↳ SOL  68.2 · " in msg
    assert "⚠️MSS 저점 70 이탈" in msg        # 추적 MSS도 신규와 같은 형식
    assert "blockquote" not in msg            # 접힘·들여쓰기 없이 전부 표시
    assert "눌림목" in msg and "·⚠️MSS" in msg  # MSS는 눌림목 칸에 태그로 병합
    # 추적 목록이 없으면 ↳ 줄도 없다
    assert "↳" not in format_events([_event("BTCUSDT")], [], {})


def test_ongoing_list_caps_per_label():
    evs = [_event(f"C{i:02d}USDT", {"label": "눌림목"}) for i in range(20)]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=evs)
    assert "↳ 외 5종" in msg
    assert msg.count("↳") == 16               # 15종 각 한 줄 + '외 N종' 줄


def test_intrabar_tag_shown():
    """진행봉 잠정 판정 알림엔 ⏳진행봉 태그가 붙는다."""
    ev = _event("BTCUSDT", {"label": "눌림목", "intrabar": True, "align": "정배열"})
    msg = format_events([ev], [], {})
    assert "⏳진행봉" in msg


def test_pump_section_with_gain_tag():
    """펌핑은 🚀 칸에 크립토만 — 저점대비 상승률·주간VWAP 태그 포함."""
    ev = SignalEvent(symbol="AKEUSDT", signal="pump_dip",
                     bar_time=pd.Timestamp("2026-07-21 04:00", tz="UTC"),
                     price=0.0015, detail={"label": "펌핑", "pump_gain": 2.0})
    msg = format_events([ev], [], {})
    assert "🚀 <b>펌핑</b>" in msg
    assert "저점대비 +200% · 월간VWAP 터치" in msg
    assert ">AKE</a>" in msg


def test_pullback_stage_tags():
    """눌림목은 타점(밴드 터치)과 대기(밴드 위)를 태그로 구분한다."""
    entry = _event("BTCUSDT", {"label": "눌림목", "stage": "타점", "align": "정배열"})
    wait = _event("ETHUSDT", {"label": "눌림목", "stage": "대기", "align": "정배열",
                              "last_price": 3400.0})
    msg = format_events([entry], [], {}, ongoing_crypto=[wait])
    assert "🎯타점" in msg
    assert "↳ ETH  3,400.0 · " in msg and "대기" in msg


def test_pump_early_section_with_rise_tag():
    """펌핑초기는 🌱 칸에 — 급등 구간 시간·상승률과 저점대비 누적을 함께 표시."""
    ev = SignalEvent(symbol="XYZUSDT", signal="pump_early",
                     bar_time=pd.Timestamp("2026-07-21 04:00", tz="UTC"),
                     price=1.05, detail={"label": "펌핑초기", "rise": 0.072,
                                         "rise_bars": 1, "total_gain": 0.18})
    msg = format_events([ev], [], {})
    assert "🌱 <b>펌핑초기</b>" in msg
    assert "4h +7.2% · 저점대비 +18%" in msg


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
    hold = _event("ETHUSDT", {"label": "눌림목", "align": "정배열", "last_price": 3400.0})
    mixed = _event("SOLUSDT", {"label": "눌림목", "align": "혼조", "last_price": 70.0})
    mss = _event("XRPUSDT", {"label": "MSS", "align": "정배열", "last_price": 3.0})
    msg = format_events([new], [], {}, ongoing_crypto=[hold, mixed, mss])
    assert "역배열" in msg
    assert "·정배열" in msg
    assert "·혼조" in msg                             # 혼조도 배열 태그로 표시
    assert "↳ XRP  3 · " in msg and "MSS" in msg
    xrp_line = next(l for l in msg.split("\n") if l.startswith("↳ XRP"))
    assert "정배열" not in xrp_line                   # MSS 항목은 MSS만
