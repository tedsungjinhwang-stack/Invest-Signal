"""상태(중복 방지)와 알림 포맷 테스트."""

import json
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


def test_dedup_key_separates_pullback_stages(tmp_path):
    """같은 봉이 대기 → 타점으로 바뀌면 새 알림이어야 한다.

    인트라바 스캔에서 '대기'로 알린 봉이 마감 때 밴드를 터치해 '타점'이
    되는 경우 — 키가 같으면 정작 중요한 진입 신호가 막힌다.
    """
    wait = _event("BTCUSDT", {"label": "눌림목", "stage": "대기"})
    entry = _event("BTCUSDT", {"label": "눌림목", "stage": "타점"})
    assert wait.dedup_key != entry.dedup_key
    st = AlertState(str(tmp_path / "state.json"))
    st.mark(wait.dedup_key)
    assert st.is_new(entry.dedup_key)
    # 단계가 없는 시그널은 키 형식이 그대로다
    assert _event("BTCUSDT").dedup_key.count("|") == 2


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
        [_event("BTCUSDT", {"label": "상승초입", "entry_ma": 130.0, "above_vwap": True,
                            "touch_time": "2026-07-19T12:00:00+00:00"})],
        [_event("122630"), _event("SOXL")],
        {"122630": "KODEX 레버리지", "SOXL": "반도체 3x"})
    assert "크립토" in msg and "ETF" in msg
    assert "binance.com/en/futures/BTCUSDT" in msg
    assert "KRX-122630" in msg                 # 한국 종목은 KRX 차트 링크
    assert "VWAP" not in msg               # VWAP은 조건으로만 쓰고 표시하지 않음
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
    assert "눌림목" in msg and " · ⚠️MSS" in msg  # MSS는 눌림목 칸에 태그로 병합
    # 추적 목록이 없으면 ↳ 줄도 없다
    assert "↳" not in format_events([_event("BTCUSDT")], [], {})


def test_ongoing_list_shows_every_symbol():
    """추적 목록은 자르지 않는다 — 20종이면 20줄 그대로."""
    evs = [_event(f"C{i:02d}USDT", {"label": "눌림목"}) for i in range(20)]
    msg = format_events([_event("BTCUSDT")], [], {}, ongoing_crypto=evs)
    assert msg.count("↳") == 20
    assert "외 " not in msg
    for i in range(20):
        assert f"C{i:02d}" in msg


def test_intrabar_tag_shown():
    """진행봉 잠정 판정 알림엔 ⏳진행봉 태그가 붙는다."""
    ev = _event("BTCUSDT", {"label": "눌림목", "intrabar": True, "align": "정배열"})
    msg = format_events([ev], [], {})
    assert "⏳진행봉" in msg



def test_pullback_stage_tags():
    """눌림목은 타점(밴드 터치)과 대기(밴드 위)를 태그로 구분한다."""
    entry = _event("BTCUSDT", {"label": "눌림목", "stage": "타점", "align": "정배열"})
    wait = _event("ETHUSDT", {"label": "눌림목", "stage": "대기", "align": "정배열",
                              "last_price": 3400.0})
    msg = format_events([entry], [], {}, ongoing_crypto=[wait])
    assert "🎯타점" in msg
    assert "↳ ETH  3,400.0 · " in msg and "대기" in msg


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
    assert " · 정배열" in msg
    assert " · 혼조" in msg                           # 혼조도 배열 태그로 표시
    assert "↳ XRP  3 · " in msg and "MSS" in msg
    xrp_line = next(l for l in msg.split("\n") if l.startswith("↳ XRP"))
    assert "정배열" not in xrp_line                   # MSS 항목은 MSS만


def test_leaders_persist_and_expire_after_watch_window(tmp_path):
    """상위권 등재 이력은 저장되고, 창을 벗어나면 prune에서 사라진다."""
    from datetime import datetime, timedelta, timezone

    from invest_signal.state import AlertState

    p = tmp_path / "s.json"
    # save()의 prune은 실제 시계로 돈다 — 고정 날짜를 쓰면 그 날짜가 과거가
    # 되는 순간 등재분이 통째로 잘려 테스트가 시한폭탄이 된다.
    now = datetime.now(timezone.utc)
    st = AlertState(str(p))
    st.touch_leaders(["AUSDT", "BUSDT"], when=now - timedelta(days=1))
    st.touch_leaders(["OLDUSDT"], when=now - timedelta(days=9))
    st.save()

    reloaded = AlertState(str(p))          # 파일에서 다시 읽어도 남아 있어야 한다
    recent = reloaded.recent_leaders(days=7, now=now)
    assert set(recent) == {"AUSDT", "BUSDT"}    # 9일 전 등재분은 창 밖
    assert recent["AUSDT"] == now - timedelta(days=1)
    # save()의 prune이 창 밖 항목을 파일에서도 지운다
    assert "OLDUSDT" not in json.loads(p.read_text(encoding="utf-8"))["leaders"]


def test_leaders_absent_in_legacy_state_file(tmp_path):
    """leaders 키가 없던 기존 상태 파일도 그대로 읽힌다."""
    from invest_signal.state import AlertState

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"alerts": {"X|pullback|2026-01-01T00:00:00+00:00": "2026-01-01T00:00:00+00:00"}}),
                 encoding="utf-8")
    st = AlertState(str(p))
    assert st.recent_leaders() == {}
    assert not st.is_new("X|pullback|2026-01-01T00:00:00+00:00")


def test_hold_line_shows_leader_break_rank_and_ma_side():
    """유지 중 줄 — 상위권이면 순위, 밀렸으면 추적일차 + 60선 위/아래."""
    top = SignalEvent(symbol="HFTUSDT", signal="leader_break",
                      bar_time=pd.Timestamp("2026-08-06T06:00:00Z"), price=0.0174,
                      detail={"label": "크립토 모멘텀 눌림목/이탈", "last_price": 0.0174,
                              "ma": 0.0180, "above_ma": False, "rank": 2,
                              "gain_24h": 0.41})
    lapsed = SignalEvent(symbol="ETHUSDT", signal="leader_break",
                         bar_time=pd.Timestamp("2026-08-04T06:00:00Z"), price=3120.0,
                         detail={"label": "크립토 모멘텀 눌림목/이탈", "last_price": 3200.0,
                                 "ma": 3155.0, "above_ma": True, "watch_days": 3,
                                 "gain_24h": -0.02})
    out = format_events([], [], {}, ongoing_crypto=[top, lapsed])
    assert "↳ HFT  0.0174 · 2위 · 🔻60선 아래 · 24h +41%" in out
    assert "↳ ETH  3,200.0 · 추적 3일차 · 60선 위 · 24h -2.0%" in out
    # 60선 위로 복귀한 종목도 목록에 남아 있어야 한다
    assert "ETH" in out


def test_crypto_board_leads_the_leader_break_section():
    """⚡칸 맨 위에 24h 상승률 순위표 — 조건 충족 종목보다 먼저 온다."""
    board = [{"symbol": "BICOUSDT", "rank": 1, "gain_24h": 0.37, "price": 0.05101},
             {"symbol": "TSTUSDT", "rank": 2, "gain_24h": 0.34, "price": 0.01407}]
    ev = SignalEvent(symbol="HFTUSDT", signal="leader_break",
                     bar_time=pd.Timestamp("2026-08-06T06:00:00Z"), price=0.0174,
                     detail={"label": "크립토 모멘텀 눌림목/이탈", "ma": 0.0180,
                             "ma_period": 20, "interval": "15m", "gain_24h": 0.13})
    msg = format_events([ev], [], {}, crypto_board=board)
    assert "📈 <b>24h 상승률 TOP</b>" in msg
    assert "1. " in msg and "BICO" in msg and "+37%" in msg
    # 순위표가 조건 충족 줄(•)보다 위에 있어야 한다
    assert msg.index("24h 상승률 TOP") < msg.index("• ")
    # 순위표 종목은 조건과 무관 — TST는 이탈하지 않았는데도 실린다
    assert "TST" in msg


def test_crypto_board_renders_even_without_leader_events():
    """이탈 종목이 없어도 순위표만으로 ⚡칸이 만들어진다."""
    board = [{"symbol": "AUSDT", "rank": 1, "gain_24h": 0.9, "price": 1.5}]
    other = SignalEvent(symbol="X", signal="uptrend_onset",
                        bar_time=pd.Timestamp("2026-08-06T06:00:00Z"), price=1.0,
                        detail={"label": "상승초입"})
    msg = format_events([other], [], {}, crypto_board=board)
    assert "⚡ <b>크립토 모멘텀 눌림목/이탈</b>" in msg
    assert "1. " in msg and "+90%" in msg


def test_no_board_leaves_message_unchanged():
    ev = SignalEvent(symbol="X", signal="uptrend_onset",
                     bar_time=pd.Timestamp("2026-08-06T06:00:00Z"), price=1.0,
                     detail={"label": "상승초입"})
    assert "24h 상승률 TOP" not in format_events([ev], [], {})


def test_returns_tag_on_uptrend_and_leader_lines():
    """상승초입·크립토 모멘텀 줄에 1h·4h·24h 수익률이 붙는다."""
    t = pd.Timestamp("2026-08-13T12:00:00Z")
    up = SignalEvent(symbol="KITEUSDT", signal="uptrend_onset", bar_time=t,
                     price=0.1017,
                     detail={"label": "상승초입", "align": "역배열",
                             "ret_1h": 0.004, "ret_4h": -0.021, "ret_24h": 0.083})
    msg = format_events([up], [], {})
    assert "1h +0.4% · 4h -2.1% · 24h +8.3%" in msg

    # leader_break는 24h를 티커 값(gain_24h)으로 쓰고, 중복 표기하지 않는다
    lb = SignalEvent(symbol="COTIUSDT", signal="leader_break", bar_time=t,
                     price=0.0632,
                     detail={"label": "크립토 모멘텀 눌림목/이탈", "ma": 0.0641,
                             "ma_period": 20, "interval": "15m",
                             "gain_24h": 0.31, "ret_1h": -0.012, "ret_4h": -0.034,
                             "ret_24h": 0.99})
    msg = format_events([lb], [], {})
    assert "1h -1.2% · 4h -3.4% · 24h +31%" in msg
    assert msg.count("24h") == 1          # ret_24h(+99%)는 무시된다
    assert "15m 20SMA" in msg


def test_returns_tag_omits_missing_values():
    """구할 수 없는 구간은 빼고 나머지만 붙인다 — 0%로 채우지 않는다."""
    t = pd.Timestamp("2026-08-13T12:00:00Z")
    ev = SignalEvent(symbol="XUSDT", signal="uptrend_onset", bar_time=t, price=1.0,
                     detail={"label": "상승초입", "ret_1h": None, "ret_4h": 0.05,
                             "ret_24h": None})
    msg = format_events([ev], [], {})
    assert "4h +5.0%" in msg
    assert "1h " not in msg and "24h " not in msg
    # 셋 다 없으면 태그 자체가 안 붙는다
    bare = SignalEvent(symbol="YUSDT", signal="uptrend_onset", bar_time=t, price=1.0,
                       detail={"label": "상승초입"})
    assert "%" not in format_events([bare], [], {})


def test_returns_not_added_to_other_signals():
    """눌림목·하락전환 줄은 그대로 — 수익률 태그는 두 시그널에만 붙는다."""
    t = pd.Timestamp("2026-08-13T12:00:00Z")
    ev = SignalEvent(symbol="SOXL", signal="pullback", bar_time=t, price=22.0,
                     detail={"label": "눌림목", "ret_1h": 0.01, "ret_4h": 0.02})
    assert "1h " not in format_events([], [ev], {"SOXL": ""})


def _hold(symbol, gain=None, days_ago=0, extra=None):
    """추적 줄 하나 — 24h 수익률과 발생 시각을 따로 준다."""
    d = {"label": "상승초입", "last_price": 100.0}
    if gain is not None:
        d["gain_24h"] = gain
    d.update(extra or {})
    return SignalEvent(symbol=symbol, signal="uptrend_onset",
                       bar_time=pd.Timestamp.now(tz="UTC") - timedelta(days=days_ago),
                       price=100.0, detail=d)


def test_hold_list_sorted_by_daily_gain():
    """추적 리스트는 24h 수익률 내림차순 — 발생 시각 순이 아니다.

    가장 오래된 셋업(C)이 제일 잘 가고 있으면 맨 위로 온다. 시간순이면
    맨 아래에 묻혀서, 스무 줄 중 지금 먹히는 줄을 찾을 수 없다.
    """
    evs = [_hold("AUSDT", gain=0.01, days_ago=0),
           _hold("BUSDT", gain=-0.05, days_ago=1),
           _hold("CUSDT", gain=0.30, days_ago=5)]
    out = format_events([], [], {}, ongoing_crypto=evs)
    order = [ln.split()[1] for ln in out.splitlines() if ln.startswith("↳")]
    assert order == ["C", "A", "B"]
    assert "↳ C  100 · 5d · 24h +30%" in out    # 정렬 기준을 줄에도 적는다


def test_hold_line_without_gain_sinks_to_bottom():
    """24h를 못 구한 항목은 0%로 채우지 않고 맨 아래로 — 그 안에서는 최신 순."""
    evs = [_hold("NOGAINUSDT", gain=None, days_ago=3),
           _hold("FRESHUSDT", gain=None, days_ago=0),
           _hold("LOSSUSDT", gain=-0.20, days_ago=1)]
    out = format_events([], [], {}, ongoing_crypto=evs)
    order = [ln.split()[1] for ln in out.splitlines() if ln.startswith("↳")]
    assert order == ["LOSS", "FRESH", "NOGAIN"]
    assert "24h" not in [ln for ln in out.splitlines() if "NOGAIN" in ln][0]


def test_hold_line_falls_back_to_candle_return():
    """티커 값이 없으면 캔들 역산(ret_24h)을 쓴다 — 둘 다 있으면 티커 우선."""
    only_candle = _hold("XUSDT", gain=None, extra={"ret_24h": 0.08})
    both = _hold("YUSDT", gain=0.02, extra={"ret_24h": 0.99})
    out = format_events([], [], {}, ongoing_crypto=[only_candle, both])
    assert "↳ X  100 · 0d · 24h +8.0%" in out
    assert "↳ Y  100 · 0d · 24h +2.0%" in out     # 99%가 아니라 티커 값
    order = [ln.split()[1] for ln in out.splitlines() if ln.startswith("↳")]
    assert order == ["X", "Y"]


def _wave(symbol, stage, gain, days_ago=0):
    return SignalEvent(symbol=symbol, signal="wave_setup",
                       bar_time=pd.Timestamp.now(tz="UTC") - timedelta(days=days_ago),
                       price=1.0,
                       detail={"label": "파동", "stage": stage, "last_price": 1.0,
                               "gain_24h": gain, "fast_line": 0.9, "fast_st": "22×3"})


def test_wave_variants_are_grouped_not_interleaved():
    """파동 칸에서 ABC와 임펄스가 섞이지 않는다 — 블록으로 묶이고 각자 수익률 순."""
    evs = [_wave("AUSDT", "임펄스", 0.50), _wave("BUSDT", "ABC", 0.01),
           _wave("CUSDT", "임펄스", 0.10), _wave("DUSDT", "ABC", 0.30)]
    out = format_events(evs, [], {})
    order = [ln.split(">")[1].split("<")[0] for ln in out.splitlines() if ln.startswith("• ")]
    # ABC가 먼저(수익률 순 D→B), 그 다음 임펄스(A→C) — 50%인 A가 맨 위로 오지 않는다
    assert order == ["D", "B", "A", "C"]


def test_wave_tracking_groups_by_variant_too():
    """추적 줄도 같은 규칙 — 변형별로 묶고 그 안에서 24h 수익률 순."""
    evs = [_wave("AUSDT", "임펄스", 0.50, 1), _wave("BUSDT", "ABC", -0.20, 2),
           _wave("CUSDT", "임펄스", 0.10, 3), _wave("DUSDT", "ABC", 0.30, 0)]
    out = format_events([], [], {}, ongoing_crypto=evs)
    order = [ln.split()[1] for ln in out.splitlines() if ln.startswith("↳ ")]
    assert order == ["D", "B", "A", "C"]


def test_wave_line_shows_the_gain_it_is_sorted_by():
    """정렬 기준인 24h가 파동 줄에 나오고, 수퍼트렌드선 값은 안 나온다."""
    out = format_events([_wave("XUSDT", "ABC", 0.083)], [], {})
    line = [ln for ln in out.splitlines() if ln.startswith("• ")][0]
    assert "24h +8.3%" in line
    assert line.endswith("ABC 돌파")
    assert "0.9" not in line and "수퍼트렌드선" not in line


def test_non_wave_signals_keep_a_single_group():
    """파동이 아닌 칸은 변형 구분이 없으므로 순수 수익률 순 그대로다."""
    evs = [_hold("AUSDT", gain=0.01), _hold("BUSDT", gain=0.40), _hold("CUSDT", gain=-0.1)]
    out = format_events([], [], {}, ongoing_crypto=evs)
    order = [ln.split()[1] for ln in out.splitlines() if ln.startswith("↳ ")]
    assert order == ["B", "A", "C"]


def test_returns_tag_lists_every_available_span():
    """줄에 1h·4h·24h·7d가 순서대로 붙고, 없는 구간만 빠진다."""
    full = _event("XUSDT", {"label": "상승초입", "ret_1h": 0.01, "ret_4h": -0.02,
                            "ret_24h": 0.13, "ret_7d": 0.42})
    line = [ln for ln in format_events([full], [], {}).splitlines()
            if ln.startswith("• ")][0]
    assert "1h +1.0% · 4h -2.0% · 24h +13% · 7d +42%" in line

    partial = _event("YUSDT", {"label": "상승초입", "ret_24h": 0.05})
    line = [ln for ln in format_events([partial], [], {}).splitlines()
            if ln.startswith("• ")][0]
    assert "24h +5.0%" in line
    assert "7d " not in line and "1h " not in line


def test_only_wave_hold_lines_carry_the_full_span_set():
    """구간 세트는 파동 추적 줄에만 붙는다.

    다른 시그널의 ret_4h·ret_7d는 트리거 봉에 얼어붙은 값이라 며칠 지난
    추적 줄에 실으면 묵은 숫자가 나간다 — 24h만 매 스캔 갱신된다.
    """
    wave = _wave("XUSDT", "ABC", 0.13)
    wave.detail.update({"ret_4h": -0.02, "ret_7d": 0.42})
    assert "4h -2.0% · 24h +13% · 7d +42%" in format_events(
        [], [], {}, ongoing_crypto=[wave])

    other = _hold("YUSDT", gain=0.13)
    other.detail.update({"ret_4h": -0.02, "ret_7d": 0.42})
    line = [ln for ln in format_events([], [], {}, ongoing_crypto=[other]).splitlines()
            if ln.startswith("↳ ")][0]
    # "24h"가 "4h"를 포함하므로 구분자까지 넣어서 본다
    assert "24h +13%" in line
    assert "·4h " not in line and " 4h " not in line
    assert "7d " not in line


def test_wave_tracking_gets_one_subheader_per_variant():
    """변형 이름은 소제목에 한 번만 — 줄마다 반복하지 않는다."""
    evs = [_wave("AUSDT", "ABC", 0.20), _wave("BUSDT", "ABC", 0.10),
           _wave("CUSDT", "임펄스", 0.30), _wave("DUSDT", "임펄스", 0.05)]
    out = format_events([], [], {}, ongoing_crypto=evs)
    body = [ln for ln in out.splitlines() if ln.startswith("↳ ") or "ⓐ" in ln or "ⓑ" in ln]
    assert body[0].strip().startswith("ⓐ") and "4h 돌파" in body[0]
    assert [ln.split()[1] for ln in body[1:3]] == ["A", "B"]
    assert body[3].strip().startswith("ⓑ") and "일봉 터치" in body[3]
    assert [ln.split()[1] for ln in body[4:6]] == ["C", "D"]
    # 소제목이 말해 주므로 추적 줄에는 변형 이름이 없다
    assert "ABC 돌파" not in "\n".join(body[1:3])
    assert out.count("ⓐ") == 1 and out.count("ⓑ") == 1


def test_hold_lines_use_one_separator_everywhere():
    """추적 줄 구분자는 신규 줄과 같은 ' · ' — 붙여 쓴 '·'가 섞이지 않는다."""
    e = _hold("XUSDT", gain=0.13)
    e.detail["align"] = "역배열"
    line = [ln for ln in format_events([], [], {}, ongoing_crypto=[e]).splitlines()
            if ln.startswith("↳ ")][0]
    assert line == "↳ X  100 · 0d · 24h +13% · 역배열"
