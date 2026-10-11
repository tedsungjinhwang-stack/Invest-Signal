"""🔥만 보기(notify.fire_only) — 🔥 줄과 🔻하락 CHoCH만 텔레그램에 싣는다."""

import json

import pandas as pd

from invest_signal import notify, scanner, sent_log
from invest_signal.signals import SignalEvent
from invest_signal.state import AlertState

T = pd.Timestamp("2026-09-30T04:00:00Z")


def _ev(sym, signal, fire=False, **detail):
    d = {"label": {"wave_setup": "파동", "vwap_onset": "상승초입",
                   "leader_break": notify.LEADER_LABEL, "choch_warn": "하락 CHoCH",
                   "spike_bar": "급등봉"}[signal], **detail}
    if fire:
        d["fire"] = True
    return SignalEvent(symbol=sym, signal=signal, bar_time=T, price=1.0, detail=d)


def test_fire_only_keeps_fire_rows_and_choch():
    evs = [_ev("AUSDT", "wave_setup", fire=True), _ev("BUSDT", "wave_setup"),
           _ev("CUSDT", "vwap_onset", fire=True), _ev("DUSDT", "leader_break"),
           _ev("EUSDT", "leader_break", triggers=["🔥진입", "1h단기선돌파"]),
           _ev("FUSDT", "choch_warn"), _ev("GUSDT", "spike_bar", fire=True)]
    got = [e.symbol for e in notify.fire_only(evs)]
    assert got == ["AUSDT", "CUSDT", "EUSDT", "FUSDT", "GUSDT"]      # 🚀급등봉은 항상
    assert [e.symbol for e in notify.fire_only(evs, ("choch_warn",))] == \
        ["AUSDT", "CUSDT", "EUSDT", "FUSDT"]


def test_wave_hold_line_shows_fire():
    e = _ev("AUSDT", "wave_setup", fire=True, stage="ABC", last_price=1.0)
    out = notify.format_events([], [], {}, [e])
    line = [ln for ln in out.splitlines() if ln.startswith("↳ A ")][0]
    assert "🔥" in line


def test_hidden_rows_roundtrip(tmp_path):
    p = str(tmp_path / "sent_log.jsonl")
    rows = sent_log.hidden_rows([_ev("BUSDT", "wave_setup", stage="ABC"),
                                 _ev("DUSDT", "leader_break", triggers=["1h단기선돌파"],
                                     rank=3)])
    sent_log.append(p, "", mode="hidden", hidden=rows)
    got = json.loads(open(p, encoding="utf-8").read())
    assert got["mode"] == "hidden"
    assert got["hidden"][0] == {"symbol": "BUSDT", "signal": "wave_setup",
                                "bar_time": T.isoformat(), "price": 1.0,
                                "label": "파동", "stage": "ABC"}
    assert got["hidden"][1]["triggers"] == ["1h단기선돌파"] and got["hidden"][1]["rank"] == 3


def _run(monkeypatch, tmp_path, crypto, ongoing, fire_only=True):
    cfg = {"notify": {"fire_only": fire_only}}
    monkeypatch.setattr(scanner.cfg_mod, "load", lambda p: cfg)
    monkeypatch.setattr(scanner, "scan_crypto",
                        lambda *a, **k: (crypto, ongoing, [{"symbol": "ZUSDT", "rank": 1,
                                                            "gain_24h": 0.5}]))
    etf = SignalEvent(symbol="SOXL", signal="pullback", bar_time=T, price=1.0,
                      detail={"label": "눌림목"})
    monkeypatch.setattr(scanner, "scan_yfinance", lambda *a, **k: ([etf], [], {}, {}))
    monkeypatch.setattr(scanner, "_scan_community", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(notify, "_community_lines", lambda c: ["📣 커뮤니티"] if c else [])
    sent = []
    monkeypatch.setattr(notify, "send_telegram", lambda msg, log=print: sent.append(msg) or True)
    state_path = str(tmp_path / "state" / "alerts_state.json")
    scanner.run("x.yaml", state_path, log=lambda *a: None)
    return sent, AlertState(state_path), sent_log.default_path(state_path)


def test_run_sends_only_fire_and_choch(monkeypatch, tmp_path):
    fire_w = _ev("AUSDT", "wave_setup", fire=True)
    plain_w = _ev("BUSDT", "wave_setup")
    choch = _ev("FUSDT", "choch_warn", broken_low=1.1, from_high=-0.05)
    hold_fire = _ev("HUSDT", "leader_break", fire=True, last_price=1.0, watch_days=2)
    hold_plain = _ev("IUSDT", "leader_break", last_price=1.0, watch_days=2)
    sent, state, log_path = _run(monkeypatch, tmp_path, [fire_w, plain_w, choch],
                                 [hold_fire, hold_plain])
    assert len(sent) == 1
    msg = sent[0]
    assert ">A<" in msg and ">F<" in msg and "\n↳ H " in msg
    assert ">B<" not in msg and "\n↳ I " not in msg and "SOXL" not in msg
    assert "📣" not in msg and "4h 시그널" not in msg
    assert msg.startswith("🔥 <b>떡상조짐</b> · 🔻 <b>하락 CHoCH</b> · ")
    assert "24h 상승률 TOP" in msg and ">Z<" in msg      # 5위까지 순위표는 그대로
    # 뺀 줄은 상태에 안 남는다 — 🔥가 나중에 붙으면 그때 나가야 한다
    assert not state.is_new(fire_w.dedup_key) and not state.is_new(choch.dedup_key)
    assert state.is_new(plain_w.dedup_key)
    row = json.loads(open(log_path, encoding="utf-8").read().splitlines()[-1])
    assert {h["symbol"] for h in row["hidden"]} == {"BUSDT", "SOXL"}


def test_run_sends_nothing_when_no_fire(monkeypatch, tmp_path):
    sent, state, log_path = _run(monkeypatch, tmp_path, [_ev("BUSDT", "wave_setup")], [])
    assert sent == []
    row = json.loads(open(log_path, encoding="utf-8").read().splitlines()[-1])
    assert row["mode"] == "hidden" and row["hidden"][0]["symbol"] == "BUSDT"


def test_run_fire_only_off_sends_everything(monkeypatch, tmp_path):
    sent, _, _ = _run(monkeypatch, tmp_path, [_ev("BUSDT", "wave_setup")], [],
                      fire_only=False)
    msg = sent[0] if len(sent) == 1 else ""
    assert ">B<" in msg and "SOXL" in msg and "📣" in msg and "24h 상승률 TOP" in msg


def test_fire_only_title_names_what_is_inside():
    fire = _ev("AUSDT", "wave_setup", fire=True)
    choch = _ev("FUSDT", "choch_warn")
    assert notify.fire_only_title([fire]) == "🔥 <b>떡상조짐</b>"
    assert notify.fire_only_title([choch]) == "🔻 <b>하락 CHoCH</b>"
    assert notify.fire_only_title([choch, fire]) == "🔥 <b>떡상조짐</b> · 🔻 <b>하락 CHoCH</b>"
    spike = _ev("GUSDT", "spike_bar")
    assert notify.fire_only_title([choch, spike, fire]) == \
        "🔥 <b>떡상조짐</b> · 🚀 <b>급등봉</b> · 🔻 <b>하락 CHoCH</b>"
    assert notify.fire_only_title([spike], ("choch_warn",)) == "🔥 <b>떡상조짐</b>"


def test_run_sends_spike_bar_without_fire(monkeypatch, tmp_path):
    """🚀급등봉은 🔥 없이도 나간다 — 새 줄(•)·추적 줄(↳) 모두."""
    spike = _ev("SUSDT", "spike_bar", body=0.09, vol_mult=12.0)
    spike_hold = _ev("TUSDT", "spike_bar", body=0.08, vol_mult=9.0, since=0.03,
                     last_price=1.0)
    plain_w = _ev("BUSDT", "wave_setup")
    sent, state, _ = _run(monkeypatch, tmp_path, [spike, plain_w], [spike_hold])
    msg = sent[0] if len(sent) == 1 else ""
    assert msg.startswith("🚀 <b>급등봉</b> · ")
    assert ">S<" in msg and "몸통 +9.0%" in msg and "\n↳ T " in msg
    assert ">B<" not in msg
    assert not state.is_new(spike.dedup_key)


def test_daily_wave_lines_pass_without_fire():
    """🌊 일봉 단기·장기 추세선 줄은 🔥 없이도 나간다(10-04). 4h 파동은 여전히 🔥만."""
    d1 = SignalEvent(symbol="DUSDT", signal="wave_setup", bar_time=T, price=1.0,
                     detail={"label": "파동", "stage": "일봉ABC", "interval": "1d",
                             "touched": "단기선", "kind": "돌파"})
    d2 = SignalEvent(symbol="EUSDT", signal="wave_setup", bar_time=T, price=1.0,
                     detail={"label": "파동", "stage": "일봉장기선돌파", "interval": "1d"})
    h4 = SignalEvent(symbol="FUSDT", signal="wave_setup", bar_time=T, price=1.0,
                     detail={"label": "파동", "stage": "ABC", "interval": "4h",
                             "touched": "단기선", "kind": "돌파"})
    assert [e.symbol for e in notify.fire_only([d1, d2, h4])] == ["DUSDT", "EUSDT"]
    assert notify.fire_only([d1], ("spike_bar",)) == []                   # 설정에서 빼면 안 나간다
    assert notify.fire_only_title([d1, d2]) == "🌊 <b>일봉 파동</b>"


def test_pullback_trigger_passes_fire_only_without_fire():
    e = _ev("PUSDT", "leader_break", triggers=["🎯눌림목타점"])
    plain = _ev("QUSDT", "leader_break", triggers=["4h단기선터치"])
    always = ("spike_bar", "leader_break:🎯눌림목타점")
    assert [x.symbol for x in notify.fire_only([e, plain], always)] == ["PUSDT"]
    assert notify.fire_only_title([e], always) == "🎯 <b>눌림목 타점</b>"


def test_leader_only_triggers_keeps_just_pullback_rows(monkeypatch, tmp_path):
    """10-11 — ⚡는 🎯눌림목타점 새 줄만. ⚡ 추적 줄·순위표는 빼고 다른 칸은 그대로."""
    pb = _ev("PUSDT", "leader_break", triggers=["🎯눌림목타점", "3파진입"])
    fire_lb = _ev("FUSDT", "leader_break", fire=True, triggers=["4h단기선터치"])
    wave = _ev("WUSDT", "wave_setup", fire=True)
    hold = _ev("HUSDT", "leader_break", fire=True, last_price=1.0, watch_days=2)
    cfg = {"notify": {"fire_only": True, "leader_only_triggers": ["🎯눌림목타점"],
                      "always_show": ["leader_break:🎯눌림목타점"]}}
    monkeypatch.setattr(scanner.cfg_mod, "load", lambda p: cfg)
    monkeypatch.setattr(scanner, "scan_crypto",
                        lambda *a, **k: ([pb, fire_lb, wave], [hold],
                                         [{"symbol": "ZUSDT", "rank": 1, "gain_24h": 0.5}]))
    monkeypatch.setattr(scanner, "scan_yfinance", lambda *a, **k: ([], [], {}, {}))
    monkeypatch.setattr(scanner, "_scan_community", lambda *a, **k: {})
    sent = []
    monkeypatch.setattr(notify, "send_telegram", lambda msg, log=print: sent.append(msg) or True)
    state_path = str(tmp_path / "state" / "alerts_state.json")
    scanner.run("x.yaml", state_path, log=lambda *x: None)
    msg = sent[0]
    assert ">P<" in msg and ">W<" in msg
    assert ">F<" not in msg and "↳ H " not in msg and "24h 상승률 TOP" not in msg
    st = AlertState(state_path)
    assert not st.is_new(pb.dedup_key) and st.is_new(fire_lb.dedup_key)


def test_onset_in_always_show_passes_without_fire():
    """10-11 — 🟢상승초입은 🔥 없이도 싣고 🔥는 표시만."""
    plain = _ev("OUSDT", "vwap_onset")
    fire = _ev("PUSDT", "vwap_onset", fire=True)
    always = ("spike_bar", "vwap_onset")
    assert [e.symbol for e in notify.fire_only([plain, fire], always)] == ["OUSDT", "PUSDT"]
    assert notify.fire_only_title([plain], always) == "🟢 <b>상승초입</b>"
    assert notify.fire_only_title([plain, fire], always) == "🔥 <b>떡상조짐</b> · 🟢 <b>상승초입</b>"


def test_hide_holds_drops_tracked_lines_only(monkeypatch, tmp_path):
    """10-11 — notify.hide_holds 칸은 • 새 줄만 싣고 ↳ 추적 줄은 뺀다."""
    new = _ev("NUSDT", "vwap_onset")
    hold = _ev("HUSDT", "vwap_onset", last_price=1.0)
    w4 = _ev("WUSDT", "wave_setup", fire=True, stage="ABC", last_price=1.0)
    wd = _ev("DUSDT", "wave_setup", stage="일봉ABC", interval="1d", last_price=1.0)
    sp = _ev("SUSDT", "spike_bar", body=0.09, vol_mult=12.0, since=0.01, last_price=1.0)
    cfg = {"notify": {"fire_only": True,
                      "always_show": ["spike_bar", "vwap_onset", "wave_setup:일봉ABC"],
                      "hide_holds": ["vwap_onset", "wave_setup:ABC"]}}
    monkeypatch.setattr(scanner.cfg_mod, "load", lambda p: cfg)
    monkeypatch.setattr(scanner, "scan_crypto", lambda *a, **k: ([new], [hold, w4, wd, sp], []))
    monkeypatch.setattr(scanner, "scan_yfinance", lambda *a, **k: ([], [], {}, {}))
    monkeypatch.setattr(scanner, "_scan_community", lambda *a, **k: {})
    sent = []
    monkeypatch.setattr(notify, "send_telegram", lambda msg, log=print: sent.append(msg) or True)
    scanner.run("x.yaml", str(tmp_path / "state" / "alerts_state.json"), log=lambda *x: None)
    msg = sent[0]
    assert ">N<" in msg and "↳ D " in msg and "↳ S " in msg
    assert "↳ H " not in msg and "↳ W " not in msg
