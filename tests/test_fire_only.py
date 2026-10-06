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


def test_run_st30_only_sends_just_30m_rows(monkeypatch, tmp_path):
    """notify.st30_only — 30m🔓단기선돌파 줄만(⚡ 🆕30m단기선돌파 포함). 순위표·ETF·커뮤니티도 뺀다."""
    a = _ev("AUSDT", "wave_setup", fire=True, st30=True)
    b = _ev("BUSDT", "wave_setup", fire=True)                       # 🔥여도 30m 아니면 뺀다
    c = _ev("CUSDT", "leader_break", triggers=["30m단기선돌파"])
    d = _ev("DUSDT", "choch_warn", broken_low=1.1, from_high=-0.05)  # 경고도 뺀다
    h = _ev("HUSDT", "vwap_onset", st30=True, last_price=1.0)
    cfg = {"notify": {"st30_only": True, "fire_only": True}}
    monkeypatch.setattr(scanner.cfg_mod, "load", lambda p: cfg)
    monkeypatch.setattr(scanner, "scan_crypto",
                        lambda *a_, **k: ([a, b, c, d], [h], [{"symbol": "ZUSDT", "rank": 1,
                                                                "gain_24h": 0.5}]))
    monkeypatch.setattr(scanner, "scan_yfinance", lambda *a_, **k: ([], [], {}, {}))
    monkeypatch.setattr(scanner, "_scan_community", lambda *a_, **k: {"x": 1})
    monkeypatch.setattr(notify, "_community_lines", lambda c_: ["📣 커뮤니티"] if c_ else [])
    sent = []
    monkeypatch.setattr(notify, "send_telegram", lambda msg, log=print: sent.append(msg) or True)
    state_path = str(tmp_path / "state" / "alerts_state.json")
    scanner.run("x.yaml", state_path, log=lambda *x: None)
    msg = sent[0]
    assert msg.startswith("🔓 <b>30m 단기선 돌파</b> · ")
    assert ">A<" in msg and ">C<" in msg and "\n↳ H " in msg
    assert ">B<" not in msg and ">D<" not in msg and "24h 상승률 TOP" not in msg and "📣" not in msg
    st = AlertState(state_path)
    assert not st.is_new(a.dedup_key) and st.is_new(b.dedup_key) and st.is_new(d.dedup_key)
