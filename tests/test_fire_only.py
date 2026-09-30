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
    assert got == ["AUSDT", "CUSDT", "EUSDT", "FUSDT"]


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
    assert "📣" not in msg and "24h 상승률 TOP" not in msg
    assert "하락 CHoCH" in msg
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
