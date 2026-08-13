"""발송 메시지 기록(JSONL) 테스트."""

import json
from datetime import datetime, timedelta, timezone

from invest_signal import sent_log


def test_append_and_recent_roundtrip(tmp_path):
    p = str(tmp_path / "state" / "sent_log.jsonl")
    sent_log.append(p, "첫 번째 알림", mode="close", counts={"crypto": 2})
    sent_log.append(p, "두 번째 알림", mode="intrabar")

    rows = sent_log.recent(p)
    assert [r["text"] for r in rows] == ["두 번째 알림", "첫 번째 알림"]   # 최신순
    assert rows[0]["mode"] == "intrabar"
    assert rows[1]["counts"] == {"crypto": 2}
    assert datetime.fromisoformat(rows[0]["ts"]).tzinfo is not None


def test_default_path_sits_next_to_state_file():
    assert sent_log.default_path("state/alerts_state.json") == "state/sent_log.jsonl"
    assert sent_log.default_path("alerts.json") == "./sent_log.jsonl"


def test_entries_older_than_retention_are_dropped(tmp_path):
    """30일 지난 기록은 다음 append에서 사라진다."""
    p = tmp_path / "sent_log.jsonl"
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=sent_log.RETENTION_DAYS + 1)).isoformat()
    keep = (now - timedelta(days=3)).isoformat()
    p.write_text("\n".join(json.dumps({"ts": t, "text": t}) for t in (old, keep)) + "\n",
                 encoding="utf-8")

    sent_log.append(str(p), "새 알림")
    texts = [r["text"] for r in sent_log.recent(str(p), limit=99)]
    assert texts == ["새 알림", keep]           # 31일 전 기록은 정리됨


def test_entry_count_is_capped(tmp_path):
    p = str(tmp_path / "sent_log.jsonl")
    for i in range(sent_log.MAX_ENTRIES + 20):
        sent_log.append(p, f"알림 {i}")
    rows = sent_log.recent(p, limit=10_000)
    assert len(rows) == sent_log.MAX_ENTRIES
    assert rows[0]["text"] == f"알림 {sent_log.MAX_ENTRIES + 19}"   # 최신은 남고
    assert all(r["text"] != "알림 0" for r in rows)                 # 가장 오래된 건 밀려남


def test_corrupt_lines_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "sent_log.jsonl"
    good = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "text": "정상"})
    p.write_text(f"{good}\n{{깨진 줄\n\n", encoding="utf-8")
    sent_log.append(str(p), "그 다음")
    assert [r["text"] for r in sent_log.recent(str(p))] == ["그 다음", "정상"]


def test_append_failure_does_not_raise(tmp_path):
    """기록 실패가 스캔을 죽이면 안 된다 — 알림은 이미 나간 뒤다."""
    bad = str(tmp_path / "afile")
    (tmp_path / "afile").write_text("not a dir", encoding="utf-8")
    logs = []
    sent_log.append(f"{bad}/sub/log.jsonl", "무시됨", log=logs.append)
    assert logs and "기록 실패" in logs[0]


def test_recent_on_missing_file_is_empty(tmp_path):
    assert sent_log.recent(str(tmp_path / "nope.jsonl")) == []


def test_history_cli_prints_recent_messages(tmp_path, capsys):
    """--history는 스캔하지 않고 최근 발송분을 HTML 없이 출력한다."""
    from invest_signal.__main__ import main

    state = tmp_path / "state" / "alerts_state.json"
    p = sent_log.default_path(str(state))
    sent_log.append(p, '🚨 <b>4h 시그널</b>\n• <a href="http://x">KITE</a>  <b>0.1</b>',
                    mode="close", counts={"crypto": 1, "etf": 0})

    assert main(["--state", str(state), "--history", "3"]) == 0
    out = capsys.readouterr().out
    assert "🚨 4h 시그널" in out and "• KITE  0.1" in out
    assert "<b>" not in out and "href" not in out
    assert "close" in out and "crypto 1" in out
    assert "etf" not in out                      # 0인 항목은 헤더에서 뺀다

    assert main(["--state", str(state), "--history", "3", "--raw"]) == 0
    assert "<b>" in capsys.readouterr().out      # --raw는 태그 그대로


def test_history_cli_on_empty_log(tmp_path, capsys):
    from invest_signal.__main__ import main

    assert main(["--state", str(tmp_path / "s.json"), "--history"]) == 0
    assert "발송 기록 없음" in capsys.readouterr().out
