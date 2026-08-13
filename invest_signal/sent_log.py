"""발송한 텔레그램 메시지 원문 기록 (JSONL).

상태 파일(alerts_state.json)에는 `종목|시그널|봉시각`만 남아서 "무엇이
언제 잡혔나"까지만 알 수 있다. 가격·수익률 태그·📈순위표·추적(↳) 줄은
렌더링 결과라 어디에도 안 남는다 — 나중에 "그때 알림이 뭐라고 왔었지"를
확인하려면 원문이 필요해서 따로 남긴다.

워크플로의 커밋 스텝이 `git add state/`로 디렉터리째 담으므로, 이 파일을
state 디렉터리에 두면 별도 설정 없이 레포에 함께 올라간다.
"""

import json
import os
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 30
MAX_ENTRIES = 500        # 일수 안이어도 이 개수를 넘으면 오래된 것부터 버린다


def default_path(state_path: str) -> str:
    """상태 파일 옆에 sent_log.jsonl — 같은 커밋에 함께 올라가도록."""
    return os.path.join(os.path.dirname(state_path) or ".", "sent_log.jsonl")


def _rows(path: str) -> list[dict]:
    """기존 기록 로드. 깨진 줄은 건너뛴다 — 로그 하나 때문에 스캔이 죽으면 안 된다."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out


def _fresh(rows: list[dict], now: datetime) -> list[dict]:
    cutoff = now - timedelta(days=RETENTION_DAYS)
    kept = []
    for r in rows:
        try:
            if datetime.fromisoformat(r["ts"]) >= cutoff:
                kept.append(r)
        except (KeyError, ValueError, TypeError):
            continue
    return kept[-MAX_ENTRIES:]


def append(path: str, text: str, mode: str = "close", counts: dict | None = None,
           now: datetime | None = None, log=print) -> None:
    """발송된 메시지 한 건을 덧붙이고 오래된 기록을 정리한다.

    기록 실패가 스캔을 실패시키면 안 되므로 예외는 로그만 남기고 삼킨다 —
    알림은 이미 나갔고, 남는 건 사후 조회용 사본일 뿐이다.
    """
    now = now or datetime.now(timezone.utc)
    row = {"ts": now.isoformat(), "mode": mode, "text": text}
    if counts:
        row["counts"] = counts
    try:
        rows = _fresh(_rows(path) + [row], now)   # 새 줄까지 넣고 잘라야 상한이 정확
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError as exc:
        log(f"[sent_log] 기록 실패 — 건너뜀 ({exc})")


def recent(path: str, limit: int = 10) -> list[dict]:
    """최근 발송분을 최신순으로. 조회용 헬퍼."""
    return list(reversed(_rows(path)))[:limit]
