"""알림 중복 방지 상태 저장 — 이미 보낸 (종목|시그널|봉시각)을 기억한다."""

import json
import os
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 30


class AlertState:
    def __init__(self, path: str):
        self.path = path
        self._alerts: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._alerts = dict(data.get("alerts", {}))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            self._alerts = {}

    def is_new(self, key: str) -> bool:
        return key not in self._alerts

    def mark(self, key: str, when: datetime | None = None) -> None:
        when = when or datetime.now(timezone.utc)
        self._alerts[key] = when.isoformat()

    def prune(self, now: datetime | None = None) -> None:
        """오래된 기록 제거 — 파일이 무한정 자라지 않게."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=RETENTION_DAYS)
        kept = {}
        for k, v in self._alerts.items():
            try:
                if datetime.fromisoformat(v) >= cutoff:
                    kept[k] = v
            except ValueError:
                continue
        self._alerts = kept

    def save(self) -> None:
        self.prune()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"alerts": self._alerts}, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
