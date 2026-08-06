"""스캔 간에 넘겨야 하는 상태 저장.

  · alerts  — 이미 보낸 (종목|시그널|봉시각). 중복 알림 방지용.
  · leaders — 크립토 모멘텀 눌림목/이탈이 상위권으로 뽑았던 종목의 마지막 등재 시각.
              순위 밖으로 밀려나도 일정 기간 계속 감시하기 위해 기억한다.
"""

import json
import os
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 30
LEADER_DAYS = 7          # 상위권에서 밀려난 뒤에도 계속 볼 기간


class AlertState:
    def __init__(self, path: str):
        self.path = path
        self._alerts: dict[str, str] = {}
        self._leaders: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._alerts = dict(data.get("alerts", {}))
            self._leaders = dict(data.get("leaders", {}))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            self._alerts = {}
            self._leaders = {}

    def touch_leaders(self, symbols, when: datetime | None = None) -> None:
        """이번 스캔에서 상위권에 든 종목의 마지막 등재 시각을 갱신."""
        stamp = (when or datetime.now(timezone.utc)).isoformat()
        for s in symbols:
            self._leaders[s] = stamp

    def recent_leaders(self, days: int = LEADER_DAYS,
                       now: datetime | None = None) -> dict[str, datetime]:
        """최근 days일 안에 상위권이었던 종목 → 마지막 등재 시각."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        out = {}
        for s, v in self._leaders.items():
            try:
                t = datetime.fromisoformat(v)
            except (ValueError, TypeError):
                continue
            if t >= cutoff:
                out[s] = t
        return out

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
        fresh = self.recent_leaders(now=now)
        self._leaders = {s: v for s, v in self._leaders.items() if s in fresh}

    def save(self) -> None:
        self.prune()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"alerts": self._alerts, "leaders": self._leaders},
                      f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
