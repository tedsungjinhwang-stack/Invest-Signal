"""스캔 간에 넘겨야 하는 상태 저장.

  · alerts  — 이미 보낸 (종목|시그널|봉시각). 중복 알림 방지용.
  · leaders — 크립토 모멘텀 눌림목/이탈이 상위권으로 뽑았던 종목의 마지막 등재
              시각과 **그 감시 창에서 찍은 최고 순위**. 순위 밖으로 밀려나도
              일정 기간 계속 감시하기 위해 기억한다.

              값은 `{"t": 시각, "best": 순위}`인데, 예전 파일은 시각 문자열
              하나였다. 읽을 때 둘 다 받는다 — 상태 파일은 레포에 커밋돼
              있어서 배포 순간에 옛 모양이 그대로 들어온다.
"""

import json
import os
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 30
LEADER_DAYS = 5          # 상위권에서 밀려난 뒤에도 계속 볼 기간 (config의 watch_days와 같이 둔다)


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
        """이번 스캔 상위권 종목의 등재 시각과 최고 순위를 갱신.

        **symbols는 순위 순서대로 넘겨야 한다** — 위치가 곧 순위다(1위부터).
        최고 순위는 감시 창 안에서 **더 좋았던 값을 유지한다**: 어제 1위였다가
        오늘 9위로 밀렸으면 1위를 기억한다. 창을 벗어나면 prune이 항목째
        지우므로 다음에 다시 들 때 새로 센다.
        """
        stamp = (when or datetime.now(timezone.utc)).isoformat()
        for rank, sym in enumerate(symbols, 1):
            best = self.best_rank(sym)
            self._leaders[sym] = {"t": stamp,
                                  "best": rank if best is None else min(best, rank)}

    @staticmethod
    def _leader_time(v):
        """옛 모양(시각 문자열)과 새 모양(dict) 둘 다 받는다."""
        raw = v.get("t") if isinstance(v, dict) else v
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None

    def best_rank(self, symbol: str) -> int | None:
        """감시 창 안에서 이 종목이 찍은 최고 순위. 모르면 None."""
        v = self._leaders.get(symbol)
        if not isinstance(v, dict):
            return None                 # 옛 기록엔 순위가 없다
        try:
            r = int(v.get("best"))
        except (TypeError, ValueError):
            return None
        return r if r > 0 else None

    def recent_leaders(self, days: int = LEADER_DAYS,
                       now: datetime | None = None) -> dict[str, datetime]:
        """최근 days일 안에 상위권이었던 종목 → 마지막 등재 시각."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        out = {}
        for s, v in self._leaders.items():
            t = self._leader_time(v)
            if t is not None and t >= cutoff:
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
