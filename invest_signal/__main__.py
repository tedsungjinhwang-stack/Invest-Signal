"""CLI 진입점: python -m invest_signal [--only crypto|etf] [--dry-run]

`--history N`을 주면 스캔하지 않고 최근 발송된 알림 원문 N건을 출력한다.
"""

import argparse
import re
import sys
from datetime import timedelta, timezone

from . import scanner, sent_log

KST = timezone(timedelta(hours=9))


def _plain(text: str) -> str:
    """텔레그램 HTML 태그를 벗겨 터미널에서 읽기 좋게."""
    text = re.sub(r'<a href="[^"]*">([^<]*)</a>', r"\1", text)
    return re.sub(r"</?[a-z]+>", "", text)


def show_history(state_path: str, limit: int, raw: bool = False) -> int:
    rows = sent_log.recent(sent_log.default_path(state_path), limit)
    if not rows:
        print("발송 기록 없음 — 아직 알림이 나가지 않았거나 30일이 지나 정리됐다.")
        return 0
    from datetime import datetime
    for i, r in enumerate(rows):
        ts = datetime.fromisoformat(r["ts"]).astimezone(KST)
        head = f"═══ {ts:%Y-%m-%d %H:%M} KST · {r.get('mode', '?')}"
        c = r.get("counts") or {}
        if c:
            head += " · " + " ".join(f"{k} {v}" for k, v in c.items() if v)
        print(head)
        print(r["text"] if raw else _plain(r["text"]))
        if i != len(rows) - 1:
            print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="invest_signal", description="4h 시그널 스캐너")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--state", default="state/alerts_state.json")
    p.add_argument("--only", choices=["crypto", "etf"], default=None,
                   help="한쪽만 스캔 (기본: 둘 다)")
    p.add_argument("--dry-run", action="store_true",
                   help="알림 발송·상태 저장 없이 결과만 출력")
    p.add_argument("--intrabar", action="store_true",
                   help="진행 중 봉을 포함해 인트라바 판정 가능한 시그널만 스캔 (크립토 전용)")
    p.add_argument("--history", nargs="?", type=int, const=5, default=None,
                   metavar="N", help="스캔 대신 최근 발송 알림 N건 출력 (기본 5)")
    p.add_argument("--raw", action="store_true", help="--history에서 HTML 태그 그대로")
    args = p.parse_args(argv)
    if args.history is not None:
        return show_history(args.state, args.history, args.raw)
    return scanner.run(args.config, args.state, only=args.only,
                       dry_run=args.dry_run, intrabar=args.intrabar)


if __name__ == "__main__":
    sys.exit(main())
