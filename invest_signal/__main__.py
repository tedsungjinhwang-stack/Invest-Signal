"""CLI 진입점: python -m invest_signal [--only crypto|etf] [--dry-run]"""

import argparse
import sys

from . import scanner


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="invest_signal", description="4h 시그널 스캐너")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--state", default="state/alerts_state.json")
    p.add_argument("--only", choices=["crypto", "etf"], default=None,
                   help="한쪽만 스캔 (기본: 둘 다)")
    p.add_argument("--dry-run", action="store_true",
                   help="알림 발송·상태 저장 없이 결과만 출력")
    args = p.parse_args(argv)
    return scanner.run(args.config, args.state, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
