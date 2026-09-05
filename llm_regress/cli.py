from __future__ import annotations

import argparse
import os
import sys

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-regress")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("baseline", help="run eval and store baseline")
    sub.add_parser("run", help="run eval and compare against baseline")
    sub.add_parser("report", help="write latest report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"command not implemented yet: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
