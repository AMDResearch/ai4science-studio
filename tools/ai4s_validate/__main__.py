from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai4s_validate.discover import discover_models, repo_root_from
from ai4s_validate.runner import fix as do_fix
from ai4s_validate.runner import run_all, run_fast


def _print_text(result, *, strict: bool) -> None:
    if not result.findings:
        print("ok — no findings")
        return
    for f in result.findings:
        loc = f.file or ""
        if f.line:
            loc = f"{loc}:{f.line}"
        model = f" [{f.model}]" if f.model else ""
        print(f"{f.level.upper():7} {f.code:28} {loc}{model}: {f.message}")
    n_err = len(result.errors)
    n_warn = len(result.warnings)
    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    if not result.ok(strict=strict):
        hint = " (strict: warnings are fatal)" if strict else ""
        print(f"FAILED{hint}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai4s_validate",
        description="Validate AI4Science Studio manifests, index, scripts, and docs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="check",
        choices=("check", "check-fast", "fix"),
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip executing preflight_*.py --dry-run",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve() if args.root else repo_root_from()

    if args.command == "fix":
        path = do_fix(root)
        print(f"wrote {path}")
        return 0

    if args.command == "check-fast":
        result = run_fast(root)
    else:
        result = run_all(root, preflight=not args.no_preflight)

    if args.json:
        payload = {
            "ok": result.ok(strict=args.strict),
            "models": [m.slug for m in discover_models(root)],
            "findings": [f.to_dict() for f in result.findings],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_text(result, strict=args.strict)

    return 0 if result.ok(strict=args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
