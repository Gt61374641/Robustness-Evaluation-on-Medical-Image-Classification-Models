"""Safely archive non-paper, git-ignored artifacts into _archive/.

Moves (never deletes) legacy diagnostic figures, batch run logs, cloud packages,
and stale figure-data backups out of the working tree so the project root and
figures/ stay focused on the authoritative, paper-facing products.

HARD SAFETY RULES
  - Only ever MOVE; nothing is deleted.
  - A hard-protected allowlist can never be touched: results/, checkpoints/,
    configs/, src/, scripts/, tests/, and the authoritative figure dirs
    (figures/main, figures/data, figures/paper_tables, figures/at_ladder,
    figures/complexity, figures/decision_boundary, figures/gradcam,
    figures/sci_clean). The one exception is a single explicitly-named stale
    backup file under figures/data (the orphan at_ladder_h2 backup).
  - Dry-run by default; pass --apply to actually move.
  - Existing destinations are skipped, never overwritten (re-runnable).

Usage:
    python scripts/archive_nonpaper_artifacts.py           # dry-run (preview)
    python scripts/archive_nonpaper_artifacts.py --apply    # perform the moves
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "_archive"

# Directories whose contents must never be moved by this script.
HARD_PROTECTED = [
    "results", "checkpoints", "configs", "src", "scripts", "tests",
    "figures/main", "figures/data", "figures/paper_tables", "figures/at_ladder",
    "figures/complexity", "figures/decision_boundary", "figures/gradcam",
    "figures/sci_clean",
]
# The only file allowed to leave a hard-protected dir (stale orphan backup).
PROTECTED_EXCEPTIONS = {"figures/data/at_ladder_h2.orphan.bak.json"}


def _rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def _is_protected(src: Path) -> bool:
    rel = _rel(src)
    if rel in PROTECTED_EXCEPTIONS:
        return False
    return any(rel == d or rel.startswith(d + "/") for d in HARD_PROTECTED)


def build_plan():
    """Return list of (source_path, dest_path) moves. Sources may not exist."""
    moves = []

    # 1. Legacy per-model diagnostic figure dirs (superseded by main/ + tables).
    for name in ("sci", "sci_defense"):
        moves.append((ROOT / "figures" / name,
                      ARCHIVE / "figures_legacy" / name))

    # 2. Batch run logs (top-level only).
    for log in sorted(ROOT.glob("*.log")):
        moves.append((log, ARCHIVE / "run_logs" / log.name))

    # 3. Cloud result/figure packages (top-level only).
    for tgz in sorted(ROOT.glob("*.tgz")):
        moves.append((tgz, ARCHIVE / "cloud_packages" / tgz.name))

    # 4. Stale orphan figure-data backup (explicit single file).
    moves.append((ROOT / "figures" / "data" / "at_ladder_h2.orphan.bak.json",
                  ARCHIVE / "old_figure_data" / "at_ladder_h2.orphan.bak.json"))

    return moves


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="perform the moves (default is a dry-run preview)")
    args = ap.parse_args()

    plan = build_plan()
    to_move, skipped, total = [], [], 0
    for src, dest in plan:
        if not src.exists():
            continue
        # Absolute safety gate: refuse to touch anything protected.
        if _is_protected(src):
            print(f"REFUSE (protected): {_rel(src)}", file=sys.stderr)
            return 2
        if dest.exists():
            skipped.append((src, dest))
            continue
        to_move.append((src, dest))
        total += _size(src)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] archive plan -> {_rel(ARCHIVE)}/\n")
    if not to_move and not skipped:
        print("nothing to archive (already clean).")
        return 0

    for src, dest in to_move:
        kind = "dir " if src.is_dir() else "file"
        print(f"  MOVE {kind} {_rel(src):55} -> {_rel(dest)}")
    for src, dest in skipped:
        print(f"  SKIP (dest exists) {_rel(src)}")

    print(f"\n  {len(to_move)} item(s), {total / 1e6:.1f} MB to move; "
          f"{len(skipped)} skipped.")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to move.")
        return 0

    for src, dest in to_move:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        print(f"  moved {_rel(src)}")
    print(f"\nDone. Moved {len(to_move)} item(s) into {_rel(ARCHIVE)}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
