"""Merge cloud-downloaded results/checkpoints into the local tree.

The cloud backfill run (seed43/44 defenses etc.) is packed as a tarball whose
paths mirror the repo layout (results/..., checkpoints/...). Extracting it
straight onto the repo would overwrite already-validated local seed42 files.
This script instead stages the archive and copies over ONLY files that are new
locally, so existing results are never clobbered unless you pass --force.

Usage:
  python scripts/merge_cloud_results.py results_multiseed.tgz         # dry-run report
  python scripts/merge_cloud_results.py results_multiseed.tgz --apply # actually copy
  python scripts/merge_cloud_results.py path/to/extracted_dir --apply # from a folder
  python scripts/merge_cloud_results.py results_multiseed.tgz --apply --force  # overwrite too
"""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MERGE_TOP = ("results", "checkpoints")  # only these trees are merged


def stage(src: Path, tmp: Path) -> Path:
    """Return a directory that contains results/ and/or checkpoints/ subtrees."""
    if src.is_dir():
        return src
    if src.suffix in (".tgz", ".gz") or src.name.endswith(".tar.gz") or src.suffix == ".tar":
        print(f"==> extracting {src.name} to staging ...")
        with tarfile.open(src) as t:
            t.extractall(tmp)
        return tmp
    raise SystemExit(f"[abort] not a directory or tarball: {src}")


def merge(staged: Path, apply: bool, force: bool):
    added, skipped, overwritten = [], [], []
    for top in MERGE_TOP:
        base = staged / top
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(staged)          # e.g. results/.../seed43/x.json
            dest = ROOT / rel
            if dest.exists():
                if force:
                    overwritten.append(rel)
                    if apply:
                        shutil.copy2(f, dest)
                else:
                    skipped.append(rel)
            else:
                added.append(rel)
                if apply:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
    return added, skipped, overwritten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="downloaded .tgz OR an extracted folder")
    ap.add_argument("--apply", action="store_true", help="actually copy (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist locally")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    if not src.exists():
        raise SystemExit(f"[abort] source not found: {src}")

    with tempfile.TemporaryDirectory() as td:
        staged = stage(src, Path(td))
        added, skipped, overwritten = merge(staged, args.apply, args.force)

    def show(title, items, cap=40):
        print(f"\n{title}: {len(items)}")
        for r in sorted(items)[:cap]:
            print(f"  {r}")
        if len(items) > cap:
            print(f"  ... (+{len(items) - cap} more)")

    show("NEW (copied)" if args.apply else "NEW (would copy)", added)
    show("OVERWRITTEN" if args.apply else "WOULD OVERWRITE (needs --force)",
         overwritten if args.force else skipped)

    mode = "APPLIED" if args.apply else "DRY-RUN (nothing written; add --apply)"
    print(f"\n==> {mode}. new={len(added)} "
          f"{'overwritten' if args.force else 'skipped-existing'}="
          f"{len(overwritten) if args.force else len(skipped)}")


if __name__ == "__main__":
    main()
