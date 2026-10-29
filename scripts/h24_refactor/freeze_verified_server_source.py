#!/usr/bin/env python3
"""Freeze exact successful server sources with cryptographic checksums.

This utility intentionally does NOT claim to extract RHF/DMRG functions.
It preserves the exact bytes for subsequent verified mechanical extraction.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jaqmc.extensions.mps_distill.schemas import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--server-yaml", type=Path, required=True)
    parser.add_argument("--server-slurm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "python": args.server_python,
        "yaml": args.server_yaml,
        "slurm": args.server_slurm,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "source_frozen_not_yet_regression_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    for key, path in paths.items():
        target = output / f"verified_server_{key}{path.suffix}"
        shutil.copyfile(path, target)
        original_hash = file_sha256(path)
        if file_sha256(target) != original_hash:
            raise RuntimeError(f"Source changed while copying: {path}")
        manifest["sources"][key] = {
            "original_path": str(path.resolve()),
            "frozen_path": str(target),
            "sha256": original_hash,
        }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
