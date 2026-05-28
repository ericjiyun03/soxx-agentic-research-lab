from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_artifact_manifest(
    *,
    output_dir: Path,
    run_id: str,
    config: dict[str, Any],
    feature_set: dict[str, Any] | None = None,
    source_files: list[Path] | None = None,
) -> dict[str, Any]:
    artifact_hashes: dict[str, str] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifact_hashes[path.name] = sha256_file(path)

    source_hashes: dict[str, str] = {}
    for path in source_files or []:
        if path.exists() and path.is_file():
            source_hashes[str(path)] = sha256_file(path)

    return {
        "run_id": run_id,
        "config_hash": sha256_json(config),
        "feature_set": feature_set or {},
        "artifact_count": len(artifact_hashes),
        "artifacts": artifact_hashes,
        "source_files": source_hashes,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
