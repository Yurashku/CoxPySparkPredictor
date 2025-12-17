from __future__ import annotations

import os
from typing import Dict, Tuple

from .artifacts import TypeArtifacts
from .utils import safe_json_dump, safe_json_load


MANIFEST = "manifest.json"


def save_artifacts(
    path: str,
    artifacts: Dict[str, TypeArtifacts],
    skipped: Dict[str, str],
    config: Dict[str, object],
) -> None:
    os.makedirs(path, exist_ok=True)
    manifest = {
        "types": list(artifacts.keys()),
        "skipped": skipped,
        "config": config,
    }
    with open(os.path.join(path, MANIFEST), "w", encoding="utf-8") as f:
        f.write(safe_json_dump(manifest))
    for t, art in artifacts.items():
        with open(os.path.join(path, f"type_{t}.json"), "w", encoding="utf-8") as f:
            f.write(safe_json_dump(art.__dict__))


def load_artifacts(path: str) -> Tuple[Dict[str, TypeArtifacts], Dict[str, str], Dict[str, object]]:
    with open(os.path.join(path, MANIFEST), "r", encoding="utf-8") as f:
        manifest = safe_json_load(f.read())
    artifacts: Dict[str, TypeArtifacts] = {}
    for t in manifest.get("types", []):
        with open(os.path.join(path, f"type_{t}.json"), "r", encoding="utf-8") as f:
            data = safe_json_load(f.read())
            artifacts[t] = TypeArtifacts(**data)
    skipped = manifest.get("skipped", {})
    config = manifest.get("config", {})
    return artifacts, skipped, config
