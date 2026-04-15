from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


def register_model_version(
    model_dir: Path,
    model_filename: str,
    run_id: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "model_filename": model_filename,
        "registered_at": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "run_id": run_id,
        "metrics": metrics,
        "parameters": _json_safe(parameters),
    }

    registry_path = model_dir / "registry.json"
    if registry_path.exists():
        registry: List[Dict[str, Any]] = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return entry


def get_latest_model_info(model_dir: Path) -> Dict[str, Any]:
    registry_path = Path(model_dir) / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    registry: List[Dict[str, Any]] = json.loads(registry_path.read_text(encoding="utf-8"))
    if not registry:
        raise ValueError("Model registry is empty.")

    return sorted(registry, key=lambda entry: entry["registered_at"], reverse=True)[0]
