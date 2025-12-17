from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def deterministic_seed(seed: Optional[int] = None, extra: Optional[int] = None) -> int:
    """Return a deterministic non-negative seed combining base seed and optional extra."""
    base = seed if seed is not None else 0
    if extra is not None:
        base = (base * 31 + extra) % (2**31 - 1)
    return base


def safe_json_dump(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def safe_json_load(data: str) -> Any:
    return json.loads(data)


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class CoxModelError(RuntimeError):
    pass


def log_skip(type_value: Any, reason: str) -> None:
    logger.warning("Skipping type %s: %s", type_value, reason)
