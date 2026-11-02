"""Lane budget guard: o4 + o5 ledger rows together may not exceed LANE_CAP."""
import json
from common.llm import LEDGER
LANE_CAP = 10.0
LANE = ("o4", "o5")

def lane_spent(prefixes=LANE) -> float:
    t = 0.0
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if str(d.get("organism", "")).split(":")[0] in prefixes:
                t += float(d.get("cost") or 0.0)
    return t

def guard(reserve: float = 0.0):
    s = lane_spent()
    if s + reserve >= LANE_CAP:
        raise RuntimeError(f"lane budget: spent {s:.4f} + reserve {reserve} >= {LANE_CAP}")
    return s
