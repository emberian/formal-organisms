"""OpenRouter client shared by every organism.

One call path, one ledger. Every request and response is appended to
results/ledger.jsonl with the pinned model id, token usage and cost as
OpenRouter reports it. A hard spend cap (FO_BUDGET_USD, default 35.0) is
enforced across all processes by summing the ledger before each call.

Usage:
    from common.llm import chat
    r = chat("anthropic/claude-haiku-5.5", messages, organism="o1", tag="ep-003",
             max_tokens=2000, temperature=0)
    r["text"], r["usage"], r["cost"]
"""
from __future__ import annotations
import fcntl, json, os, time, urllib.request, urllib.error, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "ledger.jsonl"
KEY_PATH = pathlib.Path.home() / ".openrouter-key"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Pinned identifiers (OpenRouter ids as listed 2026-10-29). Do not substitute
# moving aliases; add a new id instead.
MODELS = {
    "haiku": "anthropic/claude-haiku-5.5",
    "sonnet": "anthropic/claude-sonnet-5.5",
    "gemini-flash": "google/gemini-3.8-flash",
    "deepseek-flash": "deepseek/deepseek-v4.1-flash",
    "gpt-luna": "openai/gpt-5.6-luna",
    "qwen-flash": "qwen/qwen3.8-flash",
}

def _budget() -> float:
    return float(os.environ.get("FO_BUDGET_USD", "35.0"))

def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    with open(LEDGER) as f:
        for line in f:
            try:
                total += float(json.loads(line).get("cost") or 0.0)
            except Exception:
                pass
    return total

def _append(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

class BudgetExceeded(RuntimeError):
    pass

def chat(model: str, messages: list[dict], *, organism: str, tag: str = "",
         max_tokens: int = 2000, temperature: float = 0.0, retries: int = 4,
         extra: dict | None = None) -> dict:
    model = MODELS.get(model, model)
    if spent() >= _budget():
        raise BudgetExceeded(f"spent {spent():.4f} >= budget {_budget()}")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "usage": {"include": True}}
    if extra:
        body.update(extra)
    key = KEY_PATH.read_text().strip()
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.loads(resp.read())
            choice = d["choices"][0]
            text = choice["message"].get("content") or ""
            usage = d.get("usage") or {}
            cost = usage.get("cost")
            rec = {"ts": time.time(), "organism": organism, "tag": tag, "model": model,
                   "served_model": d.get("model"), "provider": d.get("provider"),
                   "temperature": temperature, "max_tokens": max_tokens,
                   "messages": messages, "text": text,
                   "finish_reason": choice.get("finish_reason"),
                   "usage": usage, "cost": cost, "latency_s": time.time() - t0}
            _append(rec)
            return {"text": text, "usage": usage, "cost": cost,
                    "finish_reason": choice.get("finish_reason"), "raw": d}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:300]!r}"
        except Exception as e:  # network, JSON
            last_err = repr(e)
        time.sleep(2 ** attempt)
    _append({"ts": time.time(), "organism": organism, "tag": tag, "model": model,
             "error": last_err, "cost": 0.0})
    raise RuntimeError(last_err)
