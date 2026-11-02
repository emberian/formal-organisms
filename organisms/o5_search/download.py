"""Download and pin HF checkpoints for o5. Writes results/o5_search/revisions.json."""
import json, sys, pathlib, time
from huggingface_hub import HfApi, snapshot_download
OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "o5_search" / "revisions.json"
repos = sys.argv[1:]
api = HfApi()
rec = json.loads(OUT.read_text()) if OUT.exists() else {}
for r in repos:
    try:
        info = api.model_info(r, files_metadata=True)
        sha = info.sha
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".json", ".model", ".txt")))
        t0 = time.time()
        path = snapshot_download(r, revision=sha, allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"])
        rec[r] = {"revision": sha, "bytes": size, "path": path, "download_s": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        rec[r] = {"error": repr(e)[:500]}
    OUT.write_text(json.dumps(rec, indent=1))
    print(r, rec[r], flush=True)
