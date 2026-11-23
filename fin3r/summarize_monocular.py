import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean

root, output, adapter = map(Path, sys.argv[1:])
digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
config = json.loads(Path(__file__).with_name("moge2_eval.json").read_text())
splits = json.loads(Path(__file__).with_name("moge_splits.json").read_text())["datasets"]
results = {}
for name in config:
    raw = (root / name / ".index.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != splits[name]["index_sha256"]:
        raise ValueError(f"Unexpected official {name} split")
    names = raw.decode().splitlines()
    rows = sorted((json.loads(line) for shard in output.glob(f"{name}-*.jsonl")
                   for line in shard.read_text().splitlines()), key=lambda r:r["index"])
    if len(rows) != len(names) or any((r["index"],r["image"],r["dataset"],r["candidate_sha256"]) != (i,names[i],name,digest) for i,r in enumerate(rows)):
        raise ValueError(f"Incomplete or stale rows for {name}")
    results[name] = {
        group: {metric: fmean(row["release"][group][metric] for row in rows) for metric in ("rel", "delta1")}
        for group in rows[0]["release"]
    }
macro = {
    group: {metric: fmean(dataset[group][metric] for dataset in results.values()) for metric in ("rel", "delta1")}
    for group in next(iter(results.values()))
}
out = dict(adapter_sha256=digest, images=sum(splits[n]["images"] for n in config), datasets=results, macro=macro)
assert len(config)==8 and out["images"]==5470
tmp=output/"summary.tmp"
tmp.write_text(json.dumps(out,indent=2)+"\n")
tmp.replace(output/"summary.json")
print("Monocular depth, 8 datasets:", json.dumps(macro), flush=True)
