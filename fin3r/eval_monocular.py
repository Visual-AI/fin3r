"""Single-RGB VGGT evaluation on the fixed MoGe-2 benchmark lists."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import torch
import torch.nn.functional as F
from moge.test.dataloader import EvalDataLoaderPipeline
from moge.test.metrics import compute_metrics
from fin3r.model import Fin3RVGGT


GROUPS = ("depth_scale_invariant", "depth_affine_invariant", "disparity_affine_invariant")
def ordered_samples(loader, indices):
    def read(i):
        return loader._process_instance(loader._load_instance(i))
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = {}
        for k, idx in enumerate(indices):
            if k == 0:
                work[idx] = pool.submit(read, idx)
            if k + 1 < len(indices):
                nxt = indices[k + 1]
                work[nxt] = pool.submit(read, nxt)
            yield idx, work.pop(idx).result()


def sizes(shape):
    h, w = shape
    return tuple(max(14, 14 * round(37 * side / max(h, w))) for side in (h, w))


def depth_prediction(model, images, target_hw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images, include_depth=True, include_points=False, include_camera=False)
    depth = out["depth"][0, 0, :, :, 0].float()
    return F.interpolate(depth[None, None], size=target_hw, mode="bilinear", align_corners=False)[0, 0]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_hash = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    model = Fin3RVGGT(args.base, adapters=True).eval().cuda()
    model.load_adapter(torch.load(args.adapter, weights_only=True, map_location="cpu"))
    configs = json.loads(Path(__file__).with_name("moge2_eval.json").read_text())
    for name, config in configs.items():
        config["path"] = str(args.root / name)
        # No segmentation-derived masks are used by any of the three depth groups.
        config.pop("segmentation", None)
        loader = EvalDataLoaderPipeline(**config)
        indices = list(range(rank, len(loader), world))
        dest = args.output / f"{name}-{rank:02d}.jsonl"
        temporary = dest.with_suffix(".partial")
        if dest.is_file():
            completed = [json.loads(line) for line in dest.read_text().splitlines()]
            if any(r.get("candidate_sha256") != candidate_hash for r in completed):
                raise ValueError(f"Stale adapter result: {dest}")
            if [(r["index"],r["image"]) for r in completed] != [(i,loader.filenames[i]) for i in indices]:
                raise ValueError(f"Manifest differs from completed shard {dest}")
            continue
        start = time.monotonic()
        with temporary.open("w", buffering=1) as f:
            for j, (i, sample) in enumerate(ordered_samples(loader, indices)):
                h, w = sample["depth"].shape
                images = F.interpolate(sample["image"][None].cuda(), size=sizes((h,w)),
                                       mode="bilinear", align_corners=False, antialias=True)[:, None]
                gt = dict(sample, depth=sample["depth"].cuda(), depth_mask=sample["depth_mask"].cuda())
                row = {"candidate_sha256":candidate_hash, "dataset": name, "index": i, "image": sample["filename"],
                       "valid": int(gt["depth_mask"].sum())}
                if row["valid"] == 0 or row["image"] != loader.filenames[i]:
                    raise ValueError(f"Invalid ground truth or index: {row}")
                depth = depth_prediction(model, images, (h, w))
                if not (torch.isfinite(depth) & (depth > 0)).all():
                    raise ValueError(f"Model returned invalid depth for {name}/{i}")
                values, _ = compute_metrics({"depth_scale_invariant": depth}, gt, mg=GROUPS)
                if set(values) != set(GROUPS) or not all(math.isfinite(v) for g in values.values() for v in g.values()):
                    raise ValueError(f"Invalid Model metrics for {name}/{i}: {values}")
                row["release"] = values
                f.write(json.dumps(row, allow_nan=False) + "\n")
                if rank == 0 and (j == 0 or (j+1) % 25 == 0 or j+1 == len(indices)):
                    print(f"{name} shard0 {j+1}/{len(indices)} ({time.monotonic()-start:.0f}s)", flush=True)
        temporary.replace(dest)


if __name__ == "__main__":
    main()
