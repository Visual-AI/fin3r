import os

import cv2
import numpy as np
import torch

from fin3r.data import rgb
from moge.model.v1 import MoGeModel


def cache_rank_images(mine, root, labels, weights, batch, rank, stop_file=None):
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    cv2.setNumThreads(1)
    todo = [k for k in mine if not (labels / (k + ".npz")).is_file()]
    print(f"rank={rank} cached={len(mine) - len(todo)} pending={len(todo)}", flush=True)
    if not todo:
        return
    with torch.random.fork_rng(devices=[device.index]):
        model = MoGeModel.from_pretrained(str(weights)).eval().to(device)
    for i in range(0, len(todo), batch):
        if stop_file is not None and stop_file.exists():
            print(f"rank={rank} prefilled={i} schedule ready", flush=True)
            return
        group = todo[i : i + batch]
        images = torch.stack([rgb(root, k) for k in group]).to(device)
        with torch.inference_mode():
            label = model.infer(images)
        for stem, depth, K in zip(
            group, label["depth"].float().cpu().numpy(), label["intrinsics"].float().cpu().numpy()
        ):
            good = np.isfinite(depth) & (depth > 0)
            out = labels / (stem + ".npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".{rank}.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, depth=np.where(good, depth, 0).astype(np.float32), K=K)
            tmp.replace(out)
        if rank == 0 and (i // batch) % 64 == 0:
            print(f"teacher rank 0 completed {i + len(group)}/{len(todo)}", flush=True)
    print(f"rank={rank} ready={len(mine)} generated={len(todo)}", flush=True)
