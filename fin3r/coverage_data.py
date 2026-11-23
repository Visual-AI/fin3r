"""Load one audited global schedule and cache SA-1B teacher targets per checkpoint interval."""

import hashlib
import json
from pathlib import Path

import torch.distributed as dist
from fin3r.data import FrameBatches
from torch.utils.data import DataLoader


class CoverageFin3RData:
    def __init__(self, root, labels, global_batch_size, updates_per_epoch, num_workers=0, schedule_dir=None, mono_points=False, **kwargs):
        self.mono_points = mono_points
        self.root, self.labels = Path(root), Path(labels)
        self.global_batch_size, self.updates_per_epoch = global_batch_size, updates_per_epoch
        self.num_workers = num_workers
        self.directory = Path(schedule_dir) if schedule_dir else self.root / "schedule"
        self.coverage = json.loads((self.directory / "coverage.json").read_text())
        self.raw_paths = self.read("raw-paths.json")
        self.schedule_epochs = self.coverage["epochs"]

    def read(self, name):
        data = (self.directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.coverage["sha256"][name]:
            raise ValueError(f"Training manifest has changed: {name}")
        return json.loads(data)

    def get_loader(self, epoch):
        rank = dist.get_rank()
        if (
            dist.get_world_size() != self.coverage["ranks"]
            or self.global_batch_size != self.coverage["global_batch"]
        ):
            raise ValueError("Distributed world differs from the audited training schedule")
        shard = f"{epoch - rank:02}-{rank:02}.json"
        batches = self.read(shard)
        weights = self.read("weights/" + shard) if self.coverage.get("weight_replay") else None
        if weights is not None and len(weights) != len(batches):
            raise ValueError("Misaligned comparison weights")
        if len(batches) != self.updates_per_epoch or any(len(seqs) != 2 for _, seqs in batches):
            raise ValueError("Rank batch is incomplete")
        import os

        from fin3r.teacher import cache_rank_images

        keys = [seq[0] for source, seqs in batches if source == "sa1b" for seq in seqs]
        cache_rank_images(
            keys,
            self.root,
            self.labels,
            Path(os.environ["FIN3R_WEIGHTS"]) / "moge.pt",
            batch=8,
            rank=rank,
        )
        dist.barrier()
        return DataLoader(
            FrameBatches(self.root, self.labels, batches, self.raw_paths, weights, mono_points=self.mono_points),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
