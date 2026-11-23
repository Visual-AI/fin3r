import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from fin3r.prepare import crop, tartanair_valid_depth
from torch.utils.data import Dataset

from vggt.utils.geometry import depth_to_world_coords_points


def rgb_tensor(image):
    return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)


def rgb(root, stem):
    path = root / (stem + ".jpg")
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(path)
    return rgb_tensor(image)


class FrameBatches(Dataset):
    def __init__(self, root, labels, batches, raw_paths=None, weight_counts=None, mono_points=False):
        self.root, self.labels, self.batches = root, labels, batches
        self.raw_paths = raw_paths or {}
        self.weight_counts = weight_counts
        self.mono_points = mono_points

    def __len__(self):
        return len(self.batches)

    def geometry_frame(self, stem):
        scene, frame = stem.rsplit("/", 1)
        if scene in self.raw_paths:
            base = self.root / self.raw_paths[scene] / frame
            bgr = cv2.imread(str(base) + "_rgb.png")
            if bgr is None:
                raise OSError(base)
            depth = np.load(str(base) + "_depth.npy")
            with np.load(str(base) + "_cam.npz") as meta:
                hyper = scene.startswith("hypersim/")
                if not hyper:
                    depth = tartanair_valid_depth(depth)
                cam = meta["pose" if hyper else "camera_pose"]
                bgr, depth, intrinsic = crop(
                    bgr, depth, meta["intrinsics" if hyper else "camera_intrinsics"]
                )
            valid = np.isfinite(depth) & (depth > 0) & (depth < (1e4 if hyper else 1000))
            return rgb_tensor(bgr), np.where(valid, depth, 0).astype(np.float32), intrinsic, cam
        with np.load(self.root / (stem + ".npz")) as meta:
            return rgb(self.root, stem), meta["depth"], meta["K"], meta["c2w"]

    def __getitem__(self, index):
        source, sequences = self.batches[index]
        mono = source == "sa1b"
        out = defaultdict(list)
        for frames in sequences:
            views = defaultdict(list)
            reference = None
            if self.mono_points and mono and len(frames) != 1:
                raise ValueError("Expected one SA-1B anchor to duplicate")
            for stem in frames:
                if mono:
                    views["images"].append(rgb(self.root, stem))
                    with np.load(self.labels / (stem + ".npz")) as label:
                        depth = torch.from_numpy(label["depth"].astype(np.float32))
                        views["teacher_depth"].append(depth)
                        if self.mono_points:
                            from utils3d_moge import pt
                            # MoGe stores normalized intrinsics for the unpadded image.
                            K = torch.from_numpy(label["K"].astype(np.float32))
                            clean = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0.0)
                            xyz = pt.depth_map_to_point_map(clean, intrinsics=K)
                            out["teacher_anchor_xyz"].append(xyz.unsqueeze(0))
                    continue
                image, depth, intrinsic, cam = self.geometry_frame(stem)
                views["images"].append(image)
                if source == "co3d":
                    # Geometric MVS depth contains long-range speckles inside the object mask.
                    valid_depth = depth[depth > 0]
                    if valid_depth.size:
                        depth[depth > np.percentile(valid_depth, 98)] = 0
                if reference is None:
                    reference = cam
                w2c = (np.linalg.inv(cam) @ reference)[:3]
                views["gt_depth"].append(torch.from_numpy(depth))
                pts, _, mask = depth_to_world_coords_points(depth, w2c, intrinsic)
                views["world_points"].append(torch.from_numpy(pts.astype(np.float32)))
                views["point_masks"].append(torch.from_numpy(mask))
            for key, values in views.items():
                # A monocular example occupies two VGGT slots, with identical
                # pixels and supervision. No cross-view GT exists for this pair.
                out[key].append(torch.stack(values * 2 if mono else values))
        shapes = [v.shape[-2:] for v in out["images"]]
        if len(set(shapes)) > 1:
            # Rare orientation-bucket tails. Bottom/right padding preserves camera coordinates.
            hmax, wmax = map(max, zip(*shapes))
            for key, seqs in out.items():
                for i, (height, width) in enumerate(shapes):
                    pad = (0, wmax - width, 0, hmax - height)
                    if key in ("world_points", "teacher_anchor_xyz"):
                        pad = (0, 0, *pad)
                    seqs[i] = torch.nn.functional.pad(seqs[i], pad)
        batch = {**{k: torch.stack(v) for k, v in out.items()}, "source": source}
        if self.weight_counts is not None:
            batch["loss_weight_count"] = self.weight_counts[index]
        return batch
