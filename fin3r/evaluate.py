"""Compare VGGT and a saved Fin3R adapter with Pi3's published splits and metrics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from fin3r.model import Fin3RVGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False))
    tmp.replace(path)


def setup_pi3(repo):
    sys.path.insert(0, str(repo.resolve()))
    # Pi3's datasets directory is a namespace package and can collide with
    # Hugging Face datasets in an existing training environment.
    pkg = types.ModuleType("datasets")
    pkg.__path__ = [str(repo.resolve() / "datasets")]
    sys.modules["datasets"] = pkg


def pose_scores(r, t):
    from relpose.metric import calculate_auc_np

    r, t = np.asarray(r), np.asarray(t)
    if not np.isfinite(r).all() or not np.isfinite(t).all() or len(r) == 0:
        raise ValueError("Invalid pair errors")
    scores = {}
    for tau in (5, 15, 30):
        scores[f"RRA@{tau}"] = 100 * float(np.mean(r < tau))
        scores[f"RTA@{tau}"] = 100 * float(np.mean(t < tau))
        scores[f"AUC@{tau}"] = 100 * float(calculate_auc_np(r, t, max_threshold=tau)[0])
    return scores


def pose(model, row, root, device):
    from relpose.metric import se3_to_relative_pose_error
    from utils.interfaces import load_and_resize14

    images = load_and_resize14([str(root / k) for k in row["paths"]], 518, device, False)
    if images.shape[1] != len(row["ids"]):
        raise ValueError(row["scene"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, _ = model.aggregator(images)
    enc = model.camera_head([tokens[-1][:, :, :1].float()])[-1]
    extr, _ = pose_encoding_to_extri_intri(enc, images.shape[-2:])
    n = images.shape[1]
    pred = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    gt = pred.clone()
    pred[:, :3] = extr[0].double().cpu()
    gt[:, :3] = torch.as_tensor(row["w2c"], dtype=torch.float64)[..., :3, :]
    r, t = se3_to_relative_pose_error(pred, gt, n)
    return dict(ids=row["ids"], rotation_deg=r.tolist(), translation_deg=t.tolist())


def point_scores(pred, gt, threshold):
    import open3d as o3d
    from mv_recon.utils import umeyama, accuracy, completion

    if len(gt) < 3 or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Invalid point map on annotated pixels")
    scale, R, t = umeyama(pred.T, gt.T)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(scale * (pred @ R.T) + t.T)
    g = o3d.geometry.PointCloud()
    g.points = o3d.utility.Vector3dVector(gt)
    reg = o3d.pipelines.registration
    p.transform(
        reg.registration_icp(
            p, g, threshold, np.eye(4), reg.TransformationEstimationPointToPoint()
        ).transformation
    )
    p.estimate_normals()
    g.estimate_normals()
    gn, pn = np.asarray(g.normals), np.asarray(p.normals)
    acc, am, n1, n1m = accuracy(g.points, p.points, gn, pn)
    comp, cm, n2, n2m = completion(g.points, p.points, gn, pn)
    return dict(
        acc_mean=acc,
        acc_median=am,
        comp_mean=comp,
        comp_median=cm,
        nc_mean=(n1 + n2) / 2,
        nc_median=(n1m + n2m) / 2,
    )


def point(model, dataset, name, ids, kind, device):
    from utils.interfaces import load_and_resize14

    batch = dataset.get_data(sequence_name=name, ids=ids)
    gt = batch["pointclouds"]
    valid = batch["valid_mask"]
    images = load_and_resize14(batch["image_paths"], 518, device, False)
    if images.shape[1] != len(ids):
        raise ValueError(name)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    extr, K = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    unprojected = unproject_depth_map_to_point_map(pred["depth"][0], extr[0], K[0])
    outputs = {
        "pointmap": pred["world_points"][0].float().cpu(),
        "cam_depth": torch.from_numpy(unprojected).float(),
    }
    result = {}
    for head, pts in outputs.items():
        pts = (
            F.interpolate(
                pts.permute(0, 3, 1, 2),
                size=gt.shape[1:3],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .permute(0, 2, 3, 1)
            .numpy()
        )
        result[head] = point_scores(pts[valid], gt[valid], 100 if kind == "dtu" else 0.1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["co3d", "re10k", "eth3d", "dtu"])
    for name in ("base", "adapter", "pi3", "data", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--limit-scenes", type=int, help="Evaluate a small subset in a separate smoke output"
    )
    args = parser.parse_args()
    if args.limit_scenes is not None and args.limit_scenes < 1:
        parser.error("--limit-scenes must be positive")
    setup_pi3(args.pi3)
    ran, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    device = torch.device("cuda", ran)
    torch.cuda.set_device(device)
    # Use a separate *node-local* torchrun group for each benchmark.
    if world > 1:
        torch.distributed.init_process_group("gloo")
    is_pose = args.task in ("co3d", "re10k")
    folder = args.out / (f"smoke-{args.limit_scenes}" if args.limit_scenes else "") / args.task
    root = args.data / args.task
    digest = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    if is_pose:
        rows = {r["scene"]: r for r in json.loads((root / "pose.json").read_text())}
        map_name = "CO3Dv2" if args.task == "co3d" else "Re10K"
        frozen = json.loads(
            (args.pi3 / f"datasets/seq-id-maps/{map_name}_relpose_seq-id-map_seed42.json").read_text()
        )
        if {k: r["ids"] for k, r in rows.items()} != frozen:
            raise ValueError("Benchmark split/frame IDs differ from the upstream release")
    else:
        from datasets.dtu import DTU
        from datasets.eth3d import ETH3D

        kind = args.task.upper()
        rows = json.loads(
            (args.pi3 / f"datasets/seq-id-maps/{kind}_mv-recon_seq-id-map-kf5.json").read_text()
        )
        folder.mkdir(parents=True, exist_ok=True)
        cache = str(folder / f"index_{ran}.npy")
        ds = (
            DTU(DTU_DIR=str(root), cache_file=cache)
            if args.task == "dtu"
            else ETH3D(ETH3D_DIR=str(root), cache_file=cache)
        )
        if (root / "source_image_index.json").is_file():
            index = json.loads((root / "source_image_index.json").read_text())
            for scene, ids in rows.items():
                selected = [index[scene]["all_names"][i] for i in ids]
                if selected != index[scene]["selected"] or set(selected) != set(ds.metadata[scene]):
                    raise ValueError(f"ETH3D keyframes differ for {scene}")
                rows[scene] = [ds.metadata[scene].index(name) for name in selected]
        if set(ds.sequence_list) != set(rows):
            raise ValueError("Benchmark scans differ from the upstream release")
    scenes = sorted(rows)[: args.limit_scenes]
    for variant in ("vggt", "fin3r"):
        model = Fin3RVGGT(args.base, adapters=variant == "fin3r", camera=True)
        if variant == "fin3r":
            model.load_adapter(torch.load(args.adapter, map_location="cpu", weights_only=True))
        model.to(device).eval()
        records = {}
        with torch.inference_mode():
            for i, name in enumerate(scenes[ran::world]):
                records[name] = (
                    pose(model, rows[name], root, device)
                    if is_pose
                    else point(model, ds, name, rows[name], args.task, device)
                )
                if ran == 0 and (i + 1) % 25 == 0:
                    print(
                        f"{args.task} {variant} rank 0 finished {i + 1}/{len(scenes[ran::world])}",
                        flush=True,
                    )
        shard = folder / f"{variant}.{ran}.json"
        atomic_json(shard, dict(scenes=records, adapter_sha256=digest))
        del model
        torch.cuda.empty_cache()
        if world > 1:
            torch.distributed.barrier()
        if ran == 0:
            all_rows = {}
            for i in range(world):
                part = json.loads((folder / f"{variant}.{i}.json").read_text())
                if part["adapter_sha256"] != digest or all_rows.keys() & part["scenes"].keys():
                    raise ValueError("Inconsistent evaluation shards")
                all_rows.update(part["scenes"])
            if set(all_rows) != set(scenes):
                raise ValueError("Incomplete evaluation")
            if is_pose:
                r = [x for row in all_rows.values() for x in row["rotation_deg"]]
                t = [x for row in all_rows.values() for x in row["translation_deg"]]
                scores = dict(**pose_scores(r, t), pairs=len(r))
            else:
                scores = {
                    h: {
                        k: float(np.mean([r[h][k] for r in all_rows.values()]))
                        for k in next(iter(all_rows.values()))[h]
                    }
                    for h in ("pointmap", "cam_depth")
                }
            atomic_json(
                folder / f"{variant}.json",
                dict(
                    scenes=len(scenes),
                    metrics=scores,
                    adapter_sha256=digest if variant == "fin3r" else None,
                ),
            )
        if world > 1:
            torch.distributed.barrier()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
