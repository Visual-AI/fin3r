"""Build a pose-evaluation manifest from locally downloaded images and annotations."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


def co3d(images, plan):
    records = {}
    for split in sorted(images.glob("*/set_lists/set_lists_fewview_dev.json")):
        category = split.parents[1]
        groups = defaultdict(list)
        for scene, frame, path in json.loads(split.read_bytes())["test"]:
            if scene in plan:
                groups[scene].append((frame, path))
        selected = {scene: [frames[i] for i in plan[scene]] for scene, frames in groups.items()}
        needed = {(scene, frame) for scene, frames in selected.items() for frame, _ in frames}
        poses = {}
        import ijson
        with gzip.open(category / "frame_annotations.jgz", "rb") as stream:
            for row in ijson.items(stream, "item", use_float=True):
                key = row["sequence_name"], row["frame_number"]
                if key in needed:
                    view = row["viewpoint"]
                    rot, trans = np.array(view["R"]), np.array(view["T"])
                    rot[:, :2] *= -1
                    trans[:2] *= -1
                    poses[key] = np.column_stack((rot.T, trans)).tolist()
        for scene, frames in selected.items():
            records[scene] = ([path for _, path in frames], [poses[scene, f] for f, _ in frames])
    return records


def re10k(annotations, plan):
    records = {}
    for scene, indices in plan.items():
        lines = (annotations / (scene + ".txt")).read_text().splitlines()[1:]
        frames = [lines[i].split() for i in indices]
        names = [f"{scene}/{int(row[0])}.jpg" for row in frames]
        poses = [np.array(row[7:], dtype=float).reshape(3, 4).tolist() for row in frames]
        records[scene] = names, poses
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("co3d", "re10k"))
    parser.add_argument("--images", required=True, type=Path, help="Image root; pose.json is written here")
    parser.add_argument("--annotations", type=Path, help="RealEstate10K test trajectory directory")
    args = parser.parse_args()
    if args.dataset == "re10k" and args.annotations is None:
        parser.error("--annotations is required for re10k")
    name = "CO3Dv2" if args.dataset == "co3d" else "Re10K"
    maps = Path(__file__).parent / "third_party/pi3/datasets/seq-id-maps"
    plan = json.loads((maps / f"{name}_relpose_seq-id-map_seed42.json").read_bytes())
    rows = co3d(args.images, plan) if args.dataset == "co3d" else re10k(args.annotations, plan)
    if rows.keys() != plan.keys():
        raise ValueError(f"Missing scenes, for example {sorted(plan.keys() - rows.keys())[:3]}")
    manifest = []
    for scene, (paths, poses) in sorted(rows.items()):
        for path in paths:
            if not (args.images / path).is_file():
                raise FileNotFoundError(args.images / path)
        manifest.append(dict(scene=scene, ids=plan[scene], paths=paths, w2c=poses))
    target = args.images / "pose.json"
    staged = target.with_suffix(".tmp")
    staged.write_text(json.dumps(manifest, allow_nan=False) + "\n")
    staged.replace(target)
    print(f"{len(manifest)} scenes written to {target}")


if __name__ == "__main__":
    main()
