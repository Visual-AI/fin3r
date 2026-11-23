"""Prepare the public training inputs for the frozen schedule."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import tarfile
import tempfile

import cv2
import numpy as np

from fin3r.prepare import crop


SCHEDULE = Path(__file__).resolve().parent / "schedule"


def scheduled_batches():
    for shard in sorted(SCHEDULE.glob("[0-9]*.json")):
        yield from json.loads(shard.read_bytes())


def encode_jpeg(bgr, quality):
    success, image = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    return image.tobytes()


def prepare_photos(args):
    target = args.output / "sa1b"
    target.mkdir(parents=True, exist_ok=True)
    expected = {Path(seq[0]).stem for source, batch in scheduled_batches() if source == "sa1b" for seq in batch}
    remaining = {key for key in expected if not (target / (key + ".jpg")).is_file()}
    archives = sorted(path for path in args.archives.iterdir() if path.name.endswith((".tar", ".tar.gz", ".tgz")))
    for path in archives:
        if not remaining:
            break
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".jpg"):
                    continue
                key = Path(member.name).stem
                if key not in remaining:
                    continue
                data = archive.extractfile(member).read()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Invalid image in {path}: {member.name}")
                image, _, _ = crop(image)
                partial = target / (key + ".partial")
                partial.write_bytes(encode_jpeg(image, 93))
                partial.replace(target / (key + ".jpg"))
                remaining.remove(key)
    if remaining:
        raise SystemExit(f"Missing {len(remaining)} of {len(expected)} scheduled SA-1B photos; example: {min(remaining)}")
    print(f"Prepared {len(expected)} photos in {target}")


def source_frame(path, source, record):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    canvas = (record["width"], record["height"]) if source == "co3d" else None
    bgr, _, _ = crop(bgr, canvas=canvas)
    blob = encode_jpeg(bgr, 93 if source == "co3d" else 95)
    return blob, cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def prepare_pseudo(args):
    import torch
    from fin3r.data import rgb_tensor
    from fin3r.pseudo import labels
    from vggt.models.vggt import VGGT

    model = VGGT(enable_point=False, enable_track=False)
    state = torch.load(args.base, mmap=True, weights_only=True, map_location="cpu")
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or any(not name.startswith(("point_head.", "track_head.")) for name in extra):
        raise ValueError("Expected the VGGT-1B base checkpoint")
    del state
    model.requires_grad_(False).eval().cuda()
    manifest = json.loads((SCHEDULE / "teacher-frames.json").read_bytes())
    if args.scene:
        manifest = {args.scene: manifest[args.scene]}
    for key, record in manifest.items():
        source = key.split("/")[0]
        dest = args.output / "pseudo" / key
        n = record["student_views"]
        if dest.is_dir():
            if all((dest / f"{i:02d}.{ext}").is_file() for i in range(n) for ext in ("npz", "jpg")):
                continue
            raise ValueError(f"Incomplete scene directory: {dest}")
        inputs = [source_frame(args.rgb / record["directory"] / name, source, record) for name in record["files"]]
        images = torch.stack([rgb_tensor(bgr) for _, bgr in inputs]).cuda()
        with torch.inference_mode():
            depth, intrinsics, c2w, _, _ = labels(model, images, n)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as temporary:
            staged = Path(temporary) / "frames"
            staged.mkdir()
            for i in range(n):
                (staged / f"{i:02d}.jpg").write_bytes(inputs[i][0])
                np.savez_compressed(staged / f"{i:02d}.npz", K=intrinsics[i], c2w=c2w[i], depth=depth[i])
            staged.rename(dest)
        print(key, flush=True)


def check(args):
    paths = json.loads((SCHEDULE / "raw-paths.json").read_bytes())
    counts = defaultdict(int)
    for source, batch in scheduled_batches():
        if args.source and source != args.source:
            continue
        if not args.all and counts[source]:
            continue
        for seq in (batch if args.all else batch[:1]):
            for stem in seq:
                if source == "sa1b":
                    files = [args.output / (stem + ".jpg")]
                elif source.startswith("pseudo_"):
                    files = [args.output / (stem + suffix) for suffix in (".jpg", ".npz")]
                else:
                    scene, frame = stem.rsplit("/", 1)
                    files = [args.output / paths[scene] / (frame + suffix)
                             for suffix in ("_rgb.png", "_depth.npy", "_cam.npz")]
                for path in files:
                    if not path.is_file():
                        raise SystemExit(f"Missing {source} input: {path}")
                counts[source] += 1
    print("Verified frame references:", dict(counts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(required=True)
    for name, handler in (("sa1b", prepare_photos), ("pseudo", prepare_pseudo), ("check", check)):
        sub = subs.add_parser(name)
        sub.add_argument("--output", type=Path, required=True, help="Prepared training corpus root")
        sub.set_defaults(handler=handler)
        if name == "sa1b":
            sub.add_argument("--archives", type=Path, required=True)
        elif name == "pseudo":
            sub.add_argument("--rgb", type=Path, required=True)
            sub.add_argument("--base", type=Path, required=True)
            sub.add_argument("--scene", help="Prepare just one source/scene from teacher-frames.json")
        else:
            sub.add_argument("--source", choices=("sa1b", "hypersim", "tartanair", "pseudo_co3d", "pseudo_dl3dv", "pseudo_scannet"))
            sub.add_argument("--all", action="store_true", help="Check every scheduled frame")
    options = p.parse_args()
    options.handler(options)
