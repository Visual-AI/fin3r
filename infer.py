"""Predict geometry for one image or an ordered group of images."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from fin3r.model import Fin3RVGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--base", required=True, type=Path, help="VGGT-1B checkpoint")
    parser.add_argument("--adapter", type=Path, default=Path(__file__).parent / "weights/adapter.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path, help="Output NPZ file")
    args = parser.parse_args()

    model = Fin3RVGGT(args.base, adapters=True, camera=True)
    model.load_adapter(torch.load(args.adapter, map_location="cpu", weights_only=True))
    model.to(args.device).eval()
    images = load_and_preprocess_images(args.images).unsqueeze(0).to(args.device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
    with autocast:
        pred = model(images)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pred.pop("pose_enc").float(), images.shape[-2:])
    pred.update(extrinsics=extrinsics, intrinsics=intrinsics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **{key: value[0].float().cpu().numpy() for key, value in pred.items()})


if __name__ == "__main__":
    main()
