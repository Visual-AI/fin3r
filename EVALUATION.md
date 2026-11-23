# Preparing benchmark data

Choose the benchmark you want to run. In the examples, `eval/` is a local directory. Set `FIN3R_EVAL="$PWD/eval"` for geometry or `MONO_EVAL_ROOT="$PWD/eval/monocular"` for monocular depth.

| Benchmark | Local preparation |
|---|---|
| CO3Dv2 pose | Extract the original category archives into `eval/co3d/<category>/`. Keep images, `set_lists/set_lists_fewview_dev.json` and `frame_annotations.jgz`. Run `python prepare_pose.py co3d --images eval/co3d`. [CO3Dv2](https://github.com/facebookresearch/co3d) |
| RealEstate10K pose | Download the official test camera trajectories. For the clips in the bundled Re10K seed-42 map, extract RGB frames at the annotation timestamps (microseconds) and store them as `eval/re10k/<clip>/<timestamp>.jpg`. Run `python prepare_pose.py re10k --images eval/re10k --annotations downloads/RealEstate10K/test`. Some source videos may no longer be available. The converter reports missing frames. [Trajectories and timestamp format](https://google.github.io/realestate10k/download.html) |
| ETH3D point maps | Extract the high-resolution multi-view training DSLR images, calibration and per-scene DSLR ground-truth depth archives into `eval/eth3d/<scene>/`. Run `python third_party/pi3/datasets/preprocess/prepare_eth3d.py eval/eth3d` on all 13 scenes. Keep the full undistorted image lists so the bundled keyframe indices refer to the original ordering. [ETH3D downloads](https://www.eth3d.net/datasets) |
| DTU point maps | Use the Spann3R DTU conversion. Place its scan folders directly under `eval/dtu/`. Each `scanN` needs `images/00000000.jpg`, `depths/00000000.npy`, `cams/00000000_cam.txt` and `binary_masks/00000000.png` with matching frame numbers. [Conversion instructions](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) |

For the eight **monocular depth** datasets, download the matching ZIPs from the pinned MoGe evaluation dataset. Keep each dataset’s hidden `.index.txt` when extracting. The split file in this release records both the ZIP and the index hashes. [MoGe benchmark data](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/tree/574610d77c0cc14d911a1ff46ec529ba934c8960) · [Split hashes](fin3r/moge_splits.json)

```bash
mkdir -p downloads/mono eval/monocular
for name in NYUv2 KITTI ETH3D iBims-1 Sintel DDAD DIODE HAMMER; do
  curl -fL --retry 3 -o "downloads/mono/$name.zip" \
    "https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/574610d77c0cc14d911a1ff46ec529ba934c8960/$name.zip"
  unzip -qo "downloads/mono/$name.zip" -d eval/monocular
  test -s "eval/monocular/$name/.index.txt" || exit 1
done
```

Use `bash evaluate.sh monocular` or one of the geometry benchmark names in the [run instructions](README.md#reproduce-training-or-benchmarks).
