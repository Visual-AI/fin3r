# Preparing data on your own machine

**For inference**, use your own JPEG or PNG images and the VGGT-1B checkpoint. The [quick start](README.md#run-on-your-own-images) runs on CPU or GPU. The public datasets below are needed only to reproduce training or evaluation.

## 1. Get the training images

Run commands from the release directory. Choose three local folders. `RAW` holds downloads, `RGB` holds the RGB inputs for the teacher, and `DATA` is the folder the trainer reads. The schedule and all scene and frame IDs ship with this release.

```bash
export RAW="$PWD/downloads" RGB="$PWD/teacher-rgb" DATA="$PWD/data"
mkdir -p "$RAW" "$RGB" "$DATA" weights
```

| Source | What to download and where to put it |
|---|---|
| SA-1B | Put the original image tar shards in `$RAW/sa1b/`. Keep them packed. The preparation command extracts the 64,734 scheduled JPEGs. [Meta download](https://ai.meta.com/datasets/segment-anything/) |
| Hypersim | Download the scenes in `schedule/raw-paths.json` and the dataset’s `metadata_camera_parameters.csv`. Run CUT3R’s Hypersim preprocessor with output `$DATA/hypersim`. Each camera directory should contain matching `*_rgb.png`, `*_depth.npy` and `*_cam.npz`. [Dataset](https://github.com/apple/ml-hypersim) · [CUT3R preprocessing](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) |
| TartanAir | Use the original TartanAir trajectories (Easy/Hard). Download left RGB, left depth and poses. Run CUT3R’s TartanAir preprocessor into `$DATA/tartanair`. Keep its environment, difficulty and trajectory directories. [Dataset](https://theairlab.org/tartanair-dataset/) · [CUT3R preprocessing](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) |
| CO3Dv2 | Download the RGB sequences and follow CUT3R’s CO3D preprocessing. Put the result at `$RGB/co3d/<category>/<sequence>/images/`. The teacher uses those processed RGB frames. [Dataset](https://github.com/facebookresearch/co3d) · [Preprocessing](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) |
| DL3DV-10K | Download the named scenes with the `images_4` image set. Keep `$RGB/dl3dv/<partition>/<scene>/images_4/`. The manifest specifies each partition and filename. [Dataset and downloader](https://github.com/DL3DV-10K/Dataset) |
| ScanNet v2 | Obtain the training scans from ScanNet. Export the .sens RGB, depth, poses and intrinsics first (CUT3R expects `scans_train/<scene>/`), then run CUT3R’s ScanNet RGB conversion, then link its `scans_train` output as `$RGB/scannet`. The expected layout is `<scene>/color/00000.jpg`. [Dataset and access](https://github.com/ScanNet/ScanNet) · [Preprocessing](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) |

For example, from the public CUT3R checkout with its preprocessing dependencies installed, these commands write the layouts above. Set the three absolute paths from this guide before changing directories.

```bash
cd /path/to/CUT3R/datasets_preprocess
python preprocess_hypersim.py --hypersim_dir "$RAW/hypersim" --output_dir "$DATA/hypersim"
python preprocess_tartanair.py --tartanair_dir "$RAW/tartanair" --output_dir "$DATA/tartanair"
python preprocess_co3d.py --co3d_dir "$RAW/co3d" --output_dir "$RGB/co3d" --num_sequences_per_object 100000
mkdir -p "$RAW/scannet/scans_test"  # CUT3R enumerates both split directories
python preprocess_scannet.py --scannet_dir "$RAW/scannet" --output_dir "$RAW/scannet-processed"
ln -s "$RAW/scannet-processed/scans_train" "$RGB/scannet"
cd -
```

Existing datasets can be symlinked into these locations. To inspect one exact source directory and its 16 filenames before downloading a scene:

```bash
python -c 'import json; m=json.load(open("schedule/teacher-frames.json")); k=next(iter(m)); print(k, m[k])'
```

## 2. Prepare supervision

Download `weights/vggt.pt` with the quick-start command. Download MoGe for the single-image depth labels, then extract the scheduled SA-1B photos.

```bash
curl -fL --retry 3 -o weights/moge.pt https://huggingface.co/Ruicheng/moge-vitl/resolve/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/model.pt
python prepare_data.py sa1b --archives "$RAW/sa1b" --output "$DATA"
```

Use a BF16-capable CUDA GPU to generate the multi-view labels. Try one scene first, then run the same command without `--scene` to process the full manifest. Completed scenes are reused. The frozen VGGT teacher reads 16 RGB frames per scene and saves 2, 4 or 8 student frames with depth, pixel intrinsics and camera-to-world poses.

```bash
python prepare_data.py pseudo --rgb "$RGB" --output "$DATA" --base weights/vggt.pt --scene co3d/apple_161_17678_33238
python prepare_data.py pseudo --rgb "$RGB" --output "$DATA" --base weights/vggt.pt
python prepare_data.py check --output "$DATA" --all
```

The check reports the first missing file. For a quick layout check it accepts `--source hypersim` or another source and checks one sequence when `--all` is omitted. The trainer reads synthetic depth and poses directly from the CUT3R outputs. It generates missing SA-1B depth labels with MoGe on first use and caches them under `FIN3R_LABELS`.

## 3. Optional evaluation data

See [benchmark preparation](EVALUATION.md) for separate evaluation downloads. Training and inference can run independently of these benchmarks.
