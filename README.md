# Fin3R — release version

This release adapts VGGT-1B for depth, camera pose and point-map prediction. The trained adapter is included in `weights/adapter.pth`. Start with one image or pass several views of the same scene in order.

**Training code.** For the original Fin3R paper submission, we used a trainer based on DUSt3R. For this public release, we reimplemented the trainer using the VGGT training code. We trained the bundled adapter (`weights/adapter.pth`) starting from the pretrained VGGT-1B base checkpoint using the public recipe described below.

The paper reports 10 epochs of VGGT training with 2–8 views per multi-view example. The public recipe runs for 3 epochs (1,071 updates), uses up to 32 views on synthetic sequences, adds VGGT pseudo-labels on CO3D, DL3DV and ScanNet, and supervises camera-space XYZ on single-image examples. See the [paper, Appendix A.1](https://openreview.net/pdf?id=pZIeK0Xvph) and the [previously released Fin3R checkpoints](https://github.com/Visual-AI/Fin3R#checkpoints).

## Run on your own images

From the release directory, use Python 3.12 and install the PyTorch build for your machine. The reference run used PyTorch 2.8.0 and torchvision 0.23.0.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0
python -m pip install "numpy<2" Pillow einops huggingface_hub safetensors
mkdir -p weights
curl -fL --retry 3 -o weights/vggt.pt https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt
python infer.py --base weights/vggt.pt --output prediction.npz photo.jpg
```

For multiple views, append their image paths to the command. CUDA inference uses BF16. Add `--device cpu` for CPU inference. The NPZ contains per-view depth and confidence, 3D points and confidence, OpenCV world-to-camera extrinsics and pixel intrinsics at the processed resolution. Expected weight hashes are in [release.json](release.json).

## Reproduce the public training run and benchmarks

Install the full dependencies with `python -m pip install -r requirements.txt`, then follow the [data preparation guide](DATA.md). The frozen training schedule contains 64,734 distinct SA-1B photographs and 3,810 multi-view sequences. Training updates rank-8 QKV adapters in 24 frozen VGGT image-encoder blocks using QR initialization. Depth distillation supervises both copies of each single image. Its first copy also receives normalized camera-space XYZ supervision. Multi-view sequences supervise depth and point maps in the first camera’s coordinate frame.

To reproduce the exact update order, run the following once per node on four machines with eight GPUs each. Point the environment variables at your own prepared directories. All nodes must see the same output and label cache. Use a new run directory for a fresh start. A saved training checkpoint in that directory resumes the run.

```bash
export FIN3R_DATA="$PWD/data" FIN3R_LABELS="$PWD/cache/moge"
export FIN3R_WEIGHTS="$PWD/weights" FIN3R_RUN="$PWD/runs/release"
export MASTER_ADDR=trainer-0   # host name or IP of the first node
export NODE_RANK=0            # set to 0, 1, 2 or 3 on each node
bash train.sh
```

Training ends after 3 epochs (1,071 updates at global batch size 64). Keep `max_epochs: 10` to reproduce the learning-rate schedule, which uses a 10-epoch time scale. The 3rd epoch writes `checkpoints/lora_epoch_03.pth`. For evaluation, set `FIN3R_WEIGHTS`, `EVAL_OUTPUT` and the data path below, then run one benchmark on your own CUDA machine. `GPUS` defaults to 1. `ADAPTER` defaults to the shipped weights.

```bash
export FIN3R_WEIGHTS="$PWD/weights" EVAL_OUTPUT="$PWD/scores"
export FIN3R_EVAL="$PWD/eval" MONO_EVAL_ROOT="$PWD/eval/monocular"
bash evaluate.sh eth3d    # also accepts dtu, co3d, re10k and monocular
```

## Release checkpoint results

These scores use the bundled adapter (`weights/adapter.pth`) and the evaluation protocols in this repository. The monocular average covers eight datasets, including Sintel. The paper reports a seven-dataset average that excludes Sintel. [Evaluation guide](EVALUATION.md) · [Paper, Appendix A.2](https://openreview.net/pdf?id=pZIeK0Xvph)

The release checkpoint scored 74.74 / 86.45 AUC@30 on CO3D / RealEstate10K. Direct point-map accuracy / completeness errors were 0.26598 / 0.32522 m on ETH3D and 0.98071 / 1.92995 mm on DTU. [Reference scores](benchmarks.json) · [Source notices](NOTICE.md)
