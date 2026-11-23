#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")" && pwd)
if (( $# != 1 )); then echo "Usage: $0 {co3d|re10k|eth3d|dtu|monocular}" >&2; exit 2; fi
: "${FIN3R_WEIGHTS:?directory containing vggt.pt}" "${EVAL_OUTPUT:?directory for benchmark outputs}"
export PYTHONPATH="$repo:$repo/third_party:$repo/third_party/moge:$repo/third_party/utils3d${PYTHONPATH:+:$PYTHONPATH}"
python=${PYTHON:-python}
base="$FIN3R_WEIGHTS/vggt.pt"
adapter=${ADAPTER:-$repo/weights/adapter.pth}
launch=("$python" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-1}")
case "$1" in
  co3d|re10k|eth3d|dtu)
    exec "${launch[@]}" "$repo/fin3r/evaluate.py" "$1" --base "$base" --adapter "$adapter" \
        --pi3 "$repo/third_party/pi3" --data "${FIN3R_EVAL:?prepared geometry benchmark directory}" --out "$EVAL_OUTPUT" ;;
  monocular)
    root=${MONO_EVAL_ROOT:?prepared monocular benchmark directory}
    out="$EVAL_OUTPUT/monocular"
    "${launch[@]}" "$repo/fin3r/eval_monocular.py" --base "$base" --adapter "$adapter" --root "$root" --output "$out"
    exec "$python" "$repo/fin3r/summarize_monocular.py" "$root" "$out" "$adapter" ;;
  *) echo "Unknown benchmark: $1" >&2; exit 2 ;;
esac
