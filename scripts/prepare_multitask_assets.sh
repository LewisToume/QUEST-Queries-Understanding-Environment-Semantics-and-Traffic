#!/usr/bin/env bash
# User-run downloads only: no MapTR dependency, pip/conda changes or GPU work.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SEG="$ROOT/weights/teachers/segformer_cityscapes_b5"
BASE="https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024/resolve/main"
mkdir -p -- "$SEG"

download_new() {
    local url="$1" target="$2"
    if [[ -e "$target" ]]; then
        if [[ ! -s "$target" ]]; then
            printf 'ERROR: empty existing asset preserved; inspect manually: %s\n' "$target" >&2
            return 1
        fi
        printf 'PRESERVED %s (integrity not checked)\n' "$target"
        return
    fi
    curl --fail --location --retry 3 --continue-at - "$url" --output "$target.part"
    [[ -s "$target.part" ]] || return 1
    ln -- "$target.part" "$target"
    rm -- "$target.part"
    printf 'DOWNLOADED %s\n' "$target"
}

download_new "$BASE/config.json" "$SEG/config.json"
download_new "$BASE/preprocessor_config.json" "$SEG/preprocessor_config.json"
if [[ -s "$SEG/model.safetensors" ]]; then
    printf 'PRESERVED %s/model.safetensors (integrity not checked)\n' "$SEG"
else
    download_new "$BASE/pytorch_model.bin" "$SEG/pytorch_model.bin"
fi
bash "$ROOT/scripts/prepare_metric3d_assets.sh"
printf 'Active routes: Agent=Navformer Map=nuPlan_Vector_GT Seg=SegFormer Depth=Metric3Dv2\n'
printf 'Map raster KD remains OFF. No MapTR download. No runtime validation was performed.\n'
