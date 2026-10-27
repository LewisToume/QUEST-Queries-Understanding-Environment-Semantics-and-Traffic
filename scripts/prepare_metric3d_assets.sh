#!/usr/bin/env bash
# User-run Linux asset download only. No Python, installation, training or GPU use.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/third_party/Metric3D"
WEIGHTS="$ROOT/weights/teachers/metric3d"
CHECKPOINT="$WEIGHTS/metric_depth_vit_large_800k.pth"
URL="https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth"
mkdir -p -- "$ROOT/third_party" "$WEIGHTS"
if [[ -e "$REPO" && ! -f "$REPO/hubconf.py" ]]; then
    printf 'ERROR: existing incomplete/different repository at %s; not replacing it\n' "$REPO" >&2
    exit 1
fi
if [[ ! -d "$REPO" ]]; then
    git clone --depth 1 https://github.com/YvanYin/Metric3D.git "$REPO"
else
    printf 'Using existing repository without checkout/reset: %s\n' "$REPO"
fi
if [[ -e "$CHECKPOINT" ]]; then
    if [[ ! -s "$CHECKPOINT" ]]; then
        printf 'ERROR: existing empty checkpoint preserved; inspect manually: %s\n' "$CHECKPOINT" >&2
        exit 1
    fi
    printf 'Existing checkpoint preserved (contents not checked): %s\n' "$CHECKPOINT"
else
    curl --fail --location --retry 3 --continue-at - "$URL" --output "$CHECKPOINT.part"
    # Same-filesystem exclusive publish; do not overwrite a concurrently created checkpoint.
    ln -- "$CHECKPOINT.part" "$CHECKPOINT"
    rm -- "$CHECKPOINT.part"
fi
printf 'Asset paths prepared; strict checkpoint validation happens at inference.\n'
printf 'No environment or GPU changes.\n'
