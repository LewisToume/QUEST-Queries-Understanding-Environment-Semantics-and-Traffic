#!/usr/bin/env bash
# User-run asset preparation only. No pip/conda, Python, inference or training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

prepare_source() {
    local destination="$1" url="$2" revision="$3"
    if [[ ! -e "$destination/.git" ]]; then
        if [[ -d "$destination" && -n "$(ls -A -- "$destination")" ]]; then
            printf 'BLOCKED nonempty non-Git directory: %s\n' "$destination" >&2
            exit 2
        fi
        git clone --quiet "$url" "$destination"
    fi
    local origin
    origin="$(git -C "$destination" remote get-url origin)"
    if [[ "${origin%.git}" != "${url%.git}" ]]; then
        printf 'BLOCKED unexpected origin: %s\n' "$origin" >&2
        exit 2
    fi
    if ! git -C "$destination" diff HEAD --quiet; then
        printf 'BLOCKED tracked source modifications: %s\n' "$destination" >&2
        exit 2
    fi
    if ! git -C "$destination" cat-file -e "${revision}^{commit}" 2>/dev/null; then
        git -C "$destination" fetch --quiet origin "$revision"
    fi
    git -C "$destination" checkout --quiet --detach "$revision"
    printf 'source_pin=PASS %s %s\n' "$destination" "$revision"
}

mkdir -p -- "$ROOT/third_party" "$ROOT/weights/teachers/dvgt1"
prepare_source "$ROOT/third_party/DVGT" https://github.com/wzzheng/DVGT.git 51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236
prepare_source "$ROOT/third_party/DVGT/third_party/dinov3" https://github.com/facebookresearch/dinov3.git 6876159a11b4df116f30f667f8c9888617df0751

WEIGHT="$ROOT/weights/teachers/dvgt1/dvgt1.pt"
SHA=8e1ab60f901cffc3316fa0a09f24ddcf6b90d0221cf7bfea64664c6f83cc1e22
verify_weight() {
    printf '%s  %s\n' "$SHA" "$1" | sha256sum --check --status
}
if [[ -e "$WEIGHT" ]]; then
    verify_weight "$WEIGHT" || { echo 'BLOCKED existing checkpoint hash mismatch; not overwritten' >&2; exit 2; }
else
    PART="${WEIGHT}.part"
    if [[ ! -f "$PART" ]] || ! verify_weight "$PART"; then
        echo 'Downloading official DVGT-1 weights (~6.93 GB); partial download resumes.'
        curl --fail --location --retry 3 --continue-at - --progress-bar \
            https://huggingface.co/RainyNight/DVGT-1/resolve/4918b6ed3547a00fd217d602fabb84ada180dba4/dvgt1.pt \
            --output "$PART"
    fi
    verify_weight "$PART" || { echo 'BLOCKED downloaded checksum mismatch; no checkpoint published' >&2; exit 2; }
    # Exclusive publish: never replace a checkpoint created by another process.
    ln -- "$PART" "$WEIGHT"
    rm -- "$PART"
fi
echo 'DVGT_ASSETS=PASS; environment_unchanged=True model_not_loaded=True'
