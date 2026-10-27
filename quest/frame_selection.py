"""Original OpenScene frame selection shared by offline dense teacher exporters."""

from .stage3_split import selected_frames


def select_indices(infos, sample_index, num_frames, explicit_indices=None, manifest=None, split="train"):
    if explicit_indices is not None:
        if manifest is not None:
            raise ValueError("--indices cannot be combined with --split-manifest")
        indices = list(explicit_indices)
    else:
        if sample_index < 0 or num_frames <= 0:
            raise ValueError("sample-index must be nonnegative and num-frames positive")
        source = [row["index"] for row in selected_frames(manifest, split)] if manifest else range(len(infos))
        if sample_index + num_frames > len(source):
            raise ValueError("Requested range exceeds metadata/split length")
        indices = list(source[sample_index:sample_index + num_frames])
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Require nonempty, unique original metadata indices")
    if any(index < 0 or index >= len(infos) for index in indices):
        raise ValueError("Original metadata index out of range")
    tokens = [str(infos[index]["token"]) for index in indices]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Selected metadata tokens are not unique")
    for token in tokens:
        if not token or token in (".", "..") or any(char in token for char in ("/", "\\", ":")):
            raise ValueError("Unsafe metadata token filename")
    return indices
