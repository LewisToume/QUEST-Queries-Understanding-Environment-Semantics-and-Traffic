"""Explicit CUDA bootstrap for standalone teacher exporters; no import-time GPU use."""

import torch


def initialize_cuda_device(index=0):
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("CUDA logical device index must be a nonnegative integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback or fake teacher labels")
    # reset_peak_memory_stats does not itself trigger PyTorch's lazy CUDA init.
    torch.cuda.init()
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(
            f"CUDA logical device {index} out of range: visible_devices={count}; "
            "select a physical GPU with CUDA_VISIBLE_DEVICES, then use logical cuda:0"
        )
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    torch.cuda.reset_peak_memory_stats(device)
    return device
