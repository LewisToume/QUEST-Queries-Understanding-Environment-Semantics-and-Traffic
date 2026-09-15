# Offline Distillation

Store one file per sample at `data/soft_labels/<token>.pt`.

```python
{
    "token": "<OpenScene sample token>",
    "seg": {"labels": Tensor[8, 64, 64]},
    "depth": {"values": Tensor[8, 1, 64, 64], "valid_mask": Tensor[8, 1, 64, 64]},
    "agent": {
        "labels": Tensor[N],
        "boxes": Tensor[N, 8],
        "velocity": Tensor[N, 3],
    },
    "map": {
        "labels": Tensor[M],
        "points": Tensor[M, 20, 2],
    },
}
```

Only include tasks with verified labels. Missing files and missing task keys are
treated as unavailable supervision, not as errors or synthetic targets. The
stored token must match the filename and the dataset sample token.
