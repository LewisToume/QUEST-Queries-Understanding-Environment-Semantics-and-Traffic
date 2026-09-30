# Offline Distillation

Store one Navformer Agent file per sample at
`data/soft_labels_navformer/<token>.pt`.

```python
{
    "token": "<OpenScene sample token>",
    "agent": {
        "labels": Tensor[64],
        "boxes": Tensor[64, 8],
        "velocity": Tensor[64, 3],
    },
}
```

Generate these files with `scripts/export_navformer_pseudo.py`. Missing files
are treated as unavailable supervision, not as errors or synthetic targets.
The stored token must match the filename and the dataset sample token.
Only verified Navformer vehicle and pedestrian predictions are exported, using
the configured confidence and BEV-distance filters.

```bash
python scripts/export_navformer_pseudo.py --sample-index 0 --num-frames 100
python scripts/train_stage2_distill.py
```
