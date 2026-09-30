# Offline Distillation

Store one Navformer Agent file per sample at
`data/soft_labels_navformer/<token>.pt`.

```python
{
    "token": "<OpenScene sample token>",
    "agent": {
        "labels": Tensor[64],
        "boxes_metric": Tensor[64, 7],
        "velocity_mps": Tensor[64, 3],
        "scores": Tensor[64],
        "class_support_mask": BoolTensor[4],
        "valid_mask": BoolTensor[64],
    },
}
```

Generate these files with `scripts/export_navformer_pseudo.py`. Missing files
are treated as unavailable supervision, not as errors or synthetic targets.
The stored token must match the filename and the dataset sample token.
Only verified Navformer vehicle and pedestrian predictions are exported, using
the configured confidence and BEV-distance filters. A prediction must also have
its center or a 3D box corner visible in at least one current camera. Metric box,
velocity, and teacher confidence values are preserved; normalization happens only
inside the task loss where required.

Normalized V2 soft-label files are not compatible with this schema and must be
regenerated before V3 Stage2 training.

The resulting checkpoint records the union of supported Agent classes. Inference
must apply this mask before classification softmax; the four-class taxonomy itself
does not change.

```bash
python scripts/export_navformer_pseudo.py --sample-index 0 --num-frames 100
python scripts/train_stage2_distill.py
```
