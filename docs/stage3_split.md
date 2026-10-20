# Stage 3 frame-list split

`scripts/build_stage3_split.py` preserves OpenScene metadata indices 0-4999 as
training and selects three complete 40-frame scenes from the remaining data.
Training and validation cannot share a frame token or scene token. Selection is
deterministic for a fixed `--seed` (default 42). It favors scenes in cities
already represented in training, but can include other cities when necessary.
One scene is sampled from the highest-motion quarter of eligible candidates;
there is no absolute motion threshold. No distance from training or between
validation scenes is required. The manifest reports each selected scene's
motion extent and city. If all eligible scenes are nearly static, inspect
these values before using the validation set.

The v3 manifest records original indices, tokens, scene tokens, cities,
metadata SHA256, and a schema-versioned frame-list SHA256. The builder checks
that each scene has exactly 40 frames, consistent city and optional log identity,
and strictly increasing timestamps. It does not compare coordinates between
cities. City assignment still uses the nuPlan map locator.

On the server, provide the same metadata file that `configs/stage1.yaml` uses:

```bash
python scripts/build_stage3_split.py --metadata /home/user/DataDisk/QUEST_WORK/data/openscene/openscene-v1.0/meta_datas/meta_data_mini.pkl
```

Set `NUPLAN_MAPS_ROOT` and `NUPLAN_MAP_VERSION` beforehand, or pass `--map-root`
and `--map-version`. When replacing an existing v1 or v2
`data/stage3_split.json`, the builder preserves its exact bytes as
`data/stage3_split.v1.backup.json` or `data/stage3_split.v2.backup.json`.
It refuses to replace a different v3 split or overwrite a conflicting backup.
Old manifests and capacity/teacher audits cannot certify the new split.
Inspect the printed scenes and missing-label counts before exporting labels.
The selected metadata file must match the Stage 1 dataset metadata byte-for-byte.

To fill only the selected validation targets, pass
`--split-manifest data/stage3_split.json --split validation` to the Vector GT,
Navformer Agent, and Navformer Map Teacher exporters. Map Teacher export still
runs all preceding frames in each scene to maintain temporal state, but saves
only target tokens. Existing labels are validated rather than overwritten.
Then run `scripts/audit_vector_gt_capacity.py` against the full split; it only
certifies query capacity when every Vector GT label is present and matches the
original index/token. Teacher-channel human verification remains a separate
gate. Run the Map Teacher audit with the same `--split-manifest` and an explicit
new `--output`; it records the split hash and starts with `verified=false`.
The hybrid trainer rejects an old teacher audit or missing Map Teacher labels.
GT-only training does not use or require them.
