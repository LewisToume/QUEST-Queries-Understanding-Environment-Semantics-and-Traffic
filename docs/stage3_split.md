# Stage 3 frame-list split

`scripts/build_stage3_split.py` keeps OpenScene metadata indices 0-4999 as training
and chooses three complete, roughly 40-frame held-out scenes from later indices.
City membership is resolved using the nuPlan map projection; nearest-neighbor
queries are performed separately for each city. Every selected frame must be at
least 200 m from training frames in that city. The builder never compares XY
coordinates across cities or relaxes the threshold. It reports when a reliable
log identifier is absent. The split includes all original indices, tokens,
scene tokens, cities, metadata SHA256, and a frame-list SHA256.

On the server, provide the same metadata file that `configs/stage1.yaml` uses:

```bash
python scripts/build_stage3_split.py --metadata /home/user/DataDisk/QUEST_WORK/data/openscene/openscene-v1.0/meta_datas/meta_data_mini.pkl
```

Set `NUPLAN_MAPS_ROOT` and `NUPLAN_MAP_VERSION` beforehand, or pass `--map-root`
and `--map-version`. The builder refuses to overwrite a different existing
split. Inspect the printed scene motion, nearest-training and inter-scene
distance statistics and missing-label counts before exporting labels. The
selected metadata file must match the Stage 1 dataset metadata byte-for-byte.

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
The formal trainer rejects an old teacher audit or missing Map Teacher labels.
