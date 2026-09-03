# LeRobot sample

`lerobot_v3/` is the only bundled sample. It is a compact, valid LeRobot v3
dataset containing eight contrasting centered 15-second clips selected from
the sorted Hot3D sequence exports. The episodes retain their per-frame camera and hand
labels, task text, and synchronized H.264 video.

The sample is intended to exercise the same GT-versus-prediction viewer used
for full Hot3D datasets. It contains 3,600 frames at 30 fps and is kept below
30 MB. Anonymous source collection indices and centered frame ranges are
recorded in `lerobot_v3/sample_manifest.json`.

The legacy standalone sample MP4 files are intentionally not included. To
rebuild this sample from locally available source exports, run:

```bash
python scripts/build_sample_lerobot.py \
  --source-root /path/to/hot3d_to_lerobot \
  --output data/samples/lerobot_v3
```

Redistribution approval and privacy review remain the responsibility of the
publisher. Do not add source participant IDs, usernames, machine paths, or
other private metadata to this directory.
