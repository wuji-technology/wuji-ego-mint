#!/usr/bin/env python3
"""Build the eight-candidate Hot3D LeRobot v3 sample shipped with MINT."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


VIDEO_KEY = "observation.images.ego"
# Keep earlier candidates first, followed by contrasting review candidates.
SOURCE_DATASET_INDICES = (44, 45, 29, 14, 58, 49, 60, 83)
CLIP_DURATION_SECONDS = 15.0
TARGET_WIDTH = 512
TARGET_HEIGHT = 512


def _discover_datasets(source_root: Path) -> list[Path]:
    datasets = {
        info_path.parent.parent
        for info_path in source_root.rglob("meta/info.json")
        if (info_path.parent.parent / "data").is_dir()
        and (info_path.parent.parent / "videos" / VIDEO_KEY).is_dir()
    }
    return sorted(datasets)


def _single_episode(root: Path) -> dict:
    rows = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    if len(rows) != 1:
        raise RuntimeError(f"expected one episode in {root}, found {len(rows)}")
    return rows[0]


def _task_map(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    for text_column in ("task", "__index_level_0__"):
        if text_column in table.column_names:
            return dict(
                zip(
                    (int(value) for value in table["task_index"].to_pylist()),
                    (str(value) for value in table[text_column].to_pylist()),
                )
            )
    raise RuntimeError(f"task text column not found in {root / 'meta' / 'tasks.parquet'}")


def _replace_column(table: pa.Table, name: str, values: list) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"column {name!r} not found")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def _scale_intrinsics(table: pa.Table, source_width: int, source_height: int) -> pa.Table:
    if "intrinsics" not in table.column_names:
        return table
    scale_x = TARGET_WIDTH / source_width
    scale_y = TARGET_HEIGHT / source_height
    scaled = []
    for values in table["intrinsics"].to_pylist():
        matrix = list(values)
        matrix[:3] = [value * scale_x for value in matrix[:3]]
        matrix[3:6] = [value * scale_y for value in matrix[3:6]]
        scaled.append(matrix)
    return _replace_column(table, "intrinsics", scaled)


def _extract_video(
    source: Path,
    start: float,
    frames: int,
    fps: float,
    output: Path,
) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.9f}", "-i", str(source),
        "-frames:v", str(frames),
        "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", f"{fps:g}",
        str(output),
    ]
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_root: Path, output: Path) -> dict:
    datasets = _discover_datasets(source_root)
    if not datasets:
        raise RuntimeError(f"no Hot3D LeRobot datasets found under {source_root}")
    if max(SOURCE_DATASET_INDICES) >= len(datasets):
        raise RuntimeError(
            f"need dataset index {max(SOURCE_DATASET_INDICES)}, found only {len(datasets)} datasets"
        )

    if output.exists():
        shutil.rmtree(output)
    data_dir = output / "data" / "chunk-000"
    episodes_dir = output / "meta" / "episodes" / "chunk-000"
    videos_dir = output / "videos" / VIDEO_KEY / "chunk-000"
    for directory in (data_dir, episodes_dir, videos_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_tables: list[pa.Table] = []
    tasks: list[str] = []
    task_to_index: dict[str, int] = {}
    manifest_rows: list[dict] = []
    dataset_offset = 0
    template: dict | None = None

    for new_episode, dataset_index in enumerate(SOURCE_DATASET_INDICES):
        root = datasets[dataset_index]
        info = json.loads((root / "meta" / "info.json").read_text())
        if template is None:
            template = info
        fps = float(info.get("fps", 30.0))
        clip_frames = int(round(CLIP_DURATION_SECONDS * fps))

        metadata = _single_episode(root)
        data_path = (
            root
            / "data"
            / f"chunk-{metadata['data/chunk_index']:03d}"
            / f"file-{metadata['data/file_index']:03d}.parquet"
        )
        source_data = pq.read_table(data_path)
        episode_data = source_data.filter(
            pc.equal(source_data["episode_index"], int(metadata["episode_index"]))
        )
        if episode_data.num_rows != int(metadata["length"]):
            raise RuntimeError(f"frame count mismatch for Hot3D dataset index {dataset_index}")
        if episode_data.num_rows < clip_frames:
            raise RuntimeError(
                f"Hot3D dataset index {dataset_index} is shorter than {CLIP_DURATION_SECONDS:g}s"
            )

        source_start_frame = (episode_data.num_rows - clip_frames) // 2
        selected = episode_data.slice(source_start_frame, clip_frames)
        source_shape = info["features"][VIDEO_KEY]["shape"]
        selected = _scale_intrinsics(
            selected,
            source_width=int(source_shape[1]),
            source_height=int(source_shape[0]),
        )

        source_tasks = _task_map(root)
        mapped_task_indices = []
        for old_index in selected["task_index"].to_pylist():
            text = source_tasks[int(old_index)]
            if text not in task_to_index:
                task_to_index[text] = len(tasks)
                tasks.append(text)
            mapped_task_indices.append(task_to_index[text])
        episode_tasks = list(
            dict.fromkeys(
                source_tasks[int(index)] for index in selected["task_index"].to_pylist()
            )
        )

        selected = _replace_column(
            selected, "index", list(range(dataset_offset, dataset_offset + clip_frames))
        )
        selected = _replace_column(selected, "episode_index", [new_episode] * clip_frames)
        selected = _replace_column(selected, "frame_index", list(range(clip_frames)))
        selected = _replace_column(selected, "task_index", mapped_task_indices)
        selected = _replace_column(
            selected, "timestamp", [frame / fps for frame in range(clip_frames)]
        )
        data_tables.append(selected)

        video_path = (
            root
            / "videos"
            / VIDEO_KEY
            / f"chunk-{metadata[f'videos/{VIDEO_KEY}/chunk_index']:03d}"
            / f"file-{metadata[f'videos/{VIDEO_KEY}/file_index']:03d}.mp4"
        )
        source_video_start = (
            float(metadata[f"videos/{VIDEO_KEY}/from_timestamp"])
            + source_start_frame / fps
        )
        output_video = videos_dir / f"file-{new_episode:03d}.mp4"
        _extract_video(video_path, source_video_start, clip_frames, fps, output_video)

        duration = clip_frames / fps
        episode_row = dict(metadata)
        episode_row.update(
            {
                "episode_index": new_episode,
                "length": clip_frames,
                "tasks": episode_tasks,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": new_episode,
                "dataset_from_index": dataset_offset,
                "dataset_to_index": dataset_offset + clip_frames,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": new_episode,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": duration,
            }
        )
        pq.write_table(
            pa.Table.from_pylist([episode_row]),
            episodes_dir / f"file-{new_episode:03d}.parquet",
        )
        manifest_rows.append(
            {
                "episode_index": new_episode,
                "source": "hot3d",
                "source_dataset_index": dataset_index,
                "source_frame_start": source_start_frame,
                "source_frame_stop": source_start_frame + clip_frames,
                "frames": clip_frames,
                "video": str(output_video.relative_to(output)),
                "sha256": _sha256(output_video),
            }
        )
        dataset_offset += clip_frames

    pq.write_table(
        pa.concat_tables(data_tables),
        data_dir / "file-000.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
        output / "meta" / "tasks.parquet",
        compression="zstd",
    )

    assert template is not None
    template.pop("_source_sequence", None)
    template.pop("_hot3d", None)
    template.update(
        {
            "_source_dataset": "hot3d",
            "total_episodes": len(SOURCE_DATASET_INDICES),
            "total_frames": dataset_offset,
            "total_tasks": len(tasks),
            "total_videos": len(SOURCE_DATASET_INDICES),
            "splits": {"train": f"0:{len(SOURCE_DATASET_INDICES)}"},
            "hand_frame": "world",
            "sample_sources": ["hot3d"],
            "sample_selection": {
                "strategy": "sorted dataset indices with centered clips",
                "dataset_indices": list(SOURCE_DATASET_INDICES),
                "clip_duration_seconds": CLIP_DURATION_SECONDS,
            },
        }
    )
    video_feature = template["features"][VIDEO_KEY]
    video_feature["shape"] = [TARGET_HEIGHT, TARGET_WIDTH, 3]
    video_feature["info"].update(
        {"video.height": TARGET_HEIGHT, "video.width": TARGET_WIDTH}
    )
    (output / "meta" / "info.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "format": "LeRobot v3",
        "episodes": manifest_rows,
        "total_frames": dataset_offset,
        "total_bytes_excluding_manifest": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
    }
    (output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/samples/lerobot_v3"))
    args = parser.parse_args()
    manifest = build(args.source_root.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
