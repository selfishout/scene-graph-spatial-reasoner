"""Evaluate the scene-graph reasoner on a spatial-scene-bench dataset.

Example::

    python -m sgreasoner.run_bench --data data/ssb --perception oracle
    python -m sgreasoner.run_bench --data data/ssb --perception oracle --depth-scale 1.15 --depth-noise 0.05
    python -m sgreasoner.run_bench --data data/ssb --video-count-mode max_frame
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from spatial_scene_bench.evaluate import aggregate, format_table, parse_answer, score

from .graph import SceneGraph
from .lifting import fuse_views
from .solver import Solver


def video_graph(frames, mode: str, fusion_radius: float) -> SceneGraph:
    if mode == "fusion":
        return SceneGraph(fuse_views([o for f in frames for o in f], radius=fusion_radius))
    if mode == "max_frame":
        return SceneGraph(max(frames, key=len))
    if mode == "sum":
        return SceneGraph([o for f in frames for o in f])
    raise ValueError(mode)


def run(records, backend, video_count_mode="fusion", fusion_radius=0.45, foundation=False, data_dir=None):
    cache: dict = {}
    results = []
    for rec in records:
        key = (rec["scene_id"], rec["modality"], rec.get("view"))
        if key not in cache:
            if rec["modality"] == "image" and foundation:
                meta = json.loads((Path(data_dir) / "scenes" / f"{rec['scene_id']}.json").read_text())
                cam = meta["views"][rec["view"]]
                img = Image.open(Path(data_dir) / rec["images"][0])
                objs, K, T = backend.observe(img, K=np.array(cam["K"]), cam_to_world=np.array(cam["cam_to_world"]))
                cache[key] = SceneGraph(objs, K, T)
            elif rec["modality"] == "image":
                objs, cam = backend.observe_view(rec["scene_id"], rec["view"])
                cache[key] = SceneGraph(objs, np.array(cam["K"]), np.array(cam["cam_to_world"]))
            else:
                frames, _ = backend.observe_video(rec["scene_id"])
                cache[key] = {m: video_graph(frames, m, fusion_radius) for m in ("fusion", "max_frame", "sum")}
        g = cache[key]
        if isinstance(g, dict):
            g = g[video_count_mode] if rec["type"] == "video_count" else g["fusion"]
        raw = Solver(g).answer(rec["question"]) or ""
        pred = parse_answer(raw, rec["answer_type"], rec.get("options"))
        results.append({"id": rec["id"], "type": rec["type"], "answer": rec["answer"], "prediction": pred,
                        "scores": score(rec, pred)})
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--perception", choices=["oracle", "foundation"], default="oracle")
    p.add_argument("--depth-scale", type=float, default=1.0, help="Global depth scale error (oracle only)")
    p.add_argument("--depth-noise", type=float, default=0.0, help="Relative per-pixel depth noise (oracle only)")
    p.add_argument("--no-center-correction", action="store_true")
    p.add_argument("--scale-calibration", action="store_true",
                   help="Rescale each depth map so the fitted floor matches the known camera height")
    p.add_argument("--video-count-mode", choices=["fusion", "max_frame", "sum"], default="fusion")
    p.add_argument("--fusion-radius", type=float, default=0.45)
    p.add_argument("--types", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    a = p.parse_args()

    with open(Path(a.data) / "questions.jsonl") as f:
        recs = [json.loads(line) for line in f]
    if a.types:
        recs = [r for r in recs if r["type"] in a.types]
    if a.perception == "foundation":
        from .perception import FoundationBackend
        recs = [r for r in recs if r["modality"] == "image"]
        backend = FoundationBackend()
    else:
        from .perception import OracleBackend
        backend = OracleBackend(a.data, depth_scale=a.depth_scale, depth_noise=a.depth_noise,
                                center_correction=not a.no_center_correction,
                                scale_calibration=a.scale_calibration)
    recs = recs[: a.limit] if a.limit else recs

    results = run(recs, backend, a.video_count_mode, a.fusion_radius, a.perception == "foundation", a.data)
    table = aggregate(results)
    print(format_table(table))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
