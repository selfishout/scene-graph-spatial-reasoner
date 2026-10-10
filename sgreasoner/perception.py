"""Perception backends that turn images into per-view 3D object hypotheses.

``OracleBackend``      reads rendered depth + instance masks from spatial-scene-bench.
                       Object *positions are never read from ground truth*; they are
                       re-estimated from depth, so lifting / fusion / reasoning errors
                       are measured in isolation from detection errors. Depth can be
                       corrupted to emulate monocular depth estimators.
``FoundationBackend``  Grounding DINO (open-vocabulary boxes) -> SAM (masks) ->
                       Depth Anything V2 Metric (depth) -> heuristic attributes.
"""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .lifting import Object3D, backproject, calibrate_depth_scale, gravity_frame_from_ground, object_from_points

PALETTE = {
    "red": (0.80, 0.15, 0.15), "blue": (0.15, 0.30, 0.85), "green": (0.15, 0.65, 0.25),
    "yellow": (0.90, 0.80, 0.15), "purple": (0.55, 0.20, 0.70), "cyan": (0.15, 0.75, 0.80),
    "gray": (0.50, 0.50, 0.50), "brown": (0.50, 0.32, 0.15),
}


def _corrupt_depth(depth: np.ndarray, rng: np.random.Generator, scale: float, noise: float) -> np.ndarray:
    d = depth.astype(np.float64) * scale
    if noise > 0:
        d = d * (1.0 + noise * rng.standard_normal(d.shape))
    return d


class OracleBackend:
    def __init__(self, data_dir: str | Path, min_pixels: int = 40, depth_scale: float = 1.0,
                 depth_noise: float = 0.0, center_correction: bool = True, scale_calibration: bool = False,
                 seed: int = 0):
        self.root = Path(data_dir)
        self.min_pixels = min_pixels
        self.depth_scale = depth_scale
        self.depth_noise = depth_noise
        self.center_correction = center_correction
        self.scale_calibration = scale_calibration
        self.rng = np.random.default_rng(seed)
        self._meta: dict[str, dict] = {}

    def meta(self, scene_id: str) -> dict:
        if scene_id not in self._meta:
            self._meta[scene_id] = json.loads((self.root / "scenes" / f"{scene_id}.json").read_text())
        return self._meta[scene_id]

    def _labels(self, scene_id: str) -> dict[int, dict[str, str]]:
        return {o["id"]: {k: o[k] for k in ("size", "color", "material", "shape")}
                for o in self.meta(scene_id)["scene"]["objects"]}

    def _lift(self, depth, instance, cam: dict, labels, view: int) -> list[Object3D]:
        depth = _corrupt_depth(depth, self.rng, self.depth_scale, self.depth_noise)
        K, T = np.array(cam["K"]), np.array(cam["cam_to_world"])
        if self.scale_calibration:
            depth = depth * calibrate_depth_scale(depth, K, instance == -1, T)
        objs = []
        ids, counts = np.unique(instance[instance >= 0], return_counts=True)
        for i, n in zip(ids.tolist(), counts.tolist()):
            if n < self.min_pixels:
                continue
            pts = backproject(depth, K, instance == i)
            o = object_from_points(pts, labels[i], T, n, view, self.center_correction)
            if o is not None:
                objs.append(o)
        return objs

    def observe_view(self, scene_id: str, view: int) -> tuple[list[Object3D], dict]:
        g = np.load(self.root / "geometry" / f"{scene_id}_v{view}.npz")
        cam = self.meta(scene_id)["views"][view]
        return self._lift(g["depth"], g["instance"], cam, self._labels(scene_id), view), cam

    def observe_video(self, scene_id: str) -> tuple[list[list[Object3D]], list[dict]]:
        g = np.load(self.root / "video_geometry" / f"{scene_id}.npz")
        cams = self.meta(scene_id)["video"]
        labels = self._labels(scene_id)
        frames = [self._lift(g["depth"][i], g["instance"][i], cams[i], labels, i)
                  for i in range(len(cams))]
        return frames, cams


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.6) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ious = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][ious < iou]
    return keep


def color_name(rgb_pixels: np.ndarray) -> str:
    """Nearest palette color in hue/saturation space, robust to shading."""
    rgb = np.median(rgb_pixels.astype(np.float64) / 255.0, axis=0)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    best, best_d = "gray", float("inf")
    for name, ref in PALETTE.items():
        rh, rs, rv = colorsys.rgb_to_hsv(*ref)
        dh = min(abs(h - rh), 1 - abs(h - rh))
        d = (dh * 4) ** 2 * min(s, rs) + (s - rs) ** 2 + 0.3 * (v - rv) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


class FoundationBackend:
    """Open-vocabulary 3D perception from a single RGB image using Hugging Face models."""

    def __init__(
        self,
        vocabulary: tuple[str, ...] = ("sphere", "cube", "cylinder"),
        detector: str = "IDEA-Research/grounding-dino-tiny",
        segmenter: str = "facebook/sam-vit-base",
        depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        size_threshold: float = 0.85,
        device: str | None = None,
    ):
        import torch
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, SamModel, SamProcessor)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.vocabulary = vocabulary
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.size_threshold = size_threshold
        self.det_proc = AutoProcessor.from_pretrained(detector)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(segmenter)
        self.sam = SamModel.from_pretrained(segmenter).to(self.device).eval()
        self.depth_proc = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_net = AutoModelForDepthEstimation.from_pretrained(depth_model).to(self.device).eval()

    def detect(self, image: Image.Image):
        text = ". ".join(f"a {w}" for w in self.vocabulary) + "."
        inputs = self.det_proc(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.det(**inputs)
        kwargs = dict(text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])
        try:
            r = self.det_proc.post_process_grounded_object_detection(outputs, inputs.input_ids,
                                                                     threshold=self.box_threshold, **kwargs)[0]
        except TypeError:
            r = self.det_proc.post_process_grounded_object_detection(outputs, inputs.input_ids,
                                                                     box_threshold=self.box_threshold, **kwargs)[0]
        labels = r.get("text_labels", r.get("labels"))
        boxes, scores = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
        keep = _nms(boxes, scores)
        shapes = []
        for lab in (labels[i] for i in keep):
            shapes.append(next((w for w in self.vocabulary if w in str(lab)), self.vocabulary[0]))
        return boxes[keep], scores[keep], shapes

    def segment(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, image.size[1], image.size[0]), dtype=bool)
        inputs = self.sam_proc(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        return masks[:, 0].numpy().astype(bool)

    def depth(self, image: Image.Image) -> np.ndarray:
        inputs = self.depth_proc(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.depth_net(**inputs)
        d = self.depth_proc.post_process_depth_estimation(out, target_sizes=[image.size[::-1]])[0]
        return d["predicted_depth"].cpu().numpy().astype(np.float64)

    def observe(self, image: Image.Image, K: np.ndarray | None = None, cam_to_world: np.ndarray | None = None,
                fov_y_deg: float = 50.0, view: int = 0) -> tuple[list[Object3D], np.ndarray, np.ndarray]:
        """Returns (objects, K, cam_to_world). Without extrinsics, a gravity frame is fit to the floor."""
        image = image.convert("RGB")
        W, H = image.size
        if K is None:
            f = (H / 2) / math.tan(math.radians(fov_y_deg) / 2)
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        boxes, _, shapes = self.detect(image)
        masks = self.segment(image, boxes)
        depth = self.depth(image)
        rgb = np.asarray(image)

        floor = np.ones((H, W), dtype=bool)
        floor[: H // 2] = False
        for m in masks:
            floor &= ~m
        if cam_to_world is not None:
            depth = depth * calibrate_depth_scale(depth, K, floor, cam_to_world)
        else:
            pts = backproject(depth, K, floor)
            pts = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
            cam_to_world = gravity_frame_from_ground(pts)

        objs = []
        for m, shape in zip(masks, shapes):
            n = int(m.sum())
            if n < 20:
                continue
            o = object_from_points(backproject(depth, K, m), {"shape": shape, "color": color_name(rgb[m])},
                                   cam_to_world, n, view)
            if o is None:
                continue
            o.attrs["size"] = "large" if float(np.max(o.extent)) > self.size_threshold else "small"
            highlight = np.mean(np.min(rgb[m], axis=1) > 215)
            o.attrs["material"] = "metal" if highlight > 0.01 else "rubber"
            objs.append(o)
        return objs, K, cam_to_world
