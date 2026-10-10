"""Lift 2D instance masks + metric depth into 3D object estimates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Object3D:
    """A 3D object hypothesis from one or more views."""

    attrs: dict[str, str]
    center: np.ndarray  # world (or gravity-aligned camera) frame, meters
    extent: np.ndarray  # axis-aligned size (dx, dy, dz)
    num_points: int
    pixels: int
    views: list[int] = field(default_factory=list)
    center_cam: np.ndarray | None = None  # center in the camera frame of the view it was observed in
    track_id: int | None = None

    def label(self) -> str:
        order = ("size", "color", "material", "shape")
        return " ".join(self.attrs[k] for k in order if k in self.attrs)


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Pixels -> camera-frame points (OpenCV axes). Pixel centers are at +0.5."""
    vs, us = np.nonzero(np.isfinite(depth) if mask is None else (mask & np.isfinite(depth)))
    z = depth[vs, us].astype(np.float64)
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def fit_plane_ransac(points: np.ndarray, iters: int = 200, thresh: float = 0.02,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane n.x + d = 0. Returns (unit normal, d, inlier mask)."""
    rng = rng or np.random.default_rng(0)
    best_inliers = np.zeros(len(points), dtype=bool)
    best = (np.array([0.0, 0.0, 1.0]), 0.0)
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ p[0]
        inl = np.abs(points @ n + d) < thresh
        if inl.sum() > best_inliers.sum():
            best_inliers, best = inl, (n, d)
    # least-squares refinement on inliers
    q = points[best_inliers]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[-1]
    if n @ best[0] < 0:
        n = -n
    return n, float(-n @ c), best_inliers


def gravity_frame_from_ground(ground_points_cam: np.ndarray, rng=None) -> np.ndarray:
    """Build a camera->gravity-aligned transform from points on the floor.

    The resulting frame has z up (floor at z = 0), x aligned with the camera's
    right axis and y pointing away from the camera. Useful for real images where
    extrinsics are unknown.
    """
    n, d, _ = fit_plane_ransac(ground_points_cam, rng=rng)
    if d < 0:  # make the normal point toward the camera (i.e., "up")
        n, d = -n, -d
    up = n
    x = np.array([1.0, 0.0, 0.0]) - up * up[0]
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    R = np.stack([x, y, up])  # rows: new axes expressed in camera coordinates
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (0.0, 0.0, d)
    return T


def calibrate_depth_scale(depth: np.ndarray, K: np.ndarray, floor_mask: np.ndarray, cam_to_world: np.ndarray,
                          max_points: int = 20000, rng: np.random.Generator | None = None) -> float:
    """Scale factor that puts the fitted floor plane at the camera's known height.

    Monocular metric depth is typically off by a per-frame global scale, which
    breaks multi-view consistency. When the camera pose is known (e.g. from
    SLAM / VGGT with gravity), the camera height above the floor fixes the scale:
    s = h_cam / d_plane, where d_plane is the camera-to-floor distance measured
    in the (mis-scaled) depth map.
    """
    rng = rng or np.random.default_rng(0)
    pts = backproject(depth, K, floor_mask)
    if len(pts) < 50:
        return 1.0
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    _, d, _ = fit_plane_ransac(pts, thresh=0.02 * float(np.median(pts[:, 2])), rng=rng)
    h = float(cam_to_world[2, 3])
    return h / abs(d) if abs(d) > 1e-6 else 1.0


def object_from_points(points_cam: np.ndarray, attrs: dict[str, str], cam_to_world: np.ndarray, pixels: int,
                       view: int = 0, center_correction: bool = True, trim: float = 1.0) -> Object3D | None:
    """Robust 3D center/extent from the visible surface of an object.

    Only the camera-facing surface is observed, so the median of the points is
    biased toward the camera by up to a full half-size. With
    ``center_correction`` we do amodal completion of the ground footprint:
    in a horizontal frame aligned with the viewing ray, the footprint is
    assumed to be as deep as it is wide, and is anchored at the nearest
    visible surface; vertically, objects are assumed to rest on the floor
    (z = 0 in the world / gravity frame). Without it, the median of visible
    points is used.
    """
    if len(points_cam) < 10:
        return None
    lo, hi = np.percentile(points_cam, [trim, 100 - trim], axis=0)
    keep = np.all((points_cam >= lo) & (points_cam <= hi), axis=1)
    pts_c = points_cam[keep] if keep.sum() >= 10 else points_cam
    pts_w = transform(pts_c, cam_to_world)
    wlo, whi = pts_w.min(axis=0), pts_w.max(axis=0)
    extent = whi - wlo
    center = np.array([np.median(pts_w[:, 0]), np.median(pts_w[:, 1]), 0.5 * (wlo[2] + whi[2])])
    if center_correction:
        ray = center[:2] - cam_to_world[:2, 3]
        ray /= np.linalg.norm(ray) + 1e-9
        perp = np.array([-ray[1], ray[0]])
        along, across = pts_w[:, :2] @ ray, pts_w[:, :2] @ perp
        width = float(across.max() - across.min())
        depth = max(float(along.max() - along.min()), width)
        center[:2] = ray * (along.min() + depth / 2) + perp * (across.min() + width / 2)
        extent[:2] = np.maximum(extent[:2], np.abs(ray) * depth + np.abs(perp) * width)
        if whi[2] > 0:
            center[2] = whi[2] / 2
            extent[2] = whi[2]
    center_cam = (center - cam_to_world[:3, 3]) @ cam_to_world[:3, :3]
    return Object3D(attrs=dict(attrs), center=center, extent=extent, num_points=len(pts_c), pixels=pixels,
                    views=[view], center_cam=center_cam)


def fuse_views(observations: list[Object3D], radius: float = 0.45, match_attrs: bool = True) -> list[Object3D]:
    """Merge per-frame observations of the same physical object.

    Greedy agglomerative clustering in 3D: an observation joins the nearest
    existing track within ``radius`` meters that has identical attributes.
    Tracks keep a pixel-weighted running mean of the center.
    """
    order = sorted(range(len(observations)), key=lambda i: -observations[i].pixels)
    tracks: list[Object3D] = []
    weights: list[float] = []
    for i in order:
        ob = observations[i]
        best, best_d = None, radius
        for t_idx, tr in enumerate(tracks):
            if match_attrs and tr.attrs != ob.attrs:
                continue
            d = float(np.linalg.norm(tr.center[:2] - ob.center[:2]))
            if d < best_d:
                best, best_d = t_idx, d
        if best is None:
            tracks.append(Object3D(dict(ob.attrs), ob.center.copy(), ob.extent.copy(), ob.num_points, ob.pixels,
                                   list(ob.views), track_id=len(tracks)))
            weights.append(float(ob.pixels))
        else:
            tr, w = tracks[best], weights[best]
            tr.center = (tr.center * w + ob.center * ob.pixels) / (w + ob.pixels)
            tr.extent = np.maximum(tr.extent, ob.extent)
            tr.pixels = max(tr.pixels, ob.pixels)
            tr.num_points += ob.num_points
            tr.views.extend(ob.views)
            weights[best] = w + ob.pixels
    return tracks
