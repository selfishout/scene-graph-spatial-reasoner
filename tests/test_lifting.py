import numpy as np
import pytest
from spatial_scene_bench.camera import default_camera, orbit_cameras
from spatial_scene_bench.render import render
from spatial_scene_bench.scene import Scene, SceneObject

from sgreasoner.lifting import (Object3D, backproject, calibrate_depth_scale, fit_plane_ransac, fuse_views,
                                gravity_frame_from_ground, object_from_points)


def sphere_scene():
    return Scene("s", [SceneObject(0, "sphere", "red", "large", "rubber", (0.5, 0.3, 0.55))])


def test_amodal_center_from_visible_surface():
    scene = sphere_scene()
    cam = default_camera()
    out = render(scene, cam, shadows=False)
    pts = backproject(out.depth, cam.K, out.instance == 0)
    o = object_from_points(pts, {"shape": "sphere"}, cam.cam_to_world, int((out.instance == 0).sum()))
    assert np.linalg.norm(o.center - scene.objects[0].center) < 0.05
    np.testing.assert_allclose(o.extent, 1.1, atol=0.05)


def test_center_correction_reduces_error():
    scene = sphere_scene()
    cam = default_camera()
    out = render(scene, cam, shadows=False)
    pts = backproject(out.depth, cam.K, out.instance == 0)
    errs = [np.linalg.norm(object_from_points(pts, {}, cam.cam_to_world, 100, center_correction=c).center
                           - scene.objects[0].center) for c in (False, True)]
    assert errs[1] < errs[0]


def test_ransac_plane_with_outliers():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-3, 3, (500, 2))
    plane = np.c_[xy, 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + 1.0 + rng.normal(0, 0.003, 500)]
    outliers = rng.uniform(-3, 3, (100, 3))
    n, d, inl = fit_plane_ransac(np.vstack([plane, outliers]), rng=rng)
    true_n = np.array([-0.2, 0.1, 1.0]) / np.linalg.norm([-0.2, 0.1, 1.0])
    assert abs(abs(n @ true_n) - 1) < 1e-3
    assert inl[:500].mean() > 0.95


def test_gravity_frame_recovers_floor_height():
    cam = default_camera()
    out = render(Scene("empty", []), cam, shadows=False)
    pts = backproject(out.depth, cam.K, out.instance == -1)
    T = gravity_frame_from_ground(pts[::20])
    g = pts @ T[:3, :3].T + T[:3, 3]
    assert np.abs(g[:, 2]).max() < 1e-3
    assert T[2, 3] == pytest.approx(cam.position[2], abs=1e-3)  # camera height above floor


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.15])
def test_depth_scale_calibration_from_floor(scale):
    cam = default_camera()
    out = render(sphere_scene(), cam, shadows=False)
    s = calibrate_depth_scale(out.depth * scale, cam.K, out.instance == -1, cam.cam_to_world)
    assert s * scale == pytest.approx(1.0, abs=0.01)


def _obs(attrs, xy, pixels=100, view=0):
    return Object3D(attrs, np.array([*xy, 0.3]), np.ones(3) * 0.6, 50, pixels, [view])


def test_fusion_merges_same_object_and_keeps_twins_apart():
    red = {"color": "red", "shape": "cube"}
    obs = [_obs(red, (0, 0)), _obs(red, (0.1, -0.05), view=1), _obs(red, (2.0, 0.0), view=1),
           _obs({"color": "blue", "shape": "cube"}, (0.05, 0.0), view=2)]
    tracks = fuse_views(obs, radius=0.45)
    assert len(tracks) == 3
    assert sorted(len(t.views) for t in tracks) == [1, 1, 2]


def test_video_fusion_counts_each_object_once():
    scene = Scene("v", [
        SceneObject(0, "cube", "red", "large", "rubber", (-1.5, 0.0, 0.55)),
        SceneObject(1, "cube", "red", "large", "rubber", (1.5, 0.5, 0.55)),
        SceneObject(2, "sphere", "blue", "small", "metal", (0.0, -1.5, 0.30)),
    ])
    obs = []
    for i, cam in enumerate(orbit_cameras(8, radius=7.0, elevation=3.0)):
        out = render(scene, cam, shadows=False)
        for oid in np.unique(out.instance[out.instance >= 0]):
            m = out.instance == oid
            if m.sum() < 40:
                continue
            o = scene.by_id(int(oid))
            obs.append(object_from_points(backproject(out.depth, cam.K, m), {"color": o.color, "shape": o.shape},
                                          cam.cam_to_world, int(m.sum()), i))
    assert len(obs) > 3
    assert len(fuse_views(obs)) == 3
