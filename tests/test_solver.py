import numpy as np
import pytest

from sgreasoner.graph import SceneGraph
from sgreasoner.lifting import Object3D
from sgreasoner.solver import Solver, parse, parse_options, parse_phrase

# Camera at (0, -8, 5) looking at the origin: world +x is image-right, +y is away from the viewer.
K = np.array([[257.0, 0, 160], [0, 257.0, 120], [0, 0, 1]])


def c2w():
    pos = np.array([0.0, -8.0, 5.0])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = pos
    return T


def ob(color, shape, xy, size="small", material="rubber"):
    return Object3D({"color": color, "shape": shape, "size": size, "material": material},
                    np.array([*xy, 0.3]), np.full(3, 0.6), 100, 200)


@pytest.fixture
def graph():
    return SceneGraph([
        ob("red", "sphere", (-2, 0)),
        ob("blue", "cube", (0, 0), size="large"),
        ob("green", "cylinder", (2, 0), material="metal"),
        ob("yellow", "cube", (0, 2)),
    ], K, c2w())


def test_parse_phrase():
    assert parse_phrase("large metal cubes") == {"size": "large", "material": "metal", "shape": "cube"}
    assert parse_phrase("objects") == {}


def test_parse_options():
    assert parse_options("(A) red sphere (B) blue cube (C) front-left (D) gray object") == [
        "red sphere", "blue cube", "front-left", "gray object"]


@pytest.mark.parametrize("q,expected", [
    ("How many cubes are there?", "2"),
    ("How many objects are to the left of the blue cube?", "1"),
    ("How many objects are behind the blue cube?", "1"),
    ("Is the sphere to the left of the cylinder?", "yes"),
    ("Is the yellow cube in front of the blue cube?", "no"),
    ("Are there more cubes than spheres?", "yes"),
    ("What is the distance between the center of the sphere and the center of the cylinder, in meters?", "4.00"),
    ("Which object is closest to the yellow cube in 3D space? (A) sphere (B) cylinder (C) blue cube (D) metal object",
     "C"),
    ("Imagine you are standing at the sphere and facing the cylinder. Where is the yellow cube relative to you? "
     "(A) front-left (B) front-right (C) back-left (D) back-right", "A"),
    ("How many cubes are in this room? The camera moves around the scene, so count each physical object only once.",
     "2"),
])
def test_solver(graph, q, expected):
    assert parse(q) is not None
    assert Solver(graph).answer(q) == expected


def test_unparseable_question_returns_none(graph):
    assert Solver(graph).answer("What color is the sky?") is None


def test_graph_text_serialization(graph):
    txt = graph.to_text()
    assert "small red rubber sphere" in txt and "distance 2.00 m" in txt
