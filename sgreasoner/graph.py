"""3D scene graph with viewer-centric and allocentric spatial relations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lifting import Object3D

RELATION_PHRASES = {
    "to the left of": "left",
    "to the right of": "right",
    "in front of": "front",
    "behind": "behind",
}


@dataclass
class SceneGraph:
    objects: list[Object3D]
    K: np.ndarray | None = None
    cam_to_world: np.ndarray | None = None

    def matching(self, attrs: dict[str, str]) -> list[Object3D]:
        return [o for o in self.objects if all(o.attrs.get(k) == v for k, v in attrs.items())]

    def find(self, attrs: dict[str, str]) -> Object3D | None:
        """Resolve a referring expression; ties go to the best-observed object."""
        hits = self.matching(attrs)
        return max(hits, key=lambda o: o.pixels) if hits else None

    # -------------------------------------------------------- viewer-centric
    def _to_camera(self, p: np.ndarray) -> np.ndarray:
        T = self.cam_to_world
        return (p - T[:3, 3]) @ T[:3, :3]

    def viewer_margins(self, a: Object3D, b: Object3D) -> dict[str, float]:
        """Signed evidence for 'a <rel> b' in the image (pixels for left/right, meters for depth)."""
        pa, pb = self._to_camera(a.center), self._to_camera(b.center)
        ua = self.K[0, 0] * pa[0] / pa[2]
        ub = self.K[0, 0] * pb[0] / pb[2]
        return {"left": ub - ua, "right": ua - ub, "front": pb[2] - pa[2], "behind": pa[2] - pb[2]}

    def holds(self, a: Object3D, rel: str, b: Object3D) -> bool:
        return self.viewer_margins(a, b)[rel] > 0

    # ------------------------------------------------------------ allocentric
    @staticmethod
    def distance(a: Object3D, b: Object3D) -> float:
        return float(np.linalg.norm(a.center - b.center))

    @staticmethod
    def ego_quadrant(stand: Object3D, face: Object3D, target: Object3D) -> str:
        fwd = (face.center - stand.center)[:2]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        right = np.array([fwd[1], -fwd[0]])
        rel = (target.center - stand.center)[:2]
        return f"{'front' if rel @ fwd > 0 else 'back'}-{'right' if rel @ right > 0 else 'left'}"

    # --------------------------------------------------------------- prompts
    def to_text(self, max_relations: int = 40) -> str:
        """Compact serialization for scene-graph prompting of an LLM/VLM."""
        lines = ["Objects (world frame, meters, z is up):"]
        for i, o in enumerate(self.objects):
            c, e = o.center, o.extent
            lines.append(f"  [{i}] {o.label()}: center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                         f"size=({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f})")
        if self.K is not None and self.cam_to_world is not None and len(self.objects) > 1:
            lines.append("Viewer-centric relations:")
            n = 0
            for i, a in enumerate(self.objects):
                for j, b in enumerate(self.objects):
                    if i >= j or n >= max_relations:
                        continue
                    m = self.viewer_margins(a, b)
                    lr = "left of" if m["left"] > 0 else "right of"
                    fb = "in front of" if m["front"] > 0 else "behind"
                    lines.append(f"  [{i}] is {lr} and {fb} [{j}]; distance {self.distance(a, b):.2f} m")
                    n += 1
        return "\n".join(lines)
