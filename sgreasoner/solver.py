"""Neuro-symbolic solver: parse a question into a program and execute it on a scene graph.

The grammar covers the spatial-scene-bench templates (counting, viewer-centric
relations, metric distance, nearest neighbour and egocentric direction).
Anything else falls through to :func:`sgreasoner.llm.answer_with_llm`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .graph import RELATION_PHRASES, SceneGraph

SIZES = {"small", "large"}
COLORS = {"red", "blue", "green", "yellow", "purple", "cyan", "gray", "brown"}
MATERIALS = {"rubber", "metal"}
SHAPES = {"sphere": "sphere", "spheres": "sphere", "cube": "cube", "cubes": "cube",
          "cylinder": "cylinder", "cylinders": "cylinder"}
EGO_OPTIONS = ["front-left", "front-right", "back-left", "back-right"]

_REL = "|".join(map(re.escape, RELATION_PHRASES))
_OPTS = re.compile(r"\(([A-D])\)\s*(.+?)(?=\s*\([A-D]\)|$)")

PATTERNS = [
    ("count", re.compile(r"^How many (.+?) are there\?$")),
    ("count_relation", re.compile(rf"^How many (.+?) are ({_REL}) the (.+?)\?$")),
    ("relation", re.compile(rf"^Is the (.+?) ({_REL}) the (.+?)\?$")),
    ("compare_count", re.compile(r"^Are there more (.+?) than (.+?)\?$")),
    ("closest", re.compile(r"^Which object is closest to the (.+?) in 3D space\?\s*(.*)$")),
    ("distance", re.compile(r"^What is the distance between the center of the (.+?) "
                            r"and the center of the (.+?), in meters\?$")),
    ("ego", re.compile(r"^Imagine you are standing at the (.+?) and facing the (.+?)\. "
                       r"Where is the (.+?) relative to you\?\s*(.*)$")),
    ("video_count", re.compile(r"^How many (.+?) are in this room\?")),
]


def parse_phrase(phrase: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for w in phrase.lower().split():
        if w in SIZES:
            attrs["size"] = w
        elif w in COLORS:
            attrs["color"] = w
        elif w in MATERIALS:
            attrs["material"] = w
        elif w in SHAPES:
            attrs["shape"] = SHAPES[w]
    return attrs


def parse_options(text: str) -> list[str]:
    return [m.group(2).strip() for m in _OPTS.finditer(text)]


@dataclass
class Program:
    op: str
    args: tuple


def parse(question: str) -> Program | None:
    q = question.strip()
    for op, pat in PATTERNS:
        m = pat.match(q)
        if m:
            return Program(op, m.groups())
    return None


class Solver:
    def __init__(self, graph: SceneGraph):
        self.g = graph

    def answer(self, question: str) -> str | None:
        prog = parse(question)
        if prog is None:
            return None
        return getattr(self, f"_{prog.op}")(*prog.args)

    def _count(self, phrase):
        return str(len(self.g.matching(parse_phrase(phrase))))

    _video_count = _count

    def _count_relation(self, phrase, rel_phrase, ref_phrase):
        ref = self.g.find(parse_phrase(ref_phrase))
        if ref is None:
            return "0"
        rel = RELATION_PHRASES[rel_phrase]
        cands = [o for o in self.g.matching(parse_phrase(phrase)) if o is not ref]
        return str(sum(self.g.holds(o, rel, ref) for o in cands))

    def _relation(self, a_phrase, rel_phrase, b_phrase):
        a, b = self.g.find(parse_phrase(a_phrase)), self.g.find(parse_phrase(b_phrase))
        if a is None or b is None:
            return "no"
        return "yes" if self.g.holds(a, RELATION_PHRASES[rel_phrase], b) else "no"

    def _compare_count(self, p1, p2):
        return "yes" if len(self.g.matching(parse_phrase(p1))) > len(self.g.matching(parse_phrase(p2))) else "no"

    def _closest(self, ref_phrase, opts_text):
        ref = self.g.find(parse_phrase(ref_phrase))
        options = parse_options(opts_text)
        if ref is None or not options:
            return "A"
        best, best_d = 0, float("inf")
        for i, opt in enumerate(options):
            o = self.g.find(parse_phrase(opt))
            if o is not None and o is not ref and self.g.distance(ref, o) < best_d:
                best, best_d = i, self.g.distance(ref, o)
        return "ABCD"[best]

    def _distance(self, a_phrase, b_phrase):
        a, b = self.g.find(parse_phrase(a_phrase)), self.g.find(parse_phrase(b_phrase))
        if a is None or b is None:
            return None
        return f"{self.g.distance(a, b):.2f}"

    def _ego(self, stand, face, target, opts_text):
        objs = [self.g.find(parse_phrase(p)) for p in (stand, face, target)]
        options = parse_options(opts_text) or EGO_OPTIONS
        if any(o is None for o in objs):
            return "A"
        quad = self.g.ego_quadrant(*objs)
        return "ABCD"[options.index(quad)] if quad in options else "A"
