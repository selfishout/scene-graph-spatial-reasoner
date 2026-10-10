"""Scene-graph prompting: answer free-form spatial questions with an LLM/VLM.

The 3D scene graph is serialized to text and given to any OpenAI-compatible
chat endpoint (OpenAI, vLLM, SGLang, Ollama). Passing the image as well turns
this into "graph-augmented" VLM prompting.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from .graph import SceneGraph

SYSTEM = (
    "You answer spatial questions about a 3D scene. You are given a list of detected objects with "
    "metric 3D coordinates and pairwise relations estimated by a perception pipeline. Use the "
    "coordinates for distances and directions; 'left/right/in front of/behind' refer to the camera view "
    "unless the question defines another viewpoint. Think step by step inside <think></think> and give "
    "the final short answer inside <answer></answer>."
)


def build_messages(graph: SceneGraph, question: str, image: Image.Image | None = None) -> list[dict]:
    text = f"{graph.to_text()}\n\nQuestion: {question}"
    content: list[dict] = []
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def answer_with_llm(graph: SceneGraph, question: str, model: str, image: Image.Image | None = None,
                    base_url: str | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url)
    resp = client.chat.completions.create(model=model, messages=build_messages(graph, question, image),
                                          temperature=0.0, max_tokens=1024)
    return resp.choices[0].message.content or ""
