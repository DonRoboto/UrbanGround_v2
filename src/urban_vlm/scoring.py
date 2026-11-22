"""Uncalibrated image-text logit aggregation (historical mean-of-five rule)."""
from __future__ import annotations
import math
from .data import CLASSES


def calculate_scores(logits: list[float], prompt_bank: dict[str, list[str]]) -> tuple[dict[str, float], str, float, list[dict]]:
    expected = sum(len(prompt_bank[k]) for k in CLASSES)
    if len(logits) != expected:
        raise ValueError(f"Expected {expected} model logits, got {len(logits)}")
    if any(not math.isfinite(float(x)) for x in logits):
        raise ValueError("Model returned a non-finite logit")
    per_prompt, sums = [], {}
    offset = 0
    for cls in CLASSES:
        texts = prompt_bank[cls]
        scores = [float(x) for x in logits[offset:offset + len(texts)]]
        sums[cls] = sum(scores) / len(scores)
        for index, (text, raw) in enumerate(zip(texts, scores), 1):
            per_prompt.append({"class": cls, "prompt_index": index, "prompt_text": text, "logit": raw})
        offset += len(texts)
    # Stable alphabetical class ordering provides deterministic tie resolution.
    pred = max(CLASSES, key=lambda cls: (sums[cls], -CLASSES.index(cls)))
    other = max(v for cls, v in sums.items() if cls != pred)
    return sums, pred, sums[pred] - other, per_prompt
