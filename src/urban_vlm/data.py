"""Strongly validated study inputs. All coordinates use Pillow's half-open boxes."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml
from PIL import Image, ImageOps

CLASSES = ("A", "I", "N")
EXPECTED_NAMES = tuple(f"fachada_{i:02d}.jpg" for i in range(1, 14))


def load_config(config_file: str | Path) -> dict[str, Any]:
    p = Path(config_file).resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    project_root = p.parent.parent
    for field in ("images_dir", "annotations", "rois"):
        path = Path(cfg[field])
        cfg[field] = str((project_root / path).resolve() if not path.is_absolute() else path)
    cfg["prompts"] = {
        bank: str((project_root / name).resolve() if not Path(name).is_absolute() else Path(name))
        for bank, name in cfg["prompts"].items()
    }
    cfg["config_path"] = str(p)
    cfg["project_root"] = str(project_root)
    if not cfg["model_id"]:
        raise ValueError("Missing model_id")
    if sorted(cfg["regions"]) != ["full", "roi"]:
        raise ValueError("Expect exactly regions: full and roi")
    return cfg


def read_annotations(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No annotation rows")
    result = {}
    for row in rows:
        name = row["file"].strip()
        label = row["label"].strip().upper()
        if name in result:
            raise ValueError(f"Duplicate annotation: {name}")
        if label not in CLASSES:
            raise ValueError(f"Bad label for {name}: {label!r}")
        if not row.get("evidence", "").strip():
            raise ValueError(f"Missing visual evidence for {name}")
        result[name] = row
        result[name]["label"] = label
    return result


def read_rois(path: str | Path) -> dict[str, tuple[int, int, int, int]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        name = row["file"].strip()
        if name in result:
            raise ValueError(f"Duplicate ROI: {name}")
        box = tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2"))
        if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Bad ROI for {name}: {box}")
        result[name] = box
    return result


def read_prompt_bank(path: str | Path) -> dict[str, list[str]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if set(data) != set(CLASSES):
        raise ValueError(f"Prompt bank {p.name} must contain A/I/N")
    if any(len(data[key]) != 5 for key in CLASSES):
        raise ValueError("Exactly five prompts are required per class for historical comparability")
    if any(not item.strip() for key in CLASSES for item in data[key]):
        raise ValueError("Empty prompt")
    return data


def read_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def crop_roi(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = image.size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"ROI {box} outside image {image.size}; use original 640x640 or correct ROI")
    return image.crop(box)


def validate_inputs(cfg: dict, require_images: bool = True) -> dict:
    anns = read_annotations(cfg["annotations"])
    rois = read_rois(cfg["rois"])
    for name in cfg["prompts"].values():
        read_prompt_bank(name)
    names = set(anns)
    if cfg.get("strict_expected_images", True) and names != set(EXPECTED_NAMES):
        raise ValueError(f"Expected all 13 canonical names; missing={sorted(set(EXPECTED_NAMES)-names)}; extra={sorted(names-set(EXPECTED_NAMES))}")
    if names != set(rois):
        raise ValueError(f"ROI/annotation mismatch: missing ROI {sorted(names-set(rois))}, extra ROI {sorted(set(rois)-names)}")
    images = Path(cfg["images_dir"])
    dimensions = {}
    if require_images:
        if not images.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images}")
        for name in sorted(names):
            path = images / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")
            img = read_image(path)
            crop_roi(img, rois[name])
            dimensions[name] = list(img.size)
            expected_size = cfg.get("expected_image_size")
            if expected_size and list(img.size) != list(expected_size):
                raise ValueError(f"Image {name} has size {img.size}; expected {tuple(expected_size)}. Original ROIs cannot be reused on resized images.")
    return {"image_count": len(names), "annotation_counts": {c: sum(v["label"] == c for v in anns.values()) for c in CLASSES}, "image_dimensions": dimensions}


def file_sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: str | Path, records: list[dict], fields: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
