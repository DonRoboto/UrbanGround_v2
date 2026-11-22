"""Reviewable hand-annotated ROI previews, without AI inference."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .data import read_image, read_rois, read_annotations, crop_roi


def create_roi_preview(cfg: dict, output: str | Path) -> Path:
    rois = read_rois(cfg["rois"])
    anns = read_annotations(cfg["annotations"])
    cellw, cellh = 540, 560
    canvas = Image.new("RGB", (cellw*3, cellh*5), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(sorted(anns)):
        img = read_image(Path(cfg["images_dir"]) / name)
        box = rois[name]
        crop = crop_roi(img,box)
        display = img.copy()
        ImageDraw.Draw(display).rectangle(box, outline="#ef4444", width=6)
        display.thumbnail((250, 460))
        crop.thumbnail((255, 460))
        x,y = (i%3)*cellw, (i//3)*cellh
        canvas.paste(display,(x+5,y+45))
        canvas.paste(crop,(x+275,y+45))
        draw.text((x+8,y+12), f'{name} / {anns[name]["label"]}', fill="#111827")
    p=Path(output)
    p.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(p)
    return p
