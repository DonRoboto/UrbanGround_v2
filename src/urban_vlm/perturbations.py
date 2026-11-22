"""Spatial diagnostics; NOT causal interpretability or an independent test set."""
from __future__ import annotations
from PIL import Image, ImageFilter
from .data import crop_roi


def make_variants(image: Image.Image, box: tuple[int,int,int,int], blur_radius: float = 24.0) -> dict[str, Image.Image]:
    if blur_radius <= 0:
        raise ValueError("blur_radius must be > 0")
    roi = crop_roi(image, box)
    blur = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    x1,y1,x2,y2 = box
    upper = image.copy()
    upper.paste(blur.crop((0,0,image.width,int(image.height*.45))), (0,0))
    inside = image.copy()
    inside.paste(blur.crop(box), (x1,y1))
    outside = blur.copy()
    outside.paste(image.crop(box), (x1,y1))
    return {
        "original": image,
        "roi_crop": roi,
        "blur_upper": upper,
        "blur_inside_roi": inside,
        "blur_outside_roi": outside,
    }
