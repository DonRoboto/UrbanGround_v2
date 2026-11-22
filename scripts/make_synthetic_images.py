#!/usr/bin/env python3
"""Generate non-realistic images solely to exercise image IO and CLI validate (no model performance)."""
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    out = Path("data/synthetic_images")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1,14):
        img = Image.new("RGB",(640,640),(170+i*3,180,190))
        d=ImageDraw.Draw(img)
        d.rectangle((90,50,550,580),fill=(210,206,200),outline=(50,50,60),width=4)
        d.rectangle((190,320,450,580),fill=(80,80,95),outline="white",width=5)
        d.text((210,350),f"SYNTHETIC {i:02d}",fill="white")
        img.save(out/f"fachada_{i:02d}.jpg",quality=90)
    print("Synthetic images created at",out)
    print("Run with an images_dir override in a COPY of configs/experiment.yaml; never evaluate these as real data.")

if __name__ == "__main__": main()
