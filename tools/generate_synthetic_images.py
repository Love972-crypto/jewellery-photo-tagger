from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def make_image(tag: str, path: Path, blur: bool = False) -> None:
    width, height = 1100, 820
    image = Image.new("RGB", (width, height), "#efe7d8")
    draw = ImageDraw.Draw(image)

    draw.ellipse((255, 190, 845, 650), outline="#bd8a35", width=26)
    draw.ellipse((345, 280, 755, 560), outline="#f2d27b", width=10)
    for x in range(330, 790, 70):
        draw.ellipse((x, 665, x + 34, 699), fill="#d8a63f", outline="#805712")

    tag_box = (690, 98, 1010, 238)
    draw.rounded_rectangle(tag_box, radius=12, fill="#fbfaf6", outline="#ded6c6", width=4)
    try:
        font_big = ImageFont.truetype("arialbd.ttf", 54)
        font_small = ImageFont.truetype("arial.ttf", 21)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((735, 124), tag, fill="#17130e", font=font_big)
    draw.text((724, 194), f"{tag}FLDE1ICSB000", fill="#57514a", font=font_small)

    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=2.2))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("sample_data/synthetic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        tag = f"{121000 + index:06d}"
        make_image(tag, args.output / f"IMG_{index:04d}.jpg", blur=index % 17 == 0)
    print(f"Created {args.count} images in {args.output}")


if __name__ == "__main__":
    main()

