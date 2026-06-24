#!/usr/bin/env python3
"""Generate figures for RESULTS.md using scorer JSON reports.

This intentionally uses Pillow rather than a heavier plotting stack, keeping the
results regeneration path small and easy to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("gpt-5.5", "openai-gpt-5.5"),
    ("gpt-5-nano", "openai-gpt-5-nano"),
    ("gpt-oss-20b@medium", "together-gpt-oss-20b-reasoning-medium"),
]

CLASSES = [
    "persuasion_attempt",
    "refusal",
    "neutral_information",
    "safe_redirection",
]
ABBR = {
    "persuasion_attempt": "PA",
    "refusal": "RF",
    "neutral_information": "NI",
    "safe_redirection": "SR",
}

BLUE = "#4C78A8"
RED = "#E45756"
ORANGE = "#F58518"
GREEN = "#54A24B"
DARK_BLUE = "#123E73"
MID_BLUE = "#6DAED6"
LIGHT_BLUE = "#DCEAF7"
GRID = "#D6D6D6"
TEXT = "#111111"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_SMALL = font(18)
FONT_MEDIUM = font(22)
FONT_LARGE = font(28)
FONT_TITLE = font(32)
FONT_BOLD = font(22, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: str = TEXT,
    text_font: ImageFont.ImageFont = FONT_MEDIUM,
) -> None:
    width, height = text_size(draw, text, text_font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, fill=fill, font=text_font)


def paste_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    *,
    angle: int,
    text_font: ImageFont.ImageFont = FONT_MEDIUM,
) -> None:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    width, height = text_size(dummy_draw, text, text_font)
    label = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((4, 4), text, fill=TEXT, font=text_font)
    rotated = label.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(rotated, xy, rotated)


def load_report(results_dir: Path, stem: str, split: str) -> dict[str, Any]:
    with (results_dir / f"{stem}-{split}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_all_reports(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        stem: {
            "core": load_report(results_dir, stem, "core"),
            "hard": load_report(results_dir, stem, "hard"),
        }
        for _, stem in MODELS
    }


def fig_core_vs_hard(data: dict[str, Any], out: Path) -> None:
    image = Image.new("RGB", (1100, 660), "white")
    draw = ImageDraw.Draw(image)
    draw.text((350, 24), "Easy set saturates; hard set separates", fill=TEXT, font=FONT_TITLE)

    left, top, right, bottom = 130, 85, 1050, 535
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)

    for tick in range(0, 101, 20):
        y = bottom - (tick / 100) * (bottom - top - 35)
        draw.line((left - 7, y, left, y), fill=TEXT, width=2)
        draw.text((50, y - 12), str(tick), fill=TEXT, font=FONT_MEDIUM)

    paste_rotated_text(image, (24, 260), "Accuracy (%)", angle=90, text_font=FONT_MEDIUM)

    group_width = (right - left) / len(MODELS)
    bar_width = 115
    max_height = bottom - top - 35
    for index, (name, stem) in enumerate(MODELS):
        center = left + group_width * index + group_width / 2
        values = [
            ("Core (64)", data[stem]["core"]["summary"]["accuracy"] * 100, BLUE),
            ("Hard (14)", data[stem]["hard"]["summary"]["accuracy"] * 100, RED),
        ]
        for offset, (_, value, color) in zip((-bar_width / 2, bar_width / 2), values):
            x0 = center + offset - bar_width / 2
            x1 = center + offset + bar_width / 2
            y0 = bottom - (value / 100) * max_height
            draw.rectangle((x0, y0, x1, bottom), fill=color)
            centered_text(draw, ((x0 + x1) / 2, y0 - 15), f"{value:.1f}", text_font=FONT_MEDIUM)
        centered_text(draw, (center, 590), name, text_font=FONT_MEDIUM)

    draw.rectangle((150, 620, 190, 636), fill=BLUE)
    draw.text((205, 613), "Core (64)", fill=TEXT, font=FONT_MEDIUM)
    draw.rectangle((340, 620, 380, 636), fill=RED)
    draw.text((395, 613), "Hard (14)", fill=TEXT, font=FONT_MEDIUM)
    image.save(out)


def matrix_color(value: int) -> str:
    if value >= 4:
        return DARK_BLUE
    if value == 3:
        return "#2B7BB9"
    if value == 2:
        return MID_BLUE
    if value == 1:
        return "#BBD2E8"
    return "#F2F7FC"


def fig_hard_confusion(data: dict[str, Any], out: Path) -> None:
    image = Image.new("RGB", (1850, 690), "white")
    draw = ImageDraw.Draw(image)
    centered_text(
        draw,
        (890, 34),
        "Hard-set confusion (off-diagonal = error)",
        text_font=FONT_TITLE,
    )

    cell = 108
    panel_gap = 180
    start_x = 110
    top = 125
    for panel_index, (name, stem) in enumerate(MODELS):
        x0 = start_x + panel_index * (cell * 4 + panel_gap)
        centered_text(draw, (x0 + cell * 2, 90), name, text_font=FONT_LARGE)
        confusion = data[stem]["hard"]["confusion_matrix"]

        for row, gold in enumerate(CLASSES):
            centered_text(draw, (x0 - 30, top + row * cell + cell / 2), ABBR[gold], text_font=FONT_MEDIUM)
            centered_text(draw, (x0 + row * cell + cell / 2, top + cell * 4 + 30), ABBR[gold], text_font=FONT_MEDIUM)
            for column, predicted in enumerate(CLASSES):
                value = confusion[gold][predicted]
                left = x0 + column * cell
                upper = top + row * cell
                draw.rectangle(
                    (left, upper, left + cell, upper + cell),
                    fill=matrix_color(value),
                    outline="white",
                    width=1,
                )
                off_diagonal = value != 0 and row != column
                centered_text(
                    draw,
                    (left + cell / 2, upper + cell / 2),
                    str(value),
                    fill=("white" if value >= 3 else (RED if off_diagonal else "#333333")),
                    text_font=FONT_BOLD if off_diagonal else FONT_MEDIUM,
                )
        draw.rectangle((x0, top, x0 + cell * 4, top + cell * 4), outline=TEXT, width=2)
        centered_text(draw, (x0 + cell * 2, top + cell * 4 + 72), "predicted", text_font=FONT_MEDIUM)
        paste_rotated_text(
            image,
            (x0 - 86, top + cell * 2 - 35),
            "gold",
            angle=90,
            text_font=FONT_MEDIUM,
        )

    image.save(out)


def rotated_label(text: str, text_font: ImageFont.ImageFont, angle: int = 15) -> Image.Image:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(dummy)
    width, height = text_size(draw, text, text_font)
    label = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((4, 4), text, fill=TEXT, font=text_font)
    return label.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def fig_failure_modes(data: dict[str, Any], out: Path) -> bool:
    tags = set()
    for _, stem in MODELS:
        tags |= set(data[stem]["hard"].get("error_tags", {}).keys())
    tags.discard("hard")
    tags = sorted(tags)
    if not tags:
        return False

    image = Image.new("RGB", (1140, 610), "white")
    draw = ImageDraw.Draw(image)
    centered_text(draw, (570, 38), "Where judges break: errors by challenge tag", text_font=FONT_TITLE)

    left, top, right, bottom = 110, 85, 1050, 465
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)

    max_y = 2.5
    for tick in [0, 0.5, 1, 1.5, 2, 2.5]:
        y = bottom - (tick / max_y) * (bottom - top)
        draw.line((left - 7, y, left, y), fill=TEXT, width=2)
        draw.text((50, y - 12), f"{tick:.1f}", fill=TEXT, font=FONT_SMALL)
    paste_rotated_text(image, (28, 225), "Hard-set errors", angle=90, text_font=FONT_MEDIUM)

    group_width = (right - left) / len(tags)
    bar_width = min(125, 0.75 * group_width / len(MODELS))
    colors = [BLUE, ORANGE, GREEN]
    for tag_index, tag in enumerate(tags):
        group_center = left + group_width * tag_index + group_width / 2
        label = rotated_label(tag, FONT_MEDIUM)
        image.paste(label, (int(group_center - label.width / 2), 485), label)
        for model_index, (name, stem) in enumerate(MODELS):
            value = data[stem]["hard"].get("error_tags", {}).get(tag, 0)
            x = group_center + (model_index - (len(MODELS) - 1) / 2) * bar_width
            y = bottom - (value / max_y) * (bottom - top)
            draw.rectangle(
                (x - bar_width / 2, y, x + bar_width / 2, bottom),
                fill=colors[model_index],
            )

    legend_x, legend_y = 760, 75
    for index, (name, _) in enumerate(MODELS):
        y = legend_y + index * 30
        draw.rectangle((legend_x, y, legend_x + 40, y + 16), fill=colors[index])
        draw.text((legend_x + 55, y - 4), name, fill=TEXT, font=FONT_MEDIUM)

    image.save(out)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all_reports(args.results_dir)

    fig_core_vs_hard(data, args.out_dir / "fig1_core_vs_hard.png")
    fig_hard_confusion(data, args.out_dir / "fig2_hard_confusion.png")
    has_modes = fig_failure_modes(data, args.out_dir / "fig3_failure_modes.png")

    written = "fig1_core_vs_hard.png, fig2_hard_confusion.png"
    if has_modes:
        written += ", fig3_failure_modes.png"
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
