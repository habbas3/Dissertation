#!/usr/bin/env python3
"""Generate an ablation ladder schematic as an SVG without external dependencies."""

from __future__ import annotations

from pathlib import Path

OUTPUT_PATH = Path("figures/dissertation_plots/ablation_ladder_schematic.svg")


def make_svg() -> str:
    labels = [
        "Zhao baseline",
        "+ OpenMax",
        "+ SNGP",
        "+ SA",
        "LLM-selected composition",
    ]
    fills = ["#e5e7eb", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa"]

    width, height = 1800, 420
    box_w, box_h = 280, 150
    x0, y0, gap = 70, 120, 45

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(
        '<text x="900" y="58" text-anchor="middle" font-size="44" font-weight="700" fill="#111827" '
        'font-family="Arial, Helvetica, sans-serif">Ablation Ladder Schematic</text>'
    )

    for i, (label, fill) in enumerate(zip(labels, fills, strict=True)):
        x = x0 + i * (box_w + gap)
        parts.append(
            f'<rect x="{x}" y="{y0}" width="{box_w}" height="{box_h}" rx="22" ry="22" '
            f'fill="{fill}" stroke="#1f2937" stroke-width="3"/>'
        )

        if label == "LLM-selected composition":
            parts.append(
                f'<text x="{x + box_w / 2}" y="{y0 + box_h / 2 - 10}" text-anchor="middle" '
                'font-size="32" font-weight="700" fill="#111827" font-family="Arial, Helvetica, sans-serif">'
                "LLM-selected</text>"
            )
            parts.append(
                f'<text x="{x + box_w / 2}" y="{y0 + box_h / 2 + 34}" text-anchor="middle" '
                'font-size="32" font-weight="700" fill="#111827" font-family="Arial, Helvetica, sans-serif">'
                "composition</text>"
            )
        else:
            parts.append(
                f'<text x="{x + box_w / 2}" y="{y0 + box_h / 2 + 10}" text-anchor="middle" '
                'font-size="34" font-weight="700" fill="#111827" font-family="Arial, Helvetica, sans-serif">'
                f"{label}</text>"
            )

        if i < len(labels) - 1:
            x1 = x + box_w + 10
            x2 = x + box_w + gap - 10
            y = y0 + box_h / 2
            parts.append(
                f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#374151" stroke-width="4"/>'
            )
            parts.append(
                f'<polygon points="{x2},{y} {x2-20},{y-10} {x2-20},{y+10}" fill="#374151"/>'
            )

    parts.append(
        '<text x="900" y="350" text-anchor="middle" font-size="26" fill="#374151" '
        'font-family="Arial, Helvetica, sans-serif">'
        'Incremental additions from baseline to final LLM-selected composition.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(make_svg(), encoding="utf-8")
    print(f"Saved ablation ladder figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()