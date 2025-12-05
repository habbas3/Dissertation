import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Slide 6: EOL Timeline Figure (Improved)
# -----------------------------

# Fake SOH curve generator (for illustration only)
cycles = np.linspace(0, 3000, 3000)
soh = 1.0 - 0.00025 * cycles + 0.02 * np.sin(cycles / 250)
soh = np.clip(soh, 0.5, 1.05)

# EOL cycle (for annotation)
eol_cycle = 2200

# Create two-panel figure
fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 6),
    gridspec_kw={"height_ratios": [2.2, 1]},
    constrained_layout=True
)

# ============================================================
# TOP PANEL – FULL LIFETIME CURVE
# ============================================================

ax_top = axes[0]
ax_top.plot(cycles, soh, lw=2.5, color="black")
ax_top.set_ylabel("SOH", fontsize=12)
ax_top.set_title("Full Lifetime Curve with Early-Life Windows and EOL Labeling", fontsize=13)

# Shade early-life region (0–15 cycles)
ax_top.axvspan(0, 15, color="#D0F0C0", alpha=0.5)
ax_top.text(
    7.5, 1.03,
    "Early-life region (first 15 cycles)",
    ha="center", va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none")
)

# Add horizontal EOL threshold at 75% SOH
ax_top.axhline(0.75, color="red", linestyle="--", lw=1.8)
ax_top.text(
    50, 0.755,
    "EOL Threshold (75% SOH)",
    color="red", fontsize=10, ha="left", va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none")
)

# Mark computed EOL cycle
ax_top.axvline(eol_cycle, color="red", linestyle="--", lw=1.8)
ax_top.text(
    eol_cycle + 100, 0.65,
    "Computed EOL cycle\n(EOL class used for all\nearly-life windows)",
    color="red", fontsize=10,
    ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none")
)

ax_top.set_ylim(0.55, 1.08)
ax_top.set_xlim(0, 3000)

# ============================================================
# BOTTOM PANEL – EARLY-LIFE WINDOW SEGMENTATION (ZOOM)
# ============================================================

ax_bot = axes[1]

# Zoom 0–20 cycles
zoom_mask = (cycles >= 0) & (cycles <= 20)
ax_bot.plot(cycles[zoom_mask], soh[zoom_mask], lw=2, color="black")
ax_bot.set_xlabel("Cycle Number", fontsize=12)
ax_bot.set_ylabel("SOH", fontsize=12)
ax_bot.set_title("Early-Life Windows (all inherit the same EOL class)", fontsize=13)

# Red horizontal EOL threshold for consistency
ax_bot.axhline(0.75, color="red", linestyle="--", lw=1.5)
ax_bot.text(
    0.5, 0.755,
    "75% SOH\n(EOL threshold)",
    color="red", fontsize=9,
    ha="left", va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none")
)

# Define windows
window_starts = [1, 6, 11]
window_width = 3
y_base = 0.78
height = 0.15

for i, start in enumerate(window_starts, 1):
    rect = patches.Rectangle(
        (start, y_base),
        window_width,
        height,
        linewidth=1.8,
        edgecolor="blue",
        facecolor="none"
    )
    ax_bot.add_patch(rect)
    ax_bot.text(
        start + window_width / 2,
        y_base + height / 2,
        f"Window {i}",
        ha="center", va="center",
        fontsize=10, color="blue"
    )

# Explanation annotation
ax_bot.annotate(
    "Each early-life window is\nassigned the same EOL class\nbased on the full cycle life",
    xy=(window_starts[1] + 1.5, y_base + height),
    xytext=(15, 0.95),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", lw=1.8),
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none")
)

ax_bot.set_ylim(0.7, 1.03)
ax_bot.set_xlim(0, 20)

# ============================================================
# SAVE IMAGES
# ============================================================

plt.savefig("slide6_eol_timeline_with_threshold.png", dpi=350)
plt.savefig("slide6_eol_timeline_with_threshold.svg", dpi=350)

print("Saved: slide6_eol_timeline_with_threshold.png and slide6_eol_timeline_with_threshold.svg")
