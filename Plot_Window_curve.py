#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:26:01 2025

@author: habbas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Slide 9: Cathode Family Network
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Cathode Family Transfer Graph", fontsize=12)
ax.axis("off")

# Positions for families and example members
positions = {
    # Family nodes
    "nmc":      (4, 4.5),
    "lirich":   (1.5, 2.0),
    "spinel5v": (6.5, 2.0),
    "fcg":      (4, 0.7),

    # Example NMC members
    "HE5050": (2.5, 5.6),
    "NMC111": (4,   5.9),
    "NMC532": (5.5, 5.6),

    # Example Li-rich members
    "Li1.2Ni0.3Mn0.6O2":   (0.5, 1.0),
    "Li1.35Ni0.33Mn0.67O2.35": (2.5, 1.0),

    # Example spinel member
    "5Vspinel": (6.5, 1.0)
}

# Helper to draw a node
def draw_node(ax, name, xy, family=False):
    x, y = xy
    if family:
        # Family nodes as rounded rectangles
        box = patches.FancyBboxPatch(
            (x - 0.9, y - 0.35), 1.8, 0.7,
            boxstyle="round,pad=0.2",
            facecolor="#FFEFC2",
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold")
    else:
        # Member nodes as circles
        circ = patches.Circle(
            (x, y), 0.25,
            facecolor="#E8F0FF",
            edgecolor="black",
            linewidth=1.0
        )
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", fontsize=8)

# Draw family nodes
draw_node(ax, "NMC family", positions["nmc"], family=True)
draw_node(ax, "Li-rich\nfamily", positions["lirich"], family=True)
draw_node(ax, "5Vspinel\nfamily", positions["spinel5v"], family=True)
draw_node(ax, "FCG\nfamily", positions["fcg"], family=True)

# Draw example members (optional, for context)
member_keys = [
    "HE5050", "NMC111", "NMC532",
    "Li1.2Ni0.3Mn0.6O2", "Li1.35Ni0.33Mn0.67O2.35",
    "5Vspinel"
]
for k in member_keys:
    draw_node(ax, k, positions[k], family=False)

# Draw connecting lines from family to members (no arrows)
family_members = {
    "nmc": ["HE5050", "NMC111", "NMC532"],
    "lirich": ["Li1.2Ni0.3Mn0.6O2", "Li1.35Ni0.33Mn0.67O2.35"],
    "spinel5v": ["5Vspinel"]
}
for fam, members in family_members.items():
    fx, fy = positions[fam]
    for m in members:
        mx, my = positions[m]
        ax.annotate(
            "",
            xy=(mx, my + 0.25),
            xytext=(fx, fy - 0.4 if fam == "nmc" else fy + 0.4),
            arrowprops=dict(arrowstyle="-", lw=1, color="gray")
        )

# Allowed transfer edges (family-level)
edges = [
    ("fcg", "nmc"),
    ("lirich", "spinel5v"),
    ("nmc", "fcg"),
    ("nmc", "spinel5v"),
    ("spinel5v", "lirich"),
    ("spinel5v", "nmc")
]

for src, dst in edges:
    sx, sy = positions[src]
    dx, dy = positions[dst]
    ax.annotate(
        "",
        xy=(dx, dy),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.8,
            color="darkred"
        )
    )

# Adjust axes limits to fit everything nicely
ax.set_xlim(-0.5, 8)
ax.set_ylim(0, 6.5)

plt.savefig("slide9_cathode_network.png", dpi=300)
plt.savefig("slide9_cathode_network.svg", format="svg", dpi=300)
plt.close(fig)

print("Saved: slide9_cathode_network.png and slide9_cathode_network.svg")
