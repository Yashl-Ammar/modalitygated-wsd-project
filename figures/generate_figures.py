import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

# ── shared rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "figure.dpi":         300,
    "pdf.fonttype":       42,   # embed fonts as TrueType
    "ps.fonttype":        42,
})

OUT = os.path.dirname(__file__)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Grouped Bar Chart (Mixed Condition Results)
# ════════════════════════════════════════════════════════════════════════════
def fig_mixed_conditions():
    conditions  = ["30% Helpful", "50% Helpful", "70% Helpful"]
    bert        = [71.58, 71.58, 71.58]
    clip_mm     = [44.05, 56.55, 69.66]
    gate        = [73.75, 75.89, 77.85]

    colors = {
        "bert":  "#4C72B0",
        "clip":  "#DD8452",
        "gate":  "#55A868",
    }

    x     = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bars_bert = ax.bar(x - width, bert,   width, label="BERT (text-only)",
                       color=colors["bert"],  edgecolor="white", linewidth=0.4)
    bars_clip = ax.bar(x,         clip_mm, width, label="CLIP Multimodal",
                       color=colors["clip"],  edgecolor="white", linewidth=0.4)
    bars_gate = ax.bar(x + width, gate,   width, label="Gate Ensemble (ours)",
                       color=colors["gate"],  edgecolor="white", linewidth=0.4,
                       hatch="//", zorder=3)

    # value labels
    for bars in (bars_bert, bars_clip, bars_gate):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=6.5, color="#333333",
            )

    ax.set_ylabel("F1 Score (%)")
    ax.set_xlabel("Image Mixture Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(30, 92)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    ax.legend(loc="upper left", frameon=False,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.3)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(pad=0.4)

    base = os.path.join(OUT, "fig_mixed_conditions")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Oracle Gap Line Chart
# ════════════════════════════════════════════════════════════════════════════
def fig_oracle_gap():
    labels   = ["100% Irrel.", "30% Help.", "50% Help.", "70% Help.", "100% Help."]
    x        = np.arange(len(labels))

    gate     = np.array([70.55, 73.75, 75.89, 77.85, 80.98])
    oracle   = np.array([76.96, 81.45, 84.66, 87.59, 92.75])
    bert_bl  = np.array([71.58, 71.58, 71.58, 71.58, 71.58])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # shaded oracle gap
    ax.fill_between(x, gate, oracle, alpha=0.15, color="#1f77b4", zorder=1)

    # lines
    ax.plot(x, oracle,  color="#2ca02c", marker="o", markersize=4,
            linewidth=1.4, label="Oracle", zorder=3)
    ax.plot(x, gate,    color="#1f77b4", marker="s", markersize=4,
            linewidth=1.4, label="Gate Ensemble (ours)", zorder=3)
    ax.plot(x, bert_bl, color="#7f7f7f", linestyle="--", linewidth=1.1,
            label="BERT (text-only)", zorder=2)

    # double-headed arrow + label at 50% Help. (index 2)
    gap_x   = 2
    gap_bot = gate[gap_x]
    gap_top = oracle[gap_x]
    ax.annotate(
        "",
        xy=(gap_x + 0.18, gap_top),
        xytext=(gap_x + 0.18, gap_bot),
        arrowprops=dict(
            arrowstyle="<->",
            color="#333333",
            lw=0.9,
        ),
        zorder=4,
    )
    ax.text(
        gap_x + 0.22, (gap_top + gap_bot) / 2,
        "8.77pp\noracle gap",
        fontsize=6.5, va="center", ha="left", color="#333333",
        linespacing=1.3,
    )

    ax.set_ylabel("F1 Score (%)")
    ax.set_xlabel("Retrieval Quality →")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(60, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    ax.legend(loc="upper left", frameon=False,
              handlelength=1.6, handletextpad=0.4, labelspacing=0.3)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(pad=0.4)

    base = os.path.join(OUT, "fig_oracle_gap")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Gate Triggering Rate vs Image Condition
# ════════════════════════════════════════════════════════════════════════════
def fig_gate_triggering():
    labels  = ["100% Irrel.", "30% Help.", "50% Help.", "70% Help.", "100% Help."]
    x       = np.arange(len(labels))

    routing = np.array([35.2, 37.8, 40.1, 43.3, 46.1])
    true_uh = np.array([ 3.1,  5.8,  8.4,  9.7, 11.6])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # shaded over-triggering gap
    ax.fill_between(x, true_uh, routing, alpha=0.12, color="#d62728", zorder=1)

    ax.plot(x, routing, color="#1f77b4", marker="s", markersize=4,
            linewidth=1.4, label="Gate routing rate", zorder=3)
    ax.plot(x, true_uh, color="#d62728", marker="o", markersize=4,
            linestyle="--", linewidth=1.4, label="True U+H rate (benefit)", zorder=3)

    # annotation inside shaded gap — placed at x=2 (50% Help.), midpoint
    mid_y = (routing[2] + true_uh[2]) / 2
    ax.text(2, mid_y, "Over-triggering gap",
            ha="center", va="center", fontsize=8,
            color="#d62728", style="italic")

    ax.set_ylabel("Instances Routed to Multimodal (%)")
    ax.set_xlabel("Image Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    ax.legend(loc="upper left", frameon=False,
              handlelength=1.6, handletextpad=0.4, labelspacing=0.3)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(pad=0.4)

    base = os.path.join(OUT, "fig_gate_triggering")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Image Condition Examples (2x2 text panels)
# ════════════════════════════════════════════════════════════════════════════
def fig_image_examples():
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 3.5))
    fig.subplots_adjust(wspace=0.08, hspace=0.08)

    col_titles = ["Target Word: 'bank'", "Target Word: 'crane'"]
    row_labels = ["Helpful Image", "Irrelevant Image"]

    panel_data = [
        [
            {
                "bg": "#d4edda", "border": "#28a745",
                "corner": "✓", "corner_color": "#28a745",
                "text": "'bank'\n\"The bank was\ncrowded today.\"\n\n Image: building\nwith vault doors\n→ financial institution",
            },
            {
                "bg": "#d4edda", "border": "#28a745",
                "corner": "✓", "corner_color": "#28a745",
                "text": "'crane'\n\"The crane lifted\nthe steel beam.\"\n\n Image: yellow\nconstruction crane\n→ machine (not bird)",
            },
        ],
        [
            {
                "bg": "#f8d7da", "border": "#dc3545",
                "corner": "✗", "corner_color": "#dc3545",
                "text": "'bank'\n\"The bank was\ncrowded today.\"\n\n Image: random\nsunset landscape\n→ no signal",
            },
            {
                "bg": "#f8d7da", "border": "#dc3545",
                "corner": "✗", "corner_color": "#dc3545",
                "text": "'crane'\n\"The crane lifted\nthe steel beam.\"\n\n Image: random\nfood photograph\n→ no signal",
            },
        ],
    ]

    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            d  = panel_data[r][c]

            # colored background rectangle
            ax.set_facecolor(d["bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(d["border"])
                spine.set_linewidth(1.4)
                spine.set_visible(True)

            ax.set_xticks([])
            ax.set_yticks([])

            # main panel text
            ax.text(0.5, 0.52, d["text"],
                    ha="center", va="center",
                    fontsize=8, linespacing=1.55,
                    transform=ax.transAxes)

            # corner symbol
            ax.text(0.95, 0.93, d["corner"],
                    ha="right", va="top", fontsize=10,
                    color=d["corner_color"], fontweight="bold",
                    transform=ax.transAxes)

            # column headers (top row only)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9, fontweight="bold", pad=5)

            # row labels (left column only)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, fontweight="bold", labelpad=6)

    # caption below figure
    fig.text(
        0.5, -0.04,
        "Helpful images share the word's sense embedding space;\n"
        "irrelevant images inject noise into cross-attention.",
        ha="center", va="top", fontsize=7.5, color="#444444",
        linespacing=1.4,
    )

    base = os.path.join(OUT, "fig_image_examples")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf / .png")


if __name__ == "__main__":
    fig_mixed_conditions()
    fig_oracle_gap()
    fig_gate_triggering()
    fig_image_examples()
    print("Done.")
