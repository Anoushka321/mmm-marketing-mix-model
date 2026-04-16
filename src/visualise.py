"""
visualise.py
All charts for the MMM project — designed for portfolio & stakeholder presentation.
Saves publication-quality PNGs to outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Brand palette (Charlotte Tilbury-inspired) ──────────────────────────────
PALETTE = {
    "tv":           "#C9A96E",   # gold
    "paid_social":  "#8B4A6B",   # plum
    "influencer":   "#D4778A",   # rose
    "paid_search":  "#6B3A5C",   # deep purple
    "ooh":          "#B8860B",   # dark gold
    "email":        "#9B2335",   # crimson
    "base":         "#E8DDD0",   # warm cream
    "bg":           "#FAF7F4",   # off-white
    "text":         "#2C1810",   # dark brown
    "accent":       "#8B4A6B",
}

CH_COLORS = [
    PALETTE["tv"], PALETTE["paid_social"], PALETTE["influencer"],
    PALETTE["paid_search"], PALETTE["ooh"], PALETTE["email"],
]
CH_LABELS = ["TV", "Paid Social", "Influencer", "Paid Search", "OOH", "Email"]
SPEND_COLS = [
    "tv_spend", "paid_social_spend", "influencer_spend",
    "paid_search_spend", "ooh_spend", "email_spend",
]


def _apply_brand_style(ax, fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#D4C5B5")
    ax.grid(axis="y", color="#D4C5B5", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def gbp(x, pos=None):
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"£{x/1_000:.0f}K"
    return f"£{x:.0f}"


# ── 1. Revenue waterfall: base vs channels ──────────────────────────────────

def plot_revenue_decomposition(df_raw: pd.DataFrame, contrib: pd.DataFrame):
    """Stacked area chart showing revenue decomposition over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_brand_style(ax, fig)

    ch_cols = [c for c in contrib.columns if c != "intercept"]
    base_col = "intercept" if "intercept" in contrib.columns else None

    dates = pd.to_datetime(contrib.index)
    bottom = np.zeros(len(contrib))

    if base_col:
        ax.fill_between(dates, bottom, bottom + contrib[base_col],
                        color=PALETTE["base"], alpha=0.9, label="Base (trend + seasonality)")
        bottom += contrib[base_col].values

    for col, color, label in zip(SPEND_COLS, CH_COLORS, CH_LABELS):
        if col in contrib.columns:
            vals = contrib[col].values
            ax.fill_between(dates, bottom, bottom + vals,
                            color=color, alpha=0.85, label=label)
            bottom += vals

    # Actual revenue line
    ax.plot(pd.to_datetime(df_raw["date"]), df_raw["revenue"],
            color=PALETTE["text"], linewidth=1.2, linestyle="--",
            alpha=0.6, label="Actual revenue", zorder=5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(gbp))
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Revenue (£)", fontsize=11)
    ax.set_title("Revenue decomposition — base vs media channels",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
              facecolor=PALETTE["bg"], edgecolor="#D4C5B5")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_revenue_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_revenue_decomposition.png")


# ── 2. ROI comparison bar chart ─────────────────────────────────────────────

def plot_roi_comparison(roi_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_brand_style(ax, fig)

    bars = ax.barh(roi_df["channel"], roi_df["mean_roi"],
                   color=[PALETTE["paid_social"], PALETTE["influencer"],
                          PALETTE["tv"], PALETTE["paid_search"],
                          PALETTE["email"], PALETTE["ooh"]][:len(roi_df)],
                   edgecolor="none", height=0.6)

    # ROI = 1 break-even line
    ax.axvline(x=1, color=PALETTE["text"], linewidth=1.2,
               linestyle="--", alpha=0.5, label="Break-even (ROI = 1.0)")

    for bar, val in zip(bars, roi_df["mean_roi"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}×", va="center", fontsize=10,
                color=PALETTE["text"], fontweight="bold")

    ax.set_xlabel("Return on Investment (£ revenue per £ spent)", fontsize=11)
    ax.set_title("Channel ROI — every £1 spent returns...",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, facecolor=PALETTE["bg"], edgecolor="#D4C5B5")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_roi_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_roi_comparison.png")


# ── 3. Spend vs attributed revenue bubble chart ─────────────────────────────

def plot_spend_vs_revenue(roi_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_brand_style(ax, fig)

    sizes = roi_df["mean_roi"] * 400

    for i, row in roi_df.iterrows():
        color = CH_COLORS[i % len(CH_COLORS)]
        ax.scatter(row["total_spend_gbp"] / 1e6,
                   row["attributed_revenue_gbp"] / 1e6,
                   s=row["mean_roi"] * 500, color=color,
                   alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(row["channel"],
                    (row["total_spend_gbp"] / 1e6,
                     row["attributed_revenue_gbp"] / 1e6),
                    fontsize=9, color=PALETTE["text"],
                    xytext=(8, 4), textcoords="offset points")

    # Perfect efficiency line
    max_val = max(roi_df["total_spend_gbp"].max(),
                  roi_df["attributed_revenue_gbp"].max()) / 1e6 * 1.2
    ax.plot([0, max_val], [0, max_val], "--", color="#AAAAAA",
            linewidth=1, label="Break-even line (1:1)")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.1f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.1f}M"))
    ax.set_xlabel("Total spend (£M)", fontsize=11)
    ax.set_ylabel("Attributed revenue (£M)", fontsize=11)
    ax.set_title("Spend efficiency — bubble size = ROI",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, facecolor=PALETTE["bg"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_spend_vs_revenue.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_spend_vs_revenue.png")


# ── 4. Adstock decay visualisation ─────────────────────────────────────────

def plot_adstock_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_brand_style(ax, fig)

    decays = {
        "TV (0.60)":          (0.60, PALETTE["tv"]),
        "OOH (0.70)":         (0.70, PALETTE["ooh"]),
        "Paid Social (0.30)": (0.30, PALETTE["paid_social"]),
        "Influencer (0.45)":  (0.45, PALETTE["influencer"]),
        "Paid Search (0.15)": (0.15, PALETTE["paid_search"]),
        "Email (0.05)":       (0.05, PALETTE["email"]),
    }

    weeks = np.arange(12)
    for label, (decay, color) in decays.items():
        spend = np.zeros(12)
        spend[0] = 1.0
        ads = [spend[0]]
        for t in range(1, 12):
            ads.append(spend[t] + decay * ads[-1])
        ax.plot(weeks, ads, color=color, linewidth=2.5, label=label, marker="o",
                markersize=4)

    ax.set_xlabel("Weeks after campaign", fontsize=11)
    ax.set_ylabel("Remaining ad effect (normalised)", fontsize=11)
    ax.set_title("Adstock decay — how long does each channel's effect last?",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, facecolor=PALETTE["bg"], loc="upper right")
    ax.set_xlim(0, 11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_adstock_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_adstock_curves.png")


# ── 5. Saturation / response curves ─────────────────────────────────────────

def plot_saturation_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_brand_style(ax, fig)

    alphas = {
        "Email (steep)":         (4.0, PALETTE["email"]),
        "Paid Search":           (3.5, PALETTE["paid_search"]),
        "Paid Social":           (1.8, PALETTE["paid_social"]),
        "Influencer":            (2.0, PALETTE["influencer"]),
        "TV":                    (2.5, PALETTE["tv"]),
        "OOH (slow to saturate)":(1.5, PALETTE["ooh"]),
    }

    x = np.linspace(0, 1, 200)
    for label, (alpha, color) in alphas.items():
        y = x ** alpha / (x ** alpha + 0.5 ** alpha)
        ax.plot(x, y, color=color, linewidth=2.5, label=label)

    ax.axhline(0.9, color="#AAAAAA", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(0.02, 0.91, "90% saturation", fontsize=9, color="#999999")
    ax.set_xlabel("Relative spend (0 = zero, 1 = maximum observed)", fontsize=11)
    ax.set_ylabel("Marginal response", fontsize=11)
    ax.set_title("Saturation curves — diminishing returns per channel",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, facecolor=PALETTE["bg"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_saturation_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_saturation_curves.png")


# ── 6. Budget optimisation: current vs optimal ──────────────────────────────

def plot_budget_optimisation(roi_df: pd.DataFrame):
    """
    Simulates optimal re-allocation: push budget toward highest-ROI channels.
    """
    current = roi_df.set_index("channel")["total_spend_gbp"]
    total = current.sum()

    # Simple heuristic optimal: allocate proportional to ROI^1.5 (softmax-ish)
    roi_vals = roi_df.set_index("channel")["mean_roi"]
    weights = (roi_vals ** 1.5) / (roi_vals ** 1.5).sum()
    optimal = weights * total

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax in axes:
        _apply_brand_style(ax, fig)

    channels = current.index.tolist()
    x = np.arange(len(channels))
    w = 0.38

    # Current
    axes[0].bar(x, current.values / 1e3, width=0.6,
                color=[PALETTE["paid_social"]] * len(channels),
                edgecolor="none", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, rotation=30, ha="right", fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}K"))
    axes[0].set_title("Current budget allocation", fontsize=13, fontweight="bold")

    # Optimal
    bar_colors = []
    for ch, curr_val, opt_val in zip(channels, current.values, optimal.values):
        bar_colors.append(PALETTE["email"] if opt_val > curr_val else PALETTE["ooh"])

    axes[1].bar(x, optimal.values / 1e3, width=0.6,
                color=bar_colors, edgecolor="none", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels, rotation=30, ha="right", fontsize=9)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}K"))
    axes[1].set_title("ROI-optimised allocation (same total budget)",
                      fontsize=13, fontweight="bold")

    up_patch = mpatches.Patch(color=PALETTE["email"], label="Increase spend")
    down_patch = mpatches.Patch(color=PALETTE["ooh"], label="Decrease spend")
    axes[1].legend(handles=[up_patch, down_patch], fontsize=9,
                   facecolor=PALETTE["bg"])

    # Projected uplift annotation
    proj_current = (current * roi_vals).sum()
    proj_optimal = (optimal * roi_vals).sum()
    uplift_pct = (proj_optimal - proj_current) / proj_current * 100
    fig.suptitle(
        f"Budget optimisation → projected revenue uplift: +{uplift_pct:.1f}%",
        fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.02
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_budget_optimisation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 06_budget_optimisation.png  (projected uplift: +{uplift_pct:.1f}%)")


# ── 7. Seasonality heatmap ──────────────────────────────────────────────────

def plot_seasonality_heatmap(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.groupby(["year", "month"])["revenue"].mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        pivot / 1e6, ax=ax,
        cmap=sns.light_palette(PALETTE["accent"], as_cmap=True),
        fmt=".2f", annot=True, linewidths=0.5,
        linecolor="#D4C5B5",
        cbar_kws={"label": "Avg weekly revenue (£M)"},
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_title("Revenue seasonality — average weekly revenue by month",
                 fontsize=14, fontweight="bold", pad=15, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_seasonality_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_seasonality_heatmap.png")


# ── Run all ─────────────────────────────────────────────────────────────────

def run_all_charts(df_raw, contrib, roi_df):
    plot_revenue_decomposition(df_raw, contrib)
    plot_roi_comparison(roi_df)
    plot_spend_vs_revenue(roi_df)
    plot_adstock_curves()
    plot_saturation_curves()
    plot_budget_optimisation(roi_df)
    plot_seasonality_heatmap(df_raw)
    print("\nAll charts saved to outputs/")
