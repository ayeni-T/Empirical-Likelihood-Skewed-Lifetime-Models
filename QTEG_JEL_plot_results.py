"""
QTEG_JEL_plot_results.py
========================
Generates paper-ready and supplementary figures from merged simulation
results and real data analysis.

Each figure is saved as both PDF (vector, for journal submission) and
PNG (300 dpi, for Word documents / preview).

PAPER FIGURES (4 total):
  fig_jel_sim_alpha.pdf/.png   -- CP for alpha:  1x3 panels
  fig_jel_sim_beta.pdf/.png    -- CP for beta:   1x3 panels
  fig_jel_sim_surv.pdf/.png    -- CP for S(t0):  1x3 panels
  fig_jel_realdata.pdf/.png    -- Real data CIs: 3x2 line-plot grid

SUPPLEMENTARY FIGURES:
  fig_jel_alpha_al.pdf/.png                 -- AL for alpha
  fig_jel_beta_al.pdf/.png                  -- AL for beta
  fig_jel_survival_al.pdf/.png              -- AL for S(t0)
  fig_jel_alpha_cp_all_nominals.pdf/.png    -- CP at 90/95/99% for alpha
  fig_jel_survival_cp_all_nominals.pdf/.png -- CP at 90/95/99% for S(t0)

Run after merge and realdata:
  python QTEG_JEL_Arctic.py --merge
  python QTEG_JEL_Arctic.py --realdata
  python QTEG_JEL_plot_results.py

Authors: Taiwo Michael Ayeni and Yichuan Zhao
Georgia State University, 2026
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
CSV_FILE      = os.path.join(RESULTS_DIR, "QTEG_JEL_full_results.csv")
REALDATA_FILE = os.path.join(RESULTS_DIR, "QTEG_JEL_realdata_results.json")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find {CSV_FILE}\n"
        "Run: python QTEG_JEL_Arctic.py --merge  first."
    )

df = pd.read_csv(CSV_FILE)
SCENARIOS_NUM = sorted(df["scenario"].unique())
NS            = sorted(df["n"].unique())

SC_LABELS = {
    1: r"Sc.$\,$1  ($\alpha=1.5,\;\beta=0.5$)",
    2: r"Sc.$\,$2  ($\alpha=2.0,\;\beta=1.0$)",
    3: r"Sc.$\,$3  ($\alpha=3.0,\;\beta=2.0$)",
}

DS_TICKS  = ["DS1", "DS2", "DS3", "DS4"]
DS_KEYS   = [
    "DS1: Bladder Cancer (n=128)",
    "DS2: Boeing 720 (n=213)",
    "DS3: Malignant Melanoma (n=205)",
    "DS4: Guinea Pig Survival (n=72)",
]

# ── Global RC — journal quality ────────────────────────────────────────
plt.rcParams.update({
    "font.family":            "serif",
    "font.serif":             ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":              9,
    "axes.titlesize":         9,
    "axes.titlepad":          8,
    "axes.labelsize":         9,
    "axes.labelpad":          5,
    "xtick.labelsize":        8,
    "ytick.labelsize":        8,
    "xtick.major.pad":        4,
    "ytick.major.pad":        4,
    "legend.fontsize":        8,
    "legend.framealpha":      0.8,
    "legend.edgecolor":       "0.75",
    "legend.handlelength":    2.4,
    "legend.borderpad":       0.5,
    "figure.titlesize":       10,
    "figure.titleweight":     "normal",
    "axes.grid":              True,
    "grid.alpha":             0.3,
    "grid.linestyle":         "--",
    "grid.linewidth":         0.5,
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "axes.linewidth":         0.7,
    "lines.linewidth":        1.5,
    "lines.markersize":       4.5,
    "lines.markeredgewidth":  0.6,
    "pdf.fonttype":           42,
    "ps.fonttype":            42,
})

# ── Method palette ─────────────────────────────────────────────────────
METHODS      = ["NA", "JEL", "AJEL", "EJEL"]
SURV_METHODS = ["NA", "JEL", "AJEL"]

METHOD_COLORS  = {
    "NA":   "#2171b5",
    "JEL":  "#238b45",
    "AJEL": "#d94801",
    "EJEL": "#6a3d9a",
}
METHOD_STYLES  = {"NA": "-",  "JEL": "--", "AJEL": "-.", "EJEL": ":"}
METHOD_MARKERS = {"NA": "s",  "JEL": "o",  "AJEL": "^",  "EJEL": "D"}
METHOD_MS      = {"NA": 4.5,  "JEL": 4.5,  "AJEL": 5.0,  "EJEL": 4.0}


def _legend_handles(methods):
    return [
        Line2D([0], [0],
               color=METHOD_COLORS[m],
               linestyle=METHOD_STYLES[m],
               marker=METHOD_MARKERS[m],
               markersize=METHOD_MS[m],
               linewidth=1.5,
               label=m)
        for m in methods
    ]


def _save(fig, stem):
    for ext, kw in [(".pdf", {}), (".png", {"dpi": 300})]:
        path = os.path.join(RESULTS_DIR, stem + ext)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, **kw)
        print(f"  Saved: {stem}{ext}")
    plt.close(fig)


def _finish_ax(ax, xticks, ylim=None):
    ax.set_xticks(xticks)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", which="major", length=3, width=0.6)


# ══════════════════════════════════════════════════════════════════════
# PAPER FIGURES 1–3  Simulation coverage probability
# ══════════════════════════════════════════════════════════════════════

def plot_sim_cp(cp_cols, ylabel, suptitle, stem,
                nominal=0.95, methods=None):
    if methods is None:
        methods = list(cp_cols.keys())

    sub = df[df["nominal"] == nominal].copy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.5, 2.8),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.10, right=0.97,
        bottom=0.18,
        top=0.82,
        wspace=0.10,
    )

    for ax, sc in zip(axes, SCENARIOS_NUM):
        dsc = sub[sub["scenario"] == sc].sort_values("n")
        for m in methods:
            col = cp_cols.get(m)
            if col and col in dsc.columns:
                ax.plot(dsc["n"], dsc[col],
                        color=METHOD_COLORS[m],
                        linestyle=METHOD_STYLES[m],
                        marker=METHOD_MARKERS[m],
                        markersize=METHOD_MS[m])
        ax.axhline(nominal * 100, color="black",
                   linestyle="--", linewidth=0.9, zorder=0)
        ax.set_title(SC_LABELS.get(sc, f"Sc.{sc}"))
        ax.set_xlabel("Sample size $n$")
        _finish_ax(ax, NS, ylim=(62, 101))

    axes[0].set_ylabel(ylabel)
    axes[0].legend(
        handles=_legend_handles(methods),
        loc="lower right",
        fontsize=7.5,
    )
    axes[2].annotate(
        f"nominal {int(nominal*100)}%",
        xy=(NS[-1], nominal * 100),
        xytext=(4, 2), textcoords="offset points",
        fontsize=6.5, va="bottom", color="0.35",
    )

    fig.suptitle(suptitle, x=0.535, y=0.97, fontsize=10)
    _save(fig, stem)


print("Paper figures:")

plot_sim_cp(
    cp_cols={"NA": "cp_na_a", "JEL": "cp_jel_a",
             "AJEL": "cp_ajel_a", "EJEL": "cp_ejel_a"},
    ylabel=r"Coverage probability (%) — $\hat{\alpha}$",
    suptitle=r"Figure 1.  Coverage probability for $\alpha$  (nominal = 95%)",
    stem="fig_jel_sim_alpha",
    methods=METHODS,
)

plot_sim_cp(
    cp_cols={"NA": "cp_na_b", "JEL": "cp_jel_b",
             "AJEL": "cp_ajel_b", "EJEL": "cp_ejel_b"},
    ylabel=r"Coverage probability (%) — $\hat{\beta}$",
    suptitle=r"Figure 2.  Coverage probability for $\beta$  (nominal = 95%)",
    stem="fig_jel_sim_beta",
    methods=METHODS,
)

plot_sim_cp(
    cp_cols={"NA": "cp_na_s", "JEL": "cp_jel_s", "AJEL": "cp_ajel_s"},
    ylabel=r"Coverage probability (%) — $S(t_0)$",
    suptitle=r"Figure 3.  Coverage probability for $S(t_0)$  (nominal = 95%)",
    stem="fig_jel_sim_surv",
    methods=SURV_METHODS,
)


# ══════════════════════════════════════════════════════════════════════
# PAPER FIGURE 4  Real data CI line plots
# 3 rows (alpha / beta / S(t0))  x  2 cols (CI midpoint | CI width)
# ══════════════════════════════════════════════════════════════════════

def plot_realdata_ci(outfile_stem, nominal=0.95):
    if not os.path.exists(REALDATA_FILE):
        print(f"  WARNING: {REALDATA_FILE} not found — skipping Figure 4.")
        print("  Run: python QTEG_JEL_Arctic.py --realdata  first.")
        return

    with open(REALDATA_FILE) as fh:
        rd = json.load(fh)

    ROWS = [
        dict(label=r"$\alpha$",   pe_src="fit",      pe_key="alpha",
             ci_src="ci_alpha",   methods=METHODS),
        dict(label=r"$\beta$",    pe_src="fit",      pe_key="beta",
             ci_src="ci_beta",    methods=METHODS),
        dict(label=r"$S(t_0)$",  pe_src="survival", pe_key="psi_hat",
             ci_src="survival",   methods=SURV_METHODS),
    ]

    def _get(entry, row, method):
        try:
            pe = (entry["fit"][row["pe_key"]]
                  if row["pe_src"] == "fit"
                  else entry["survival"][row["pe_key"]])
            ci_dict = entry[row["ci_src"]]
            key = method.lower()
            ci  = ci_dict.get(key, ci_dict.get(method, [np.nan, np.nan]))
            lo, hi = float(ci[0]), float(ci[1])
            return lo, hi, float(pe)
        except Exception:
            return np.nan, np.nan, np.nan

    n_rows = len(ROWS)
    n_ds   = len(DS_KEYS)
    xs     = np.arange(n_ds)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(6.5, 7.2),      # ── slightly taller to give legend room
    )
    fig.subplots_adjust(
        left=0.12, right=0.97,
        bottom=0.13,             # ── increased: space for x-label + legend
        top=0.91,
        hspace=0.55,
        wspace=0.38,
    )

    col_titles = ["CI midpoint and spread", "CI width"]
    for c, ct in enumerate(col_titles):
        axes[0][c].set_title(ct, fontsize=9, pad=8)

    for r, row in enumerate(ROWS):
        methods = row["methods"]
        ax_mid  = axes[r][0]
        ax_wid  = axes[r][1]

        mid_data = {m: [] for m in methods}
        wid_data = {m: [] for m in methods}
        pe_data  = []

        for ds in DS_KEYS:
            entry = rd.get(ds, {})
            pe_vals = []
            for m in methods:
                lo, hi, pe = _get(entry, row, m)
                mid_data[m].append((lo + hi) / 2 if np.isfinite(lo) else np.nan)
                wid_data[m].append(hi - lo       if np.isfinite(lo) else np.nan)
                pe_vals.append(pe)
            pe_data.append(np.nanmean(pe_vals))

        # ── left panel ────────────────────────────────────────────────
        for m in methods:
            mids = np.array(mid_data[m])
            wids = np.array(wid_data[m])
            half = wids / 2
            ax_mid.fill_between(
                xs, mids - half, mids + half,
                alpha=0.08, color=METHOD_COLORS[m],
            )
            ax_mid.plot(xs, mids,
                        color=METHOD_COLORS[m],
                        linestyle=METHOD_STYLES[m],
                        marker=METHOD_MARKERS[m],
                        markersize=METHOD_MS[m],
                        linewidth=1.5)

        ax_mid.plot(xs, pe_data,
                    color="black", linestyle="-",
                    linewidth=0.9, marker="|",
                    markersize=6, markeredgewidth=1.1,
                    label="MLE", zorder=5)

        ax_mid.set_ylabel(row["label"], fontsize=9, labelpad=5)
        ax_mid.set_xticks(xs)
        ax_mid.set_xticklabels(DS_TICKS, fontsize=8)
        ax_mid.tick_params(axis="both", which="major", length=3, width=0.6)
        ax_mid.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))

        # ── right panel ───────────────────────────────────────────────
        for m in methods:
            wids = np.array(wid_data[m])
            ax_wid.plot(xs, wids,
                        color=METHOD_COLORS[m],
                        linestyle=METHOD_STYLES[m],
                        marker=METHOD_MARKERS[m],
                        markersize=METHOD_MS[m],
                        linewidth=1.5)

        ax_wid.set_xticks(xs)
        ax_wid.set_xticklabels(DS_TICKS, fontsize=8)
        ax_wid.tick_params(axis="both", which="major", length=3, width=0.6)
        ax_wid.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))

        # x-axis label on bottom row only
        if r == n_rows - 1:
            ax_mid.set_xlabel("Dataset", fontsize=9, labelpad=8)  # ── increased labelpad
            ax_wid.set_xlabel("Dataset", fontsize=9, labelpad=8)  # ── increased labelpad

    # ── shared legend — placed above the x-axis labels ────────────────
    leg_handles = _legend_handles(METHODS)
    leg_handles += [
        Line2D([0], [0], color="black", linestyle="-",
               marker="|", markersize=6, markeredgewidth=1.1,
               linewidth=0.9, label="MLE")
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        ncol=len(leg_handles),
        fontsize=8,
        framealpha=0.9,
        edgecolor="0.75",
        bbox_to_anchor=(0.535, 0.035),  # ── raised from 0.01 to 0.035
        handlelength=2.2,
    )

    fig.suptitle(
        f"Figure 4.  Real data confidence intervals  (nominal = {int(nominal*100)}%)\n"
        r"Left: CI midpoint $\pm\frac{1}{2}$width (shaded).  Right: CI width.",
        x=0.535, y=0.99,
        fontsize=9,
        linespacing=1.6,
    )

    _save(fig, outfile_stem)


plot_realdata_ci("fig_jel_realdata")


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ══════════════════════════════════════════════════════════════════════

def plot_al(al_cols, ylabel, suptitle, stem,
            nominal=0.95, methods=None):
    if methods is None:
        methods = list(al_cols.keys())

    sub = df[df["nominal"] == nominal].copy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.5, 2.8),
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.11, right=0.97,
        bottom=0.18,
        top=0.82,
        wspace=0.32,
    )

    for ax, sc in zip(axes, SCENARIOS_NUM):
        dsc = sub[sub["scenario"] == sc].sort_values("n")
        for m in methods:
            col = al_cols.get(m)
            if col and col in dsc.columns:
                ax.plot(dsc["n"], dsc[col],
                        color=METHOD_COLORS[m],
                        linestyle=METHOD_STYLES[m],
                        marker=METHOD_MARKERS[m],
                        markersize=METHOD_MS[m])
        ax.set_title(SC_LABELS.get(sc, f"Sc.{sc}"))
        ax.set_xlabel("Sample size $n$")
        _finish_ax(ax, NS)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))

    axes[0].set_ylabel(ylabel)
    axes[0].legend(
        handles=_legend_handles(methods),
        loc="upper right",
        fontsize=7.5,
    )
    fig.suptitle(suptitle, x=0.535, y=0.97, fontsize=10)
    _save(fig, stem)


def plot_all_nominals(cp_cols, param_label, stem, methods=None):
    if methods is None:
        methods = list(cp_cols.keys())

    nominals = sorted(df["nominal"].unique())
    n_nom    = len(nominals)
    n_sc     = len(SCENARIOS_NUM)

    fig, axes = plt.subplots(
        n_nom, n_sc,
        figsize=(6.5, 6.5),
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.11, right=0.97,
        bottom=0.12,
        top=0.90,
        hspace=0.50,
        wspace=0.18,
    )

    for r, nom in enumerate(nominals):
        sub = df[df["nominal"] == nom]
        for c, sc in enumerate(SCENARIOS_NUM):
            ax  = axes[r][c]
            dsc = sub[sub["scenario"] == sc].sort_values("n")
            for m in methods:
                col = cp_cols.get(m)
                if col and col in dsc.columns:
                    ax.plot(dsc["n"], dsc[col],
                            color=METHOD_COLORS[m],
                            linestyle=METHOD_STYLES[m],
                            marker=METHOD_MARKERS[m],
                            markersize=METHOD_MS[m])
            ax.axhline(nom * 100, color="black",
                       linestyle="--", linewidth=0.9, zorder=0)
            if r == 0:
                ax.set_title(SC_LABELS.get(sc, f"Sc.{sc}"))
            if c == 0:
                ax.set_ylabel(f"CP (%) — {nom:.0%}", fontsize=8,
                              labelpad=5)
            _finish_ax(ax, NS, ylim=(62, 101))
            if r == n_nom - 1:
                ax.set_xlabel("Sample size $n$", fontsize=8, labelpad=5)

    fig.legend(
        handles=_legend_handles(methods),
        loc="lower center",
        ncol=len(methods),
        fontsize=8,
        framealpha=0.9,
        edgecolor="0.75",
        bbox_to_anchor=(0.535, 0.01),
        handlelength=2.4,
    )
    fig.suptitle(
        f"Coverage probability — {param_label} — all nominal levels",
        x=0.535, y=0.96, fontsize=10,
    )
    _save(fig, stem)


print("\nSupplementary figures:")

plot_al(
    al_cols={"NA": "al_na_a", "JEL": "al_jel_a",
             "AJEL": "al_ajel_a", "EJEL": "al_ejel_a"},
    ylabel=r"Average length — $\alpha$",
    suptitle=r"Supplementary S1.  Average interval length for $\alpha$  (95%)",
    stem="fig_jel_alpha_al",
    methods=METHODS,
)

plot_al(
    al_cols={"NA": "al_na_b", "JEL": "al_jel_b",
             "AJEL": "al_ajel_b", "EJEL": "al_ejel_b"},
    ylabel=r"Average length — $\beta$",
    suptitle=r"Supplementary S2.  Average interval length for $\beta$  (95%)",
    stem="fig_jel_beta_al",
    methods=METHODS,
)

plot_al(
    al_cols={"NA": "al_na_s", "JEL": "al_jel_s", "AJEL": "al_ajel_s"},
    ylabel=r"Average length — $S(t_0)$",
    suptitle=r"Supplementary S3.  Average interval length for $S(t_0)$  (95%)",
    stem="fig_jel_survival_al",
    methods=SURV_METHODS,
)

plot_all_nominals(
    cp_cols={"NA": "cp_na_a", "JEL": "cp_jel_a",
             "AJEL": "cp_ajel_a", "EJEL": "cp_ejel_a"},
    param_label=r"$\alpha$",
    stem="fig_jel_alpha_cp_all_nominals",
    methods=METHODS,
)

plot_all_nominals(
    cp_cols={"NA": "cp_na_s", "JEL": "cp_jel_s", "AJEL": "cp_ajel_s"},
    param_label=r"$S(t_0)$",
    stem="fig_jel_survival_cp_all_nominals",
    methods=SURV_METHODS,
)

# ── Final summary ──────────────────────────────────────────────────────
print(f"\nAll figures saved to: {RESULTS_DIR}")
print("\nPAPER FIGURES:")
for stem in ["fig_jel_sim_alpha", "fig_jel_sim_beta",
             "fig_jel_sim_surv",  "fig_jel_realdata"]:
    for ext in (".pdf", ".png"):
        p = os.path.join(RESULTS_DIR, stem + ext)
        print(f"  [{'OK' if os.path.exists(p) else 'MISSING'}]  {stem}{ext}")

print("\nSUPPLEMENTARY FIGURES:")
for stem in ["fig_jel_alpha_al", "fig_jel_beta_al", "fig_jel_survival_al",
             "fig_jel_alpha_cp_all_nominals",
             "fig_jel_survival_cp_all_nominals"]:
    for ext in (".pdf", ".png"):
        p = os.path.join(RESULTS_DIR, stem + ext)
        print(f"  [{'OK' if os.path.exists(p) else 'MISSING'}]  {stem}{ext}")
