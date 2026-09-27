"""
make_revised_figures.py
-----------------------
Builds the figures and tables the revised manuscript needs, from the CSVs the
sub-pocket analysis already produced.

Outputs, into --out:
  figure4_contacts.{png,svg}     per-residue contact frequency, mean +/- SD over
                                 replicates, D2 (top) and 5-HT2A (bottom)
  figure6_anchor.{png,svg}       anchor-aspartate contact shown as the three
                                 individual replicate values per drug -- the
                                 figure that makes the irreproducibility visible
  table6_anchor.csv              the same numbers as a table
  table3_d2_contacts.csv         per-residue mean +/- SD at D2
  table4_ht2a_contacts.csv       per-residue mean +/- SD at 5-HT2A

Expects the six analysis output folders produced by subpocket_analysis.py, each
containing contacts_all_replicates.csv.

Usage (from /workspace):
    python make_revised_figures.py \
        --d2-dirs  d2_system/d2_subpockets d2_system/briv_subpockets3 d2_system/pira_subpockets3 \
        --ht-dirs  5ht2a_system/ht2a_lev_subpockets 5ht2a_system/ht2a_brv_subpockets 5ht2a_system/ht2a_pir_subpockets \
        --out revised_figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DRUGS = ["levetiracetam", "brivaracetam", "piracetam"]
COL = {"levetiracetam": "#2c6fbb", "brivaracetam": "#e8833a", "piracetam": "#3a9a5c"}
ANCHOR = {"D2": "Asp114", "5-HT2A": "Asp155"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def load(folder, drug):
    """contacts_all_replicates.csv -> tidy frame: canonical, replicate, pct."""
    path = os.path.join(folder, "contacts_all_replicates.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["canonical"])
    df["drug"] = drug
    return df[["drug", "replicate", "canonical", "canonical_resnum", "pct", "region"]]


def summarise(df):
    """mean and SD across replicates for every residue."""
    g = df.groupby(["drug", "canonical", "canonical_resnum", "region"])["pct"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["drug", "canonical", "resnum", "region", "mean", "sd", "n"]
    # residues absent from a replicate contribute 0, not a missing value
    return out


def fill_zeros(df, n_reps=3):
    """A residue not listed for a replicate was contacted in 0% of its frames."""
    rows = []
    for drug, sub in df.groupby("drug"):
        reps = sorted(sub.replicate.unique())
        residues = sub[["canonical", "canonical_resnum", "region"]].drop_duplicates()
        for _, r in residues.iterrows():
            for rep in reps:
                hit = sub[(sub.canonical == r.canonical) & (sub.replicate == rep)]
                rows.append({
                    "drug": drug, "replicate": rep, "canonical": r.canonical,
                    "canonical_resnum": r.canonical_resnum, "region": r.region,
                    "pct": float(hit.pct.iloc[0]) if len(hit) else 0.0,
                })
    return pd.DataFrame(rows)


def contact_panel(ax, summary, receptor, top_n=16):
    """Grouped bars, mean +/- SD across replicates, for the most-contacted residues."""
    order = (summary.groupby("canonical")["mean"].max()
             .sort_values(ascending=False).head(top_n).index.tolist())
    order = sorted(order, key=lambda c: summary.loc[summary.canonical == c,
                                                    "resnum"].iloc[0])
    x = np.arange(len(order))
    width = 0.26
    for k, drug in enumerate(DRUGS):
        sub = summary[summary.drug == drug].set_index("canonical")
        m = [sub["mean"].get(c, 0.0) for c in order]
        s = [sub["sd"].get(c, 0.0) if not np.isnan(sub["sd"].get(c, np.nan)) else 0.0
             for c in order]
        ax.bar(x + (k - 1) * width, m, width, yerr=s, capsize=2,
               label=drug, color=COL[drug], edgecolor="black", linewidth=0.4,
               error_kw=dict(elinewidth=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("frames with min heavy-atom\ndistance \u2264 4.0 \u00c5 (%)")
    ax.set_ylim(0, 115)
    ax.set_title(receptor, loc="left", fontsize=10, fontweight="bold")
    anchor = ANCHOR[receptor]
    for i, c in enumerate(order):
        if c == anchor:
            ax.axvspan(i - 0.45, i + 0.45, color="#cccccc", alpha=0.35, zorder=0)


def anchor_figure(d2, ht, outdir):
    """Three replicate points per drug per receptor, so the spread is visible."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (df, receptor) in zip(axes, [(d2, "D2"), (ht, "5-HT2A")]):
        anchor = ANCHOR[receptor]
        for k, drug in enumerate(DRUGS):
            vals = df[(df.drug == drug) & (df.canonical == anchor)].pct.values
            if len(vals) == 0:
                vals = np.zeros(3)
            jitter = np.linspace(-0.11, 0.11, len(vals))
            ax.scatter(np.full(len(vals), k) + jitter, vals, s=42,
                       color=COL[drug], edgecolor="black", linewidth=0.5, zorder=3)
            ax.hlines(vals.mean(), k - 0.24, k + 0.24, color="black",
                      linewidth=1.4, zorder=4)
        ax.set_xticks(range(len(DRUGS)))
        ax.set_xticklabels([d[:4].capitalize() for d in DRUGS], fontsize=9)
        ax.set_title(f"{receptor} \u2014 {anchor} (3.32)", fontsize=9.5)
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("anchor contact (% of frames)")
    fig.suptitle("Anchor-aspartate contact, individual replicates", fontsize=10.5)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"figure6_anchor.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dirs", nargs=3, required=True,
                    help="lev, briv, pira analysis folders for D2, in that order")
    ap.add_argument("--ht-dirs", nargs=3, required=True,
                    help="lev, briv, pira analysis folders for 5-HT2A, in that order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d2 = fill_zeros(pd.concat([load(f, d) for f, d in zip(args.d2_dirs, DRUGS)]))
    ht = fill_zeros(pd.concat([load(f, d) for f, d in zip(args.ht_dirs, DRUGS)]))

    s_d2, s_ht = summarise(d2), summarise(ht)
    s_d2.round(1).to_csv(os.path.join(args.out, "table3_d2_contacts.csv"), index=False)
    s_ht.round(1).to_csv(os.path.join(args.out, "table4_ht2a_contacts.csv"), index=False)

    # ---- Figure 4 -------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4))
    contact_panel(axes[0], s_d2, "D2")
    contact_panel(axes[1], s_ht, "5-HT2A")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(args.out, f"figure4_contacts.{ext}"))
    plt.close(fig)

    # ---- Figure 6 and Table 6 -------------------------------------------
    anchor_figure(d2, ht, args.out)

    rows = []
    for df, receptor in [(d2, "D2"), (ht, "5-HT2A")]:
        for drug in DRUGS:
            v = df[(df.drug == drug) &
                   (df.canonical == ANCHOR[receptor])].sort_values("replicate")
            vals = list(v.pct.values) or [0.0, 0.0, 0.0]
            while len(vals) < 3:
                vals.append(0.0)
            rows.append({
                "receptor": receptor, "anchor": ANCHOR[receptor], "drug": drug,
                "replicate_1": round(vals[0], 1), "replicate_2": round(vals[1], 1),
                "replicate_3": round(vals[2], 1),
                "mean": round(float(np.mean(vals)), 1),
                "sd": round(float(np.std(vals, ddof=1)), 1),
            })
    t6 = pd.DataFrame(rows)
    t6.to_csv(os.path.join(args.out, "table6_anchor.csv"), index=False)
    print(t6.to_string(index=False))
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
