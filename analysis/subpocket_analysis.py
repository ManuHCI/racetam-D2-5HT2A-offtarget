"""
subpocket_analysis.py
---------------------
Tests whether independent replicates visited structurally distinct regions of
the receptor, rather than disagreeing at random.

For each replicate it computes per-residue contact frequencies over the
equilibrated segment, maps processed residue numbers to canonical numbering via
the mapping CSV, assigns each canonical residue to a transmembrane helix or
loop from user-supplied boundaries, and reports where each trajectory spent its
time.

This is exploratory analysis of an unexpected result. If the replicates cluster
into recognisable sub-pockets, that is a describable finding; if the contacts
are scattered with no structure, they are simply noisy and this analysis says
so. Both outcomes are useful.

Usage
    python subpocket_analysis.py \
        --traj day12_production.dcd day12_production_rep2.dcd day12_production_rep3.dcd \
        --labels rep1 rep2 rep3 \
        --top day12_minimized.pdb --mapping d2_mapping.csv \
        --ligand-resname UNK --frame-ps 50 --discard-ns 5 \
        --receptor D2 --out d2_subpockets

Requirements: mdtraj, numpy, pandas, matplotlib
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

AA3 = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln',
       'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys',
       'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp',
       'Y': 'Tyr', 'V': 'Val'}

# Canonical-numbering boundaries of the seven transmembrane helices.
# Approximate, taken from the UniProt/GPCRdb topology of each receptor; residues
# outside every range are reported as loop/terminus.
TM_BOUNDS = {
    "D2": {                       # human DRD2, UniProt P14416
        "TM1": (33, 55), "TM2": (66, 89), "TM3": (105, 127), "TM4": (148, 169),
        "TM5": (186, 210), "TM6": (374, 397), "TM7": (409, 431),
    },
    "5HT2A": {                    # human HTR2A, UniProt P28223
        "TM1": (75, 100), "TM2": (111, 133), "TM3": (148, 170), "TM4": (191, 212),
        "TM5": (229, 253), "TM6": (324, 348), "TM7": (360, 383),
    },
}


def region_of(resnum, bounds):
    for name, (lo, hi) in bounds.items():
        if lo <= resnum <= hi:
            return name
    return "loop/terminus"


def contact_frequencies(traj, lig, prot, start, cutoff_nm=0.40, near_nm=0.6):
    sub = traj[start:]
    seen = set()
    stride = max(1, sub.n_frames // 200)
    for f in range(0, sub.n_frames, stride):
        for a in md.compute_neighbors(sub[f], near_nm, lig,
                                      haystack_indices=prot)[0]:
            seen.add(sub.topology.atom(int(a)).residue.index)
    rows = []
    for ri in sorted(seen):
        res = sub.topology.residue(ri)
        atoms = [a.index for a in res.atoms if a.element.symbol != "H"]
        pairs = np.array([(i, j) for i in lig for j in atoms])
        d = md.compute_distances(sub, pairs, periodic=True).min(axis=1)
        rows.append({"processed_resnum": res.resSeq,
                     "processed_name": res.name,
                     "pct": float((d <= cutoff_nm).mean() * 100)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--ligand-resname", required=True)
    ap.add_argument("--frame-ps", type=float, default=50.0)
    ap.add_argument("--discard-ns", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="ignore residues contacted in fewer than this %% of frames")
    ap.add_argument("--receptor", choices=sorted(TM_BOUNDS), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.traj) != len(args.labels):
        raise SystemExit("--traj and --labels must have the same length")
    os.makedirs(args.out, exist_ok=True)
    bounds = TM_BOUNDS[args.receptor]

    mapping = pd.read_csv(args.mapping).dropna(subset=["canonical_resnum"])
    mapping["canonical_resnum"] = mapping["canonical_resnum"].astype(int)
    start = int(args.discard_ns * 1000 / args.frame_ps)

    per_rep = {}
    for path, label in zip(args.traj, args.labels):
        traj = md.load(path, top=args.top)
        lig = traj.topology.select(
            f"resname {args.ligand_resname} and element != H")
        prot = traj.topology.select(
            f"protein and element != H and not resname {args.ligand_resname}")
        df = contact_frequencies(traj, lig, prot, start)
        df = df.merge(mapping, on="processed_resnum", how="left")
        df["canonical"] = (df.canonical_aa.map(AA3).fillna(df.canonical_aa)
                           + df.canonical_resnum.astype("Int64").astype(str))
        df["region"] = df.canonical_resnum.apply(
            lambda n: region_of(n, bounds) if pd.notna(n) else "unmapped")
        df["replicate"] = label
        per_rep[label] = df
        print(f"{label}: {len(df)} residues scored, "
              f"{(df.pct >= args.threshold).sum()} above {args.threshold:g}%")

    allc = pd.concat(per_rep.values(), ignore_index=True)
    allc.to_csv(os.path.join(args.out, "contacts_all_replicates.csv"), index=False)

    # ---- where did each replicate spend its time? ------------------------
    sig = allc[allc.pct >= args.threshold]
    summary = (sig.pivot_table(index="region", columns="replicate",
                               values="pct", aggfunc="sum")
               .reindex(list(bounds) + ["loop/terminus", "unmapped"])
               .fillna(0.0).round(0))
    print("\nTotal contact weight by region (sum of per-residue %):")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.out, "region_summary.csv"))

    # ---- which residues, side by side ------------------------------------
    wide = (sig.pivot_table(index=["region", "canonical"], columns="replicate",
                            values="pct", aggfunc="max")
            .fillna(0.0).round(1))
    wide["n_replicates"] = (wide > args.threshold).sum(axis=1)
    wide = wide.sort_values(["region", args.labels[0]], ascending=[True, False])
    print("\nPer-residue contact frequency (%) by replicate:")
    print(wide.to_string())
    wide.to_csv(os.path.join(args.out, "residues_by_region.csv"))

    shared = wide[wide.n_replicates == len(args.labels)]
    print(f"\nResidues contacted above {args.threshold:g}% in ALL replicates: "
          f"{len(shared)}")
    if len(shared):
        print(shared.drop(columns="n_replicates").to_string())

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    summary.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("summed per-residue contact (%)")
    ax.set_xlabel("")
    ax.set_title(f"{args.receptor}: where each replicate made its contacts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "region_summary.png"), dpi=300)

    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
