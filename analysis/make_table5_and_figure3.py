"""
make_table5_and_figure3.py
--------------------------
Produces the two remaining items for the revised manuscript:

  table5_stability.csv   receptor Ca RMSD, mean RMSF, radius of gyration,
                         receptor-only SASA and contact retention, reported as
                         mean +/- SD across the three replicates of each system.

  figure3_stability.{png,svg}
                         minimum ligand-receptor heavy-atom distance over time,
                         one panel per receptor, all three replicates of each
                         drug overlaid, with the 4.0 A contact criterion marked.
                         This replaces the ligand-RMSD panel of the original
                         Figure 3, which the revised Methods disowns: RMSD after
                         global superposition projects loop and terminal motion
                         onto the apparent ligand displacement in these
                         truncated constructs.

Run from /workspace with the pod started:

    python make_table5_and_figure3.py --out revised_figures

SASA and RMSF are computed on a strided subset (default every 10th frame) to
keep the run to a few minutes; the contact metrics use every frame.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

D2 = "d2_system"
HT = "5ht2a_system"

# receptor, drug, ligand resname, topology, [trajectories in replicate order]
SYSTEMS = [
    ("D2", "levetiracetam", "UNK", f"{D2}/day12_minimized.pdb",
     [f"{D2}/day12_production.dcd", f"{D2}/day12_production_rep2.dcd",
      f"{D2}/day12_production_rep3.dcd"]),
    ("D2", "brivaracetam", "BRV", f"{D2}/day14_briv_minimized.pdb",
     [f"{D2}/day14_briv_production.dcd", f"{D2}/day14_briv_production_rep2.dcd",
      f"{D2}/day14_briv_production_rep3.dcd"]),
    ("D2", "piracetam", "PIR", f"{D2}/day18_pira_minimized.pdb",
     [f"{D2}/day18_pira_production.dcd", f"{D2}/day18_pira_production_rep2.dcd",
      f"{D2}/day18_pira_production_rep3.dcd"]),
    ("5-HT2A", "levetiracetam", "LEV", f"{HT}/ht2a_lev_minimized.pdb",
     [f"{HT}/ht2a_lev_production.dcd", f"{HT}/ht2a_lev_production_rep2.dcd",
      f"{HT}/ht2a_lev_production_rep3.dcd"]),
    ("5-HT2A", "brivaracetam", "BRV", f"{HT}/ht2a_brv_minimized.pdb",
     [f"{HT}/ht2a_brv_production.dcd", f"{HT}/ht2a_brv_production_rep2.dcd",
      f"{HT}/ht2a_brv_production_rep3.dcd"]),
    ("5-HT2A", "piracetam", "PIR", f"{HT}/ht2a_pir_minimized.pdb",
     [f"{HT}/ht2a_pir_production.dcd", f"{HT}/ht2a_pir_production_rep2.dcd",
      f"{HT}/ht2a_pir_production_rep3.dcd"]),
]

COL = {"levetiracetam": "#2c6fbb", "brivaracetam": "#e8833a", "piracetam": "#3a9a5c"}
CUTOFF_NM = 0.40
PROBE_NM = 0.14
N_SPHERE = 960

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})



AMINO = ("ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
         "PHE PRO SER THR TRP TYR VAL ASH GLH LYN").split()


def receptor_atoms(traj, ligand_resname):
    """Standard amino acids only.

    mdtraj's "protein" selector cannot be trusted on these topologies: the D2
    membrane PDB carries more than 9,999 residues, so the residue-number field
    overflows and the selector picks up lipid and water as well.  Selecting by
    residue name avoids that entirely.
    """
    keep = set(AMINO)
    return np.array([a.index for a in traj.topology.atoms
                     if a.residue.name in keep
                     and a.residue.name != ligand_resname])


def min_distance(traj, lig, prot, chunk=100):
    pairs = np.array([(i, j) for i in lig for j in prot])
    out = np.empty(traj.n_frames)
    for s in range(0, traj.n_frames, chunk):
        out[s:s + chunk] = md.compute_distances(
            traj[s:s + chunk], pairs, periodic=True).min(axis=1)
    return out


def analyse(receptor, drug, resname, top, traj_path, discard_frames, stride):
    traj = md.load(traj_path, top=top)[discard_frames:]
    lig = traj.topology.select(f"resname {resname} and element != H")
    rec_all = receptor_atoms(traj, resname)
    heavy = {a.index for a in traj.topology.atoms if a.element.symbol != "H"}
    prot = np.array([i for i in rec_all if i in heavy])
    if len(lig) == 0:
        raise SystemExit(f"no ligand atoms for '{resname}' in {traj_path}")

    d = min_distance(traj, lig, prot)
    retention = float((d <= CUTOFF_NM).mean() * 100)

    rec = traj.atom_slice(rec_all)
    sub = rec[::stride]
    ca = rec.topology.select("name CA")
    rmsd = md.rmsd(rec, rec, 0, atom_indices=ca) * 10.0
    rmsf = md.rmsf(sub, sub, 0, atom_indices=ca) * 10.0
    rg = md.compute_rg(sub) * 10.0
    sasa = md.shrake_rupley(sub, probe_radius=PROBE_NM,
                            n_sphere_points=N_SPHERE, mode="atom").sum(axis=1) * 100.0

    return {
        "receptor": receptor, "drug": drug,
        "min_dist_A": float(d.mean() * 10), "contact_retention_pct": retention,
        "receptor_CA_RMSD_A": float(rmsd.mean()),
        "mean_RMSF_A": float(rmsf.mean()),
        "Rg_A": float(rg.mean()),
        "SASA_A2": float(sasa.mean()),
    }, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-ps", type=float, default=50.0)
    ap.add_argument("--discard-ns", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=10,
                    help="frame stride for the expensive SASA/RMSF metrics")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    discard = int(args.discard_ns * 1000 / args.frame_ps)

    rows, traces = [], {}
    for receptor, drug, resname, top, trajs in SYSTEMS:
        per_rep = []
        traces[(receptor, drug)] = []
        for k, tp in enumerate(trajs, start=1):
            if not os.path.exists(tp):
                print(f"  missing, skipping: {tp}")
                continue
            m, d = analyse(receptor, drug, resname, top, tp, discard, args.stride)
            m["replicate"] = k
            per_rep.append(m)
            traces[(receptor, drug)].append(d * 10.0)
            print(f"  {receptor:7s} {drug:14s} rep{k}  "
                  f"retention {m['contact_retention_pct']:.1f}%  "
                  f"min-dist {m['min_dist_A']:.2f} A")
        if not per_rep:
            continue
        df = pd.DataFrame(per_rep)
        agg = {"receptor": receptor, "drug": drug, "n_replicates": len(per_rep)}
        for col in ["receptor_CA_RMSD_A", "mean_RMSF_A", "Rg_A", "SASA_A2",
                    "min_dist_A", "contact_retention_pct"]:
            agg[col] = f"{df[col].mean():.2f} \u00b1 {df[col].std(ddof=1):.2f}"
        rows.append(agg)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "table5_stability.csv"), index=False)
    print("\n" + table.to_string(index=False))

    # ---- Figure 3 --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, receptor in zip(axes, ["D2", "5-HT2A"]):
        for drug in ["levetiracetam", "brivaracetam", "piracetam"]:
            for k, d in enumerate(traces.get((receptor, drug), [])):
                t = (np.arange(len(d)) * args.frame_ps / 1000.0) + args.discard_ns
                ax.plot(t, d, lw=0.6, color=COL[drug], alpha=0.85,
                        label=drug if k == 0 else None)
        ax.axhline(4.0, ls="--", c="grey", lw=0.9)
        ax.text(0.98, 4.25, "contact criterion \u2264 4.0 \u00c5", ha="right",
                fontsize=7.5, color="grey", transform=ax.get_yaxis_transform())
        ax.set_xlabel("time (ns)")
        ax.set_title(receptor, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 6)
    axes[0].set_ylabel("min ligand\u2013receptor\nheavy-atom distance (\u00c5)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(args.out, f"figure3_stability.{ext}"))
    plt.close(fig)
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
