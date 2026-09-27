"""
make_supplementary_figures.py
-----------------------------
Regenerates Supplementary Figures S1-S4 from the replicate trajectories,
correcting the three defects in the submitted version:

  * S3 SASA was computed over the full solute (protein + bilayer) for the D2
    systems, giving values of ~86,000-100,000 A^2 against ~15,500 A^2 for the
    other receptors.  Everything here is receptor-only: standard amino acids,
    heavy atoms, with lipid, water and ions excluded, Shrake-Rupley with a
    1.4 A probe and 960 sphere points per atom.

  * The panels showed one trajectory per drug.  Each panel now shows the mean
    across three replicates with the replicate range shaded, so the reader can
    see the spread rather than one run.

  * Captions asserted uniform stability.  The script prints, per system, the
    drift between the first and last 5 ns of each metric, so the captions can
    describe what the traces actually do.

mdtraj's "protein" selector is not used: the D2 membrane topology carries more
than 9,999 residues, the residue-number field overflows, and the selector then
picks up lipid and water.  Residues are selected by name instead.

Usage (from /workspace, pod running):

    python make_supplementary_figures.py --out supplementary_figures

Takes roughly 15-25 minutes; SASA and RMSF use a strided subset.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

D2, HT = "d2_system", "5ht2a_system"

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

AMINO = set(("ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS HID HIE HIP ILE LEU LYS "
             "MET PHE PRO SER THR TRP TYR VAL ASH GLH LYN").split())
COL = {"levetiracetam": "#2c6fbb", "brivaracetam": "#e8833a", "piracetam": "#3a9a5c"}
DRUGS = ["levetiracetam", "brivaracetam", "piracetam"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def receptor_atoms(traj, ligand):
    return np.array([a.index for a in traj.topology.atoms
                     if a.residue.name in AMINO and a.residue.name != ligand])


def polar_contacts(traj, ligand, cutoff_nm=0.35):
    lig = traj.topology.select(f"resname {ligand} and (element O or element N)")
    rec = [a.index for a in traj.topology.atoms
           if a.residue.name in AMINO and a.element.symbol in ("O", "N")]
    if len(lig) == 0 or not rec:
        return np.zeros(traj.n_frames)
    pairs = np.array([(i, j) for i in lig for j in rec])
    out = np.empty(traj.n_frames)
    for s in range(0, traj.n_frames, 100):
        d = md.compute_distances(traj[s:s + 100], pairs, periodic=True)
        out[s:s + 100] = (d <= cutoff_nm).sum(axis=1)
    return out


def metrics(path, top, ligand, discard, stride):
    traj = md.load(path, top=top)[discard:]
    rec = receptor_atoms(traj, ligand)
    prot = traj.atom_slice(rec)
    sub = prot[::stride]
    ca = prot.topology.select("name CA")

    sasa = md.shrake_rupley(sub, probe_radius=0.14, n_sphere_points=960,
                            mode="atom").sum(axis=1) * 100.0
    rg = md.compute_rg(sub) * 10.0
    rmsf = md.rmsf(sub, sub, 0, atom_indices=ca) * 10.0
    hb = polar_contacts(traj, ligand)
    return {"sasa": sasa, "rg": rg, "rmsf": rmsf, "hbond": hb,
            "n_sub": sub.n_frames, "n_full": traj.n_frames}


def band(ax, series, colour, label, dt_ns, offset_ns):
    """Mean line with the replicate range shaded."""
    n = min(len(s) for s in series)
    arr = np.vstack([s[:n] for s in series])
    t = np.arange(n) * dt_ns + offset_ns
    ax.fill_between(t, arr.min(axis=0), arr.max(axis=0), color=colour, alpha=0.22,
                    linewidth=0)
    ax.plot(t, arr.mean(axis=0), color=colour, lw=0.9, label=label)


def drift(series):
    """Change between the first and last tenth of a trace, averaged over replicates."""
    vals = []
    for s in series:
        k = max(2, len(s) // 10)
        vals.append(float(np.mean(s[-k:]) - np.mean(s[:k])))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-ps", type=float, default=50.0)
    ap.add_argument("--discard-ns", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    discard = int(args.discard_ns * 1000 / args.frame_ps)
    dt_sub = args.frame_ps * args.stride / 1000.0
    dt_full = args.frame_ps / 1000.0

    data, notes = {}, []
    for receptor, drug, ligand, top, trajs in SYSTEMS:
        series = []
        for p in trajs:
            if not os.path.exists(p):
                print(f"  missing: {p}")
                continue
            series.append(metrics(p, top, ligand, discard, args.stride))
            print(f"  {receptor:7s} {drug:14s} {os.path.basename(p)}")
        if series:
            data[(receptor, drug)] = series
            notes.append({
                "receptor": receptor, "drug": drug,
                "SASA_A2_mean": round(float(np.mean([s["sasa"].mean() for s in series])), 0),
                "SASA_drift_A2": round(drift([s["sasa"] for s in series]), 0),
                "Rg_A_mean": round(float(np.mean([s["rg"].mean() for s in series])), 2),
                "Rg_drift_A": round(drift([s["rg"] for s in series]), 3),
                "RMSF_A_mean": round(float(np.mean([s["rmsf"].mean() for s in series])), 2),
                "polar_contacts_mean": round(float(np.mean([s["hbond"].mean() for s in series])), 2),
            })

    pd.DataFrame(notes).to_csv(os.path.join(args.out, "supplementary_values.csv"),
                               index=False)
    print("\n" + pd.DataFrame(notes).to_string(index=False))

    receptors = ["D2", "5-HT2A"]

    # ---- S1: per-residue RMSF -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for ax, rec in zip(axes, receptors):
        for drug in DRUGS:
            s = data.get((rec, drug))
            if not s:
                continue
            arr = np.vstack([x["rmsf"][:min(len(y["rmsf"]) for y in s)] for x in s])
            x = np.arange(arr.shape[1])
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0),
                            color=COL[drug], alpha=0.2, linewidth=0)
            ax.plot(x, arr.mean(axis=0), color=COL[drug], lw=0.8, label=drug)
        ax.set_xlabel("residue index"); ax.set_title(rec, loc="left", fontweight="bold")
    axes[0].set_ylabel("C\u03b1 RMSF (\u00c5)"); axes[0].legend(frameon=False, fontsize=7.5)
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "S1_rmsf.png")); plt.close(fig)

    # ---- S2 Rg, S3 SASA, S4 polar contacts ------------------------------
    for key, ylab, fname, dt in [
            ("rg", "radius of gyration (\u00c5)", "S2_radius_of_gyration.png", dt_sub),
            ("sasa", "receptor-only SASA (\u00c5\u00b2)", "S3_sasa.png", dt_sub),
            ("hbond", "polar contacts \u2264 3.5 \u00c5", "S4_polar_contacts.png", dt_full)]:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
        for ax, rec in zip(axes, receptors):
            for drug in DRUGS:
                s = data.get((rec, drug))
                if not s:
                    continue
                band(ax, [x[key] for x in s], COL[drug], drug, dt, args.discard_ns)
            ax.set_xlabel("time (ns)")
            ax.set_title(rec, loc="left", fontweight="bold")
        axes[0].set_ylabel(ylab); axes[0].legend(frameon=False, fontsize=7.5)
        plt.tight_layout(); fig.savefig(os.path.join(args.out, fname)); plt.close(fig)

    print(f"\nwritten to {args.out}/")
    print("Use supplementary_values.csv to write captions that describe the "
          "traces accurately, including any drift.")


if __name__ == "__main__":
    main()
