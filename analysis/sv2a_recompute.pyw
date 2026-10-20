import MDAnalysis as mda
from MDAnalysis.analysis import rms, distances
import numpy as np, pandas as pd, os

WORK = "/workspace/SV2A/dock_md"
FRAME_PS   = 10.0
DISCARD_NS = 5.0

SYSTEMS = [("levetiracetam", 100.0), ("brivaracetam", 50.0), ("piracetam", 50.0)]

def ligsel(u):
    sel = "resname LIG"
    if u.select_atoms(sel).n_atoms == 0:
        cand = [r.resname for r in u.residues
                if r.resname not in ("HOH","WAT","NA","CL","POPC","SOD","CLA")
                and not u.select_atoms(f"protein and resname {r.resname}").n_atoms]
        cand = sorted(cand, key=lambda rn: abs(u.select_atoms(f"resname {rn}").n_atoms - 26))
        sel = f"resname {cand[0]}"
    return sel

rows = []
for name, total_ns in SYSTEMS:
    u = mda.Universe(os.path.join(WORK, f"{name}_system.pdb"),
                     os.path.join(WORK, f"{name}_traj.dcd"))
    sel = ligsel(u); lig = u.select_atoms(sel); prot = u.select_atoms("protein")
    u.trajectory[0]
    pocket = u.select_atoms("protein and byres (around 4.5 group ligsel)", ligsel=lig)
    sitesel = "name CA and resid " + " ".join(map(str, sorted(set(pocket.residues.resids))))

    Rl = rms.RMSD(u, u, select=sel, ref_frame=0); Rl.run()
    Rs = rms.RMSD(u, u, select=sitesel, ref_frame=0); Rs.run()
    Rp = rms.RMSD(u, u, select="protein and name CA", ref_frame=0); Rp.run()

    has_box = u.dimensions is not None and u.dimensions[:3].min() > 0
    mindist = np.array([distances.distance_array(lig.positions, prot.positions,
                        box=(u.dimensions if has_box else None)).min() for ts in u.trajectory])

    t = np.linspace(0, total_ns, len(Rl.results.rmsd)); m = t >= DISCARD_NS
    rows.append({"ligand": name, "production_ns": total_ns,
        "ligand_rmsd_A": round(float(Rl.results.rmsd[m,2].mean()),2),
        "binding_site_ca_rmsd_A": round(float(Rs.results.rmsd[m,2].mean()),2),
        "whole_protein_ca_rmsd_A": round(float(Rp.results.rmsd[m,2].mean()),2),
        "min_lig_prot_dist_A": round(float(mindist[m].mean()),2),
        "contact_pct": round(100*float(np.mean(mindist[m] <= 4.0)))})
    print(f"{name:14s} ligRMSD {rows[-1]['ligand_rmsd_A']}  "
          f"siteCa {rows[-1]['binding_site_ca_rmsd_A']}  "
          f"minD {rows[-1]['min_lig_prot_dist_A']}  contact {rows[-1]['contact_pct']}%")

df = pd.DataFrame(rows); df.to_csv(os.path.join(WORK,"sv2a_table2_recomputed.csv"), index=False)
print()
print(df.to_string(index=False))
print("\nwritten:", os.path.join(WORK,"sv2a_table2_recomputed.csv"))
