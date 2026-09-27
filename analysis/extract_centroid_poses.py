"""
extract_centroid_poses.py
-------------------------
Extracts one representative pose per replicate, so Figure 5 can show that the
ligand settles differently in independent trajectories of the same system.

For each replicate it clusters the ligand heavy atoms (after superposing the
binding-site residues, so the clustering describes the pose within the pocket
rather than receptor drift), takes the centroid of the most populated cluster,
and writes that frame as a PDB containing the receptor and the ligand.

It also prints, per replicate, the cluster population and the pairwise ligand
RMSD between the three centroids -- the number that says whether the poses are
genuinely different or merely rendered from different frames.

Usage (from /workspace, pod running):

    python extract_centroid_poses.py \
        --traj d2_system/day12_production.dcd \
               d2_system/day12_production_rep2.dcd \
               d2_system/day12_production_rep3.dcd \
        --top  d2_system/day12_minimized.pdb \
        --ligand-resname UNK --frame-ps 50 --discard-ns 5 \
        --label d2_lev --out figure5_poses

Then open the three PDBs in ChimeraX, orient them identically, and render.

Requires: mdtraj, numpy, scipy
"""

import argparse
import os

import mdtraj as md
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

AMINO = ("ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
         "PHE PRO SER THR TRP TYR VAL ASH GLH LYN").split()


def receptor_atoms(traj, ligand_resname):
    """Standard amino acids only -- mdtraj's 'protein' selector is unreliable on
    topologies with more than 9,999 residues, where the residue number overflows."""
    keep = set(AMINO)
    return np.array([a.index for a in traj.topology.atoms
                     if a.residue.name in keep and a.residue.name != ligand_resname])


def pocket_atoms(traj, lig, rec, cutoff_nm=0.8):
    """Heavy atoms of residues near the ligand in the first analysed frame."""
    heavy = {a.index for a in traj.topology.atoms if a.element.symbol != "H"}
    rec_heavy = [i for i in rec if i in heavy]
    pairs = np.array([(i, j) for i in lig for j in rec_heavy])
    d = md.compute_distances(traj[0], pairs, periodic=True)[0]
    near = {traj.topology.atom(int(pairs[k, 1])).residue.index
            for k in np.where(d < cutoff_nm)[0]}
    return np.array([a.index for a in traj.topology.atoms
                     if a.residue.index in near and a.element.symbol != "H"])


def centroid_frame(traj, lig, pocket, cutoff_A, max_frames):
    """Most populated cluster of ligand poses; returns (frame index, population %)."""
    stride = max(1, traj.n_frames // max_frames)
    sub = traj[::stride]
    sub.superpose(sub, 0, atom_indices=pocket)

    n = sub.n_frames
    xyz = sub.xyz[:, lig, :]
    rms = np.zeros((n, n))
    for i in range(n):
        disp = xyz - xyz[i]
        rms[i] = np.sqrt((disp ** 2).sum(axis=2).mean(axis=1)) * 10.0

    Z = linkage(squareform(rms, checks=False), method="average")
    labels = fcluster(Z, t=cutoff_A, criterion="distance")
    counts = np.bincount(labels)
    biggest = counts.argmax()
    members = np.where(labels == biggest)[0]

    # centroid = member with the smallest summed RMSD to the other members
    sums = rms[np.ix_(members, members)].sum(axis=1)
    centre = members[sums.argmin()]
    return int(centre * stride), 100.0 * len(members) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--ligand-resname", required=True)
    ap.add_argument("--frame-ps", type=float, default=50.0)
    ap.add_argument("--discard-ns", type=float, default=5.0)
    ap.add_argument("--cluster-cutoff", type=float, default=2.0,
                    help="ligand RMSD cutoff in Angstrom")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="frames used for clustering (memory scales as the square)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    discard = int(args.discard_ns * 1000 / args.frame_ps)

    centroid_coords, report = [], []
    for k, path in enumerate(args.traj, start=1):
        traj = md.load(path, top=args.top)[discard:]
        lig = traj.topology.select(
            f"resname {args.ligand_resname} and element != H")
        if len(lig) == 0:
            raise SystemExit(f"no ligand atoms for '{args.ligand_resname}'")
        rec = receptor_atoms(traj, args.ligand_resname)
        pocket = pocket_atoms(traj, lig, rec)

        idx, pop = centroid_frame(traj, lig, pocket,
                                  args.cluster_cutoff, args.max_frames)
        frame = traj[idx]
        keep = np.concatenate([rec, traj.topology.select(
            f"resname {args.ligand_resname}")])
        out_pdb = os.path.join(args.out, f"{args.label}_rep{k}_centroid.pdb")
        frame.atom_slice(np.sort(keep)).save_pdb(out_pdb)

        # keep pocket-superposed ligand coordinates for the comparison below
        t = traj[:]
        t.superpose(traj[idx], atom_indices=pocket)
        centroid_coords.append(t.xyz[idx, lig, :])

        report.append((k, idx, pop, out_pdb))
        print(f"  rep{k}: centroid at frame {idx} "
              f"({idx * args.frame_ps / 1000.0 + args.discard_ns:.1f} ns), "
              f"cluster population {pop:.0f}%  ->  {os.path.basename(out_pdb)}")

    if len(centroid_coords) > 1:
        print("\nPairwise ligand RMSD between centroids (\u00c5):")
        for i in range(len(centroid_coords)):
            for j in range(i + 1, len(centroid_coords)):
                d = centroid_coords[i] - centroid_coords[j]
                r = np.sqrt((d ** 2).sum(axis=1).mean())* 10.0
                print(f"  rep{i+1} vs rep{j+1}: {r:.2f}")
        print("\nValues above roughly 2 \u00c5 indicate the replicates settled into "
              "distinguishable poses; below that they found the same one.")

    print(f"\nwritten to {args.out}/  -- open the PDBs in ChimeraX, superpose on "
          "the receptor, orient identically, and render the three panels.")


if __name__ == "__main__":
    main()
