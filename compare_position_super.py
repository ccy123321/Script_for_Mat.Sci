#!/usr/bin/env python3
"""
Compare per-element atomic displacements between two FHI-aims geometry.in files
using ASE, with optional element and z filtering.

Usage:
  python compare_displacements_ase.py geometry_initial.in geometry_final.in \
    [--out result.out] [--elements Pb,I,Br,V] \
    [--zfrac 0.0:0.2,0.8:1.0] [--zcart 14.0:16.0]
"""

import argparse
import numpy as np
from ase.io import read
from ase.geometry import find_mic

def parse_windows(spec):
    """Parse a window spec like '0.0:0.2,0.8:1.0' -> [(0.0,0.2),(0.8,1.0)]"""
    if not spec:
        return None
    out = []
    for part in spec.split(','):
        if not part:
            continue
        a, b = part.split(':')
        out.append((float(a), float(b)))
    return out

def in_any_window(val, windows):
    """Return True if val is inside any (a,b) window."""
    for a, b in windows:
        if a <= val <= b:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Compute per-element atomic displacements using ASE")
    parser.add_argument("initial", help="Initial geometry.in")
    parser.add_argument("final", help="Final geometry.in")
    parser.add_argument("--out", default="displacements_ase.out", help="Output text file")
    parser.add_argument("--elements", default=None,
                        help="Comma-separated list of elements to include (default: all elements)")
    parser.add_argument("--zfrac", default=None,
                        help="Fractional z filter windows, e.g., 0.0:0.2,0.8:1.0")
    parser.add_argument("--zcart", default=None,
                        help="Cartesian z (Å) filter windows, e.g., -1.0:2.0,14.0:16.0")
    args = parser.parse_args()

    # Read input structures
    init = read(args.initial, format="aims")
    final = read(args.final, format="aims")

    if len(init) != len(final):
        raise ValueError("Two geometries have different number of atoms!")

    # Ensure element order is consistent
    for i, (a, b) in enumerate(zip(init, final)):
        if a.symbol != b.symbol:
            raise ValueError(f"Atom {i} element mismatch: {a.symbol} vs {b.symbol}")

    # Compute displacement vectors with MIC
    cell = init.get_cell()
    pbc = [True, True, True]
    dr, _ = find_mic(final.get_positions() - init.get_positions(), cell, pbc)
    disp = np.linalg.norm(dr, axis=1)

    # Fractional coordinates (for zfrac filter)
    frac_coords = init.get_scaled_positions()

    # Parse filters
    element_filter = None
    if args.elements:
        element_filter = set([e.strip() for e in args.elements.split(",") if e.strip()])
    wzf = parse_windows(args.zfrac)
    wzc = parse_windows(args.zcart)

    # Collect statistics per element
    element_data = {}
    rows = []
    for i, atom in enumerate(init):
        sym = atom.symbol
        if element_filter and sym not in element_filter:
            continue

        fz = frac_coords[i][2]
        zc = atom.position[2]

        # Apply z filters
        if wzf and not in_any_window(fz, wzf):
            continue
        if wzc and not in_any_window(zc, wzc):
            continue

        rows.append((i+1, sym, dr[i][0], dr[i][1], dr[i][2], disp[i], fz, zc))
        if sym not in element_data:
            element_data[sym] = []
        element_data[sym].append(disp[i])

    # Prepare output text
    output_lines = []
    output_lines.append("Per-atom displacements (Å):")
    output_lines.append(f"{'Idx':>4s} {'El':>2s} {'dx':>10s} {'dy':>10s} {'dz':>10s} {'|dr|':>10s} {'fz':>8s} {'zc':>10s}")
    for idx, sym, dx, dy, dz, d, fz, zc in rows:
        output_lines.append(f"{idx:4d} {sym:>2} {dx:10.6f} {dy:10.6f} {dz:10.6f} {d:10.6f} {fz:8.4f} {zc:10.4f}")

    output_lines.append("\nSummary by element:")
    for sym, values in element_data.items():
        arr = np.array(values)
        output_lines.append(f"{sym:>2}  N={len(arr):4d}  avg={arr.mean():.6f}  max={arr.max():.6f}")

    # Write to file
    with open(args.out, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    # Print to screen
    for line in output_lines:
        print(line)

    print(f"\nResults written to {args.out}")

if __name__ == "__main__":
    main()
