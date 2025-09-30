#!/usr/bin/env python3
import sys
from os import listdir
from os.path import isfile

def read_atoms():
    atoms = []
    with open("geometry.in") as f:
        for line in f:
            w = line.split()
            if not w: continue
            if w[0] in ("atom","atom_frac"):
                atoms.append(w[-1])
    return atoms

def list_band_files():
    files = [f for f in listdir('.') if isfile(f)]
    band = sorted([f for f in files if f.startswith("band") and f.endswith(".out") and "mlk" not in f])
    mlks = sorted([f for f in files if f.startswith("bandmlk") and f.endswith(".out")])
    return band, mlks

def load_bands(bandfiles):
    energies = {}
    for file_id, bf in enumerate(bandfiles):
        with open(bf) as fin:
            file_lines = [ln.split() for ln in fin if ln.strip()]
        n_states = (len(file_lines[0])-4)//2
        for s in range(1, n_states+1):
            energies.setdefault(s, [[] for _ in range(len(bandfiles))])
        for items in file_lines:
            for s in range(1, n_states+1):
                e = float(items[2*s+3])
                energies[s][file_id].append(e)
    return energies

def find_vbm_cbm_locs(energies, fermi=0.0):
    states_sorted = sorted(energies.keys())
    vbm_state = None
    for s in states_sorted:
        if s+1 in energies and max(sum(energies[s+1], [])) > fermi:
            vbm_state = s
            break
    if vbm_state is None:
        raise RuntimeError("Cannot locate VBM/CBM")

    def argmax_loc(grid_lists):
        best = (-1e9, None, None)
        for fid, arr in enumerate(grid_lists):
            for kid, val in enumerate(arr):
                if val>best[0]:
                    best=(val,fid,kid)
        return best
    def argmin_loc(grid_lists):
        best = (1e9, None, None)
        for fid, arr in enumerate(grid_lists):
            for kid, val in enumerate(arr):
                if val<best[0]:
                    best=(val,fid,kid)
        return best
    vbm_E, vbm_f, vbm_k = argmax_loc(energies[vbm_state])
    cbm_E, cbm_f, cbm_k = argmin_loc(energies[vbm_state+1])
    return (vbm_state, vbm_E, vbm_f, vbm_k), (vbm_state+1, cbm_E, cbm_f, cbm_k)

def mulliken_for_state_k(mlkfile, target_state, target_k, atoms, select_species, substate=None):
    LINE_ATOM = 3
    MLK_START = 5
    n = len(atoms)
    twice = 2*n
    spec_sum = {sp:0.0 for sp in select_species}
    with open(mlkfile) as f:
        current_state = None
        current_k = None
        cur_atoms=[]; cur_vals=[]
        for line in f:
            t=line.strip()
            if not t: continue
            if t.startswith("k point number:"):
                token = t.split()[3]
                token = token.strip(" ;:")
                current_k = int(token)
            elif t.startswith("State") and len(t.split())==2:
                current_state = int(t.split()[1])
                cur_atoms=[]; cur_vals=[]
            elif t[0].isdigit():
                w=t.split()
                cur_atoms.append(w[LINE_ATOM])
                if substate is None:
                    val = float(w[MLK_START])
                else:
                    val = float(w[MLK_START+1+substate])
                cur_vals.append(val)
                if len(cur_atoms)==twice and current_state==target_state and current_k==target_k:
                    for i, sp in enumerate(atoms):
                        if sp in select_species:
                            spec_sum[sp] += cur_vals[2*i] + cur_vals[2*i+1]
                    return spec_sum
    raise RuntimeError(f"Not found state={target_state} k={target_k} in {mlkfile}")

def normalize(dic):
    tot = sum(dic.values())
    return {k:(v/tot if tot>0 else 0.0) for k,v in dic.items()}

if __name__=="__main__":
    if len(sys.argv)<2:
        print("用法: python extract_contrib.py <species...> [--per-atom] [--substate s|p|d|f]")
        sys.exit(1)
    args = sys.argv[1:]
    per_atom = False
    substate = None
    if "--per-atom" in args:
        per_atom=True; args.remove("--per-atom")
    if "--substate" in args:
        ix = args.index("--substate")
        orb = args[ix+1].lower()
        args = args[:ix]+args[ix+2:]
        map_orb = {"s":0,"p":1,"d":2,"f":3}
        substate = map_orb.get(orb, None)
    selected = args

    atoms = read_atoms()
    bandfiles, mlkfiles = list_band_files()
    energies = load_bands(bandfiles)
    (v_state, v_E, v_f, v_k), (c_state, c_E, c_f, c_k) = find_vbm_cbm_locs(energies, fermi=0.0)

    print(f"VBM state={v_state}, E={v_E:.6f} eV @ file#{v_f}, k#{v_k}")
    print(f"CBM state={c_state}, E={c_E:.6f} eV @ file#{c_f}, k#{c_k}\n")

    def get_contrib_for(state, f_id, k_id):
        mlkfile = mlkfiles[f_id]
        raw = mulliken_for_state_k(mlkfile, state, k_id, atoms, selected, substate=substate)
        frac = normalize(raw)
        if per_atom:
            counts = {sp: atoms.count(sp) for sp in selected}
            frac = {sp: (frac[sp]/counts[sp] if counts[sp]>0 else 0.0) for sp in selected}
            frac = normalize(frac)
        return frac

    contrib_vbm   = get_contrib_for(v_state,   v_f, v_k)
    contrib_vbm_1 = get_contrib_for(v_state-1, v_f, v_k)
    contrib_cbm   = get_contrib_for(c_state,   c_f, c_k)
    contrib_cbm_1 = get_contrib_for(c_state+1, c_f, c_k)

    header = "Species,VBM,VBM-1,CBM,CBM+1,VBM_E,VBM-1_E,CBM_E,CBM+1_E"
    print(header)
    with open("contrib_selected.csv","w") as fout:
        fout.write(header+"\n")
        for sp in selected:
            row = [sp,
                   f"{contrib_vbm.get(sp,0.0):.3f}",
                   f"{contrib_vbm_1.get(sp,0.0):.3f}",
                   f"{contrib_cbm.get(sp,0.0):.3f}",
                   f"{contrib_cbm_1.get(sp,0.0):.3f}",
                   f"{v_E:.6f}",
                   f"{energies[v_state-1][v_f][v_k]:.6f}" if (v_state-1) in energies else "NA",
                   f"{c_E:.6f}",
                   f"{energies[c_state+1][c_f][c_k]:.6f}" if (c_state+1) in energies else "NA"]
            line = ",".join(row)
            print(line)
            fout.write(line+"\n")

    print("\n✅ 结果已保存到 contrib_selected.csv")
