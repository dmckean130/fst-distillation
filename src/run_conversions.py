"""Pipeline per dataset:
 
  1. load psi/delta tables from the W&B artifact
  2. build the product via bimachine_to_fst (capped at 10**6 states)
  3. trim to accessible AND co-accessible using the raw arcs
  4. build a pyfoma FST from the trimmed arcs, then filter_accessible ->
     determinize -> minimize, gated on size

"""

import argparse
import csv
import time
from collections import defaultdict, deque

import wandb

from src.bimachine_io import load_tables
from src.bimachine_to_fst import bimachine_to_fst
from src.product_fst import product_to_pyfoma

USER = "dmckean130-university-of-colorado-boulder"
PROJECT = "fst-distillation.extraction.v2"

MAX_STATES = 10**6
MAX_PYFOMA = 200_000

# hardcoding runs 
RUNS = [
    ("g2p/geo", "jtdov1jn"),
    ("g2p/fre", "6okg216x"),
    ("g2p/dut", "yj614phc"),
    ("histnorm/deu", "x57e3mo2"),
]

COLUMNS = [
    "dataset", "run_id",
    "psi_entries", "delta_L_keys", "delta_R_keys", "n_start", "n_finals",
    "build_status", "build_seconds", "n_arcs", "states_reachable",
    "states_trimmed", "arcs_trimmed", "trim_seconds",
    "pyfoma_status", "states_built", "states_accessible",
    "determinize", "states_determinized", "determinize_seconds",
    "minimize", "states_minimized", "minimize_seconds",
]

def trim(arcs, finals, start):
    """Keep states that are reachable from start and can reach a final."""
    reachable = set(start) | {d for (_, _, _, d) in arcs}

    predecessors = defaultdict(list)
    for src, _, _, dest in arcs: 
        predecessors[dest].append(src)

    co = set(finals)
    queue = deque(finals)
    while queue:
        node = queue.popleft()
        for pre in predecessors[node]:
            if pre not in co: 
                co.add(pre)
                queue.append(pre)

    keep = reachable & co
    kept_arcs = [x for x in arcs if x[0] in keep and x[3] in keep]
    return keep, kept_arcs

def run_one(dataset, run_id, max_pyfoma):
    print(f"\n{'=' * 60}\n{dataset}  (run {run_id})")
    row = {"dataset": dataset, "run_id": run_id}
 
    artifact = wandb.Api().artifact(f"{USER}/{PROJECT}/bimachine-{run_id}:latest")
    tables = load_tables(f"{artifact.download()}/bimachine.json")
 
    row["psi_entries"] = len(tables.psi)
    row["delta_L_keys"] = len(tables.delta_L)
    row["delta_R_keys"] = len(tables.delta_R)
    row["n_start"] = len(tables.F_R)
    print(f"  psi={row['psi_entries']}  delta_L={row['delta_L_keys']}  "
          f"delta_R={row['delta_R_keys']}  |F_R|={row['n_start']}")
 
    t0 = time.time()
    arcs, finals, start, status = bimachine_to_fst(tables, max_states=MAX_STATES)
    row["build_seconds"] = round(time.time() - t0, 2)
    row["build_status"] = status
    row["n_arcs"] = len(arcs)
    row["n_finals"] = len(finals)
    row["states_reachable"] = len(set(start) | {d for (_, _, _, d) in arcs})
    print(f"  build: {status} in {row['build_seconds']}s | "
          f"arcs={row['n_arcs']} reachable={row['states_reachable']} "
          f"finals={row['n_finals']}")
 
    if status == "capped":
        print(f"  CAPPED at {MAX_STATES} states -- product too large to enumerate.")
        row["pyfoma_status"] = "not_attempted: build capped"
        return row
 
    t0 = time.time()
    keep, kept_arcs = trim(arcs, finals, start)
    row["trim_seconds"] = round(time.time() - t0, 2)
    row["states_trimmed"] = len(keep)
    row["arcs_trimmed"] = len(kept_arcs)
    print(f"  trimmed: {len(keep)} states, {len(kept_arcs)} arcs "
          f"({row['trim_seconds']}s)")
 
    if len(keep) > max_pyfoma:
        print(f"  SKIPPING pyfoma: {len(keep)} > {max_pyfoma}")
        row["pyfoma_status"] = f"not_attempted: {len(keep)} > {max_pyfoma}"
        return row
 
    kept_finals = [f for f in finals if f in keep]
    kept_start = {s for s in start if s in keep}
 
    try:
        fst = product_to_pyfoma(kept_arcs, kept_finals, kept_start)
        row["states_built"] = len(fst.states)
        fst = fst.filter_accessible()
        row["states_accessible"] = len(fst.states)
        row["pyfoma_status"] = "built"
        print(f"  pyfoma: built={row['states_built']} "
              f"accessible={row['states_accessible']}")
    except Exception as e:
        row["pyfoma_status"] = f"failed: {type(e).__name__}: {e}"
        print(f"  pyfoma FAILED: {type(e).__name__}: {e}")
        return row
 
    t0 = time.time()
    try:
        det = fst.determinize()
        row["determinize"] = "completed"
        row["states_determinized"] = len(det.states)
        fst = det
        print(f"  determinize: {row['states_determinized']} states "
              f"({time.time() - t0:.1f}s)")
    except Exception as e:
        row["determinize"] = f"failed: {type(e).__name__}: {e}"
        print(f"  determinize FAILED: {type(e).__name__}: {e}")
    row["determinize_seconds"] = round(time.time() - t0, 2)
 
    t0 = time.time()
    try:
        mini = fst.minimize()
        row["minimize"] = "completed"
        row["states_minimized"] = len(mini.states)
        print(f"  minimize: {row['states_minimized']} states "
              f"({time.time() - t0:.1f}s)")
    except Exception as e:
        row["minimize"] = f"failed: {type(e).__name__}: {e}"
        print(f"  minimize FAILED: {type(e).__name__}: {e}")
    row["minimize_seconds"] = round(time.time() - t0, 2)
 
    return row
 
 
def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="notes/conversion_results.csv")
    parser.add_argument("--only", help="run a single dataset, e.g. g2p/geo")
    parser.add_argument("--max-pyfoma", type=int, default=MAX_PYFOMA)
    args = parser.parse_args()
 
    todo = [(d, r) for d, r in RUNS if args.only is None or d == args.only]
    rows = []
    for dataset, run_id in todo:
        try:
            rows.append(run_one(dataset, run_id, args.max_pyfoma))
        except Exception as e:
            print(f"  DATASET FAILED: {type(e).__name__}: {e}")
            rows.append({"dataset": dataset, "run_id": run_id,
                         "build_status": f"error: {type(e).__name__}: {e}"})
        write_csv(rows, args.out)   # rewrite after every dataset
        print(f"  [saved {len(rows)} rows to {args.out}]")
 
    print(f"\n{'=' * 60}\nSUMMARY")
    for r in rows:
        print(f"  {r['dataset']:16s} build={r.get('build_status')} "
              f"trimmed={r.get('states_trimmed')} "
              f"det={r.get('determinize')} min={r.get('states_minimized')}")
 
 
if __name__ == "__main__":
    main()