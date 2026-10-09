"""RCD + conversion on the set B (bpe) machines.

Reads run IDs from notes/harvest_runs_bpe.csv, so nothing is hardcoded.
Run from the repo root:  python bpe_rcd_conversion.py
Writes notes/rcd_results_bpe.csv and notes/conversion_results_bpe.csv.
The original right-merge CSVs are not touched.
"""
import csv

from src.right_context_metric import compute_rcd, load_psi_for_run
from src.run_conversions import MAX_PYFOMA, run_one, write_csv

TASK = {"deu": "histnorm", "swe": "histnorm", "geo": "g2p"}

arms = list(csv.DictReader(open("notes/harvest_runs_bpe.csv")))
runs = [(f"{TASK[a['short']]}/{a['short']}", a["best_run_id"])
        for a in arms if a["short"] in TASK and a["best_run_id"]]
print("bpe runs:", runs)

# RCD
rcd_rows = []
for dataset, run_id in runs:
    stats = compute_rcd(load_psi_for_run(run_id))
    rcd_rows.append({"dataset": dataset, "run_id": run_id, **stats})
    print(f"{dataset}: rcd_supported={stats['rcd_supported']:.4f}  "
          f"singly_observed={stats['frac_singly_observed']:.1%}")
with open("notes/rcd_results_bpe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rcd_rows[0]))
    w.writeheader()
    w.writerows(rcd_rows)

# Conversion
conv_rows = []
for dataset, run_id in runs:
    try:
        conv_rows.append(run_one(dataset, run_id, MAX_PYFOMA))
    except Exception as e:
        conv_rows.append({"dataset": dataset, "run_id": run_id,
                          "build_status": f"error: {type(e).__name__}: {e}"})
    write_csv(conv_rows, "notes/conversion_results_bpe.csv")
    print(f"{dataset}: build={conv_rows[-1].get('build_status')} "
          f"trimmed={conv_rows[-1].get('states_trimmed')}")