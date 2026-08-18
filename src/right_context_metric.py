import argparse
import csv
from collections import defaultdict
from statistics import mean, median

import wandb

from src.bimachine_io import load_tables

# setting wandb project and login now
USER = "dmckean130-university-of-colorado-boulder"
PROJECT = "fst-distillation.extraction.v2"

# bimachine runs (hardcoded)
DEFAULT_RUNS = {
    "g2p/geo": "jtdov1jn",
    "g2p/fre": "6okg216x",
    "g2p/dut": "yj614phc",
    "histnorm/deu": "x57e3mo2",
}

def compute_rcd(psi):
    """Compute RCD from a bimachine output table (psi). Returns a dict of statistics.
    """
    outputs_by_pa = defaultdict(set)
    rstates_by_pa = defaultdict(set)

    for (p, r, a), out in psi.items():
        outputs_by_pa[(p, a)].add(out)
        rstates_by_pa[(p, a)].add(r)

    all_pairs = list(outputs_by_pa)
    if not all_pairs:
        raise ValueError("empty output table")

    def is_dependent(pa):
        return len(outputs_by_pa[pa]) > 1

    rcd_raw = sum(is_dependent(pa) for pa in all_pairs) / len(all_pairs)

    supported = [pa for pa in all_pairs if len(rstates_by_pa[pa]) >= 2]
    rcd_supported = (sum(is_dependent(pa) for pa in supported) / len(supported)
                     if supported
                     else float("nan")
    )

    weights = {pa:len(rstates_by_pa[pa]) - 1 for pa in all_pairs}
    total_weight = sum(weights.values())
    rcd_weighted = (sum(weights[pa] for pa in all_pairs if is_dependent(pa)) / total_weight
                    if total_weight
                    else float("nan")
    )

    supports = [len(rstates_by_pa[pa]) for pa in all_pairs]

    return{
        "psi_entries": len(psi),
        "num_pa_pairs": len(all_pairs),
        "num_supported": len(supported),
        "frac_singly_observed": sum(s == 1 for s in supports) / len(supports),
        "max_support": max(supports),
        "rcd_raw": rcd_raw,
        "rcd_supported": rcd_supported,
        "rcd_weighted": rcd_weighted,
    }


def load_psi_for_run(run_id):
    """Download a bimachine artifact from W&B and return its psi table."""
    artifact = wandb.Api().artifact(f"{USER}/{PROJECT}/bimachine-{run_id}:latest")
    path = artifact.download()
    return load_tables(f"{path}/bimachine.json").psi
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="notes/rcd_results.csv")
    args = parser.parse_args()
 
    rows = []
    for dataset, run_id in DEFAULT_RUNS.items():
        print(f"\n=== {dataset}  (run {run_id})")
        try:
            stats = compute_rcd(load_psi_for_run(run_id))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
 
        stats = {"dataset": dataset, "run_id": run_id, **stats}
        rows.append(stats)
 
        print(f"  psi entries        : {stats['psi_entries']}")
        print(f"  (p,a) pairs        : {stats['num_pa_pairs']}")
        print(
            f"  supported pairs    : {stats['num_supported']} "
            f"({stats['num_supported'] / stats['num_pa_pairs']:.1%})"
        )
        print(
            f"  singly observed    : {stats['frac_singly_observed']:.1%}  "
            f"<- the sparsity exposure"
        )
        print(
            f"  support  mean/med/max : {stats['mean_support']:.2f} / "
            f"{stats['median_support']:.1f} / {stats['max_support']}"
        )
        print(f"  rcd_raw            : {stats['rcd_raw']:.4f}")
        print(f"  rcd_supported      : {stats['rcd_supported']:.4f}   <- headline")
        print(f"  rcd_weighted       : {stats['rcd_weighted']:.4f}")
 
    if not rows:
        print("\nNo datasets succeeded.")
        return
 
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
 
    print("\n--- ranked by rcd_supported (high to low) ---")
    for r in sorted(rows, key=lambda r: -r["rcd_supported"]):
        print(
            f"  {r['dataset']:16s} {r['rcd_supported']:.4f}   "
            f"(raw {r['rcd_raw']:.4f}, mean support {r['mean_support']:.2f})"
        )
 
 
if __name__ == "__main__":
    main()



