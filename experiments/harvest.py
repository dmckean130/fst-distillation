"""Block 7: harvest every result into one table.
Usage:
    python -m experiments.harvest
    python -m experiments.harvest --debug-keys   # dump summary keys and exit
"""

import argparse
import csv
import os
from collections import defaultdict

import wandb

ENTITY = "dmckean130-university-of-colorado-boulder"
PROJECT = "fst-distillation.extraction.v2"

# Which task each dataset belongs to, and which comparison set.
#   A = matched-budget paired comparison (bimachine AND transduction, 12/25)
#   B = bimachine only, compared against the paper's FULL-budget FST numbers
#   C = inflection control (blocked: features require --mode sample, which
#       extract_bimachine does not implement)
DATASETS = {
    "deu": ("histnorm", "A", 0.214),
    "swe": ("histnorm", "A", 0.579),
    "geo": ("g2p", "A", 0.596),
    "fre": ("g2p", "B", 0.200),
    "dut": ("g2p", "B", 0.149),
    "spa": ("histnorm", "B", 0.646),
    "isl": ("histnorm", "B", 0.507),
    "czn": ("inflection", "C", 0.666),
    "kon": ("inflection", "C", 0.846),
}


def get_metric(summary, key):
    """Read a metric that W&B may store flat ('test.f1') or nested."""
    try:
        if key in summary:
            return summary[key]
    except (TypeError, KeyError):
        pass
    head, _, tail = key.partition(".")
    if not tail:
        return None
    sub = summary.get(head)
    if sub is None:
        return None
    try:
        return sub[tail]
    except (TypeError, KeyError):
        return None
    
    '''if key in summary:
        return summary[key]
    head, _, tail = key.partition(".")
    if tail and isinstance(summary.get(head), dict):
        return summary[head].get(tail)
    return None'''


def parse_sweep_name(name):
    """'deu.deu.bimachine.right' -> ('deu', 'bimachine', 'right').

    Returns None for pre-Day-2 bare names, which pooled objectives together
    and are not usable for the paired comparison.
    """
    parts = name.split(".")
    if len(parts) != 4:
        return None
    _, dataset, objective, merge = parts
    if dataset not in DATASETS:
        return None
    return dataset, objective, merge


def newest_sweep_per_arm(project):
    """Pick one sweep per (dataset, objective), preferring the most recent."""
    candidates = defaultdict(list)
    for sweep in project.sweeps():
        parsed = parse_sweep_name(sweep.name)
        if parsed is None:
            continue
        dataset, objective, merge = parsed
        runs = [r for r in sweep.runs if r.state == "finished"]
        if not runs:
            continue
        newest = max(r.created_at for r in runs)
        candidates[(dataset, objective)].append((newest, merge, sweep, runs))

    chosen = {}
    for arm, options in candidates.items():
        options.sort(key=lambda x: x[0], reverse=True)
        chosen[arm] = options[0]
        if len(options) > 1:
            print(f"  {arm[0]}/{arm[1]}: {len(options)} sweeps, "
                  f"using newest ({options[0][0]})")
    return chosen


def harvest_arm(dataset, objective, merge, sweep, runs):
    """Build one long-form row for a (dataset, objective) arm."""
    best = sweep.best_run()
    s = best.summary
    task, comparison_set, published = DATASETS[dataset]

    row = {
        "dataset": f"{task}/{dataset}",
        "short": dataset,
        "task": task,
        "set": comparison_set,
        "objective": objective,
        "merge_outputs": merge,
        "sweep": sweep.name,
        "n_finished_runs": len(runs),
        "best_run": best.name,
        "best_run_id": best.id,
        "published_fst_f1": published,
    }

    for key, col in [
        ("test.f1", "test_f1"),
        ("test.cer", "test_cer"),
        ("eval.f1", "eval_f1"),
        ("eval.cer", "eval_cer"),
        ("test.accepted_percentage", "test_accepted"),
        ("num_states", "num_states"),
        ("num_forward_states", "num_forward_states"),
        ("num_backward_states", "num_backward_states"),
        ("output_table_size", "output_table_size"),
        ("product_upper_bound", "product_upper_bound"),
        ("eval.error_total", "error_total"),
        ("test.error_total", "test_error_total"),
    ]:
        row[col] = get_metric(s, key)

    # Error breakdown -- name uncertain, so try several and record what hit.
    for kind in ("forward", "backward", "output"):
        row[f"error_{kind}"] = get_metric(s, f"eval.error_{kind}")
        row[f"test_error_{kind}"] = get_metric(s, f"test.error_{kind}")
        val = None
        for candidate in (f"eval.error_{kind}", f"error_{kind}",
                          f"eval.errors.{kind}", f"eval.error_types.{kind}"):
            val = get_metric(s, candidate)
            if val is not None:
                break
        row[f"error_{kind}"] = val

    row["serialization_error"] = get_metric(s, "bimachine_serialization_error")
    return row


def load_join(path, key="dataset"):
    """Load a CSV keyed by dataset path (e.g. 'g2p/geo')."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping join")
        return {}
    with open(path) as f:
        return {r[key]: r for r in csv.DictReader(f)}


def fmt(v, places=3):
    if v is None or v == "":
        return "--"
    try:
        return f"{float(v):.{places}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_int(v):
    if v is None or v == "":
        return "--"
    try:
        return f"{int(float(v)):,}"
    except (TypeError, ValueError):
        return str(v)


def build_wide(rows, rcd, conv):
    """One row per dataset: both objectives side by side, plus RCD and sizes."""
    by_dataset = defaultdict(dict)
    for r in rows:
        by_dataset[r["dataset"]][r["objective"]] = r

    wide = []
    for dataset, arms in sorted(by_dataset.items()):
        bi = arms.get("bimachine", {})
        tr = arms.get("transduction", {})
        base = bi or tr
        rcd_row = rcd.get(dataset, {})
        conv_row = conv.get(dataset, {})

        wide.append({
            "dataset": dataset,
            "set": base.get("set"),
            "published_fst_f1": base.get("published_fst_f1"),
            "bimachine_f1": bi.get("test_f1"),
            "bimachine_cer": bi.get("test_cer"),
            "fst_f1": tr.get("test_f1"),
            "fst_cer": tr.get("test_cer"),
            "delta_vs_published": (
                bi["test_f1"] - base["published_fst_f1"]
                if bi.get("test_f1") is not None else None
            ),
            "delta_vs_matched_fst": (
                bi["test_f1"] - tr["test_f1"]
                if bi.get("test_f1") is not None and tr.get("test_f1") is not None
                else None
            ),
            "num_forward_states": bi.get("num_forward_states"),
            "num_backward_states": bi.get("num_backward_states"),
            "fst_num_states": tr.get("num_states"),
            "output_table_size": bi.get("output_table_size"),
            "rcd_raw": rcd_row.get("rcd_raw"),
            "rcd_supported": rcd_row.get("rcd_supported"),
            "rcd_weighted": rcd_row.get("rcd_weighted"),
            "mean_support": rcd_row.get("mean_support"),
            "frac_singly_observed": rcd_row.get("frac_singly_observed"),
            "product_trimmed": conv_row.get("states_trimmed"),
            "product_determinized": conv_row.get("states_determinized"),
            "product_minimized": conv_row.get("states_minimized"),
            "error_forward": bi.get("error_forward"),
            "error_backward": bi.get("error_backward"),
            "error_output": bi.get("error_output"),
            "error_total": bi.get("error_total"),
            "test_error_total": bi.get("test_error_total"),
            "test_error_forward": bi.get("test_error_forward"),
            "test_error_backward": bi.get("test_error_backward"),
            "test_error_output": bi.get("test_error_output"),
        })
    return wide


LATEX_CAPTION = r"""\caption{Bimachine and FST results. \textbf{Set A} is a
matched-budget paired comparison: both arms ran 12 neural and 25 extraction
runs, so \emph{Bimachine} and \emph{FST (ours)} are directly comparable.
\textbf{Set B} has no matched FST arm; its bimachine ran at the same reduced
budget but is compared against the \emph{full}-budget published numbers of
Ginn et al., so those differences understate the bimachine. Set C
(inflection) is absent: features require \texttt{--mode sample}, which is
unimplemented for bimachine extraction. RCD is right-context dependence,
restricted to $(p,a)$ pairs observed with at least two distinct right-states.
Product sizes are available only for datasets whose bimachines serialized;
\texttt{swe}, \texttt{spa} and \texttt{isl} hit the epsilon guard.}"""


def write_latex(wide, path):
    cols = ["Dataset", "Set", "Published", "Bimachine", "FST (ours)",
            "RCD", "Prod.", "Min."]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        " & ".join(cols) + r" \\", r"\midrule",
    ]
    for r in wide:
        lines.append(" & ".join([
            r["dataset"].replace("_", r"\_"),
            str(r["set"]),
            fmt(r["published_fst_f1"]),
            fmt(r["bimachine_f1"]),
            fmt(r["fst_f1"]),
            fmt(r["rcd_supported"]),
            fmt_int(r["product_trimmed"]),
            fmt_int(r["product_minimized"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", LATEX_CAPTION,
              r"\label{tab:results}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-out", default="notes/harvest_runs.csv")
    p.add_argument("--table-out", default="notes/results_table.csv")
    p.add_argument("--latex-out", default="notes/results_table.tex")
    p.add_argument("--rcd", default="notes/rcd_results.csv")
    p.add_argument("--conversions", default="notes/conversion_results.csv")
    p.add_argument("--debug-keys", action="store_true")
    args = p.parse_args()

    api = wandb.Api()
    project = api.project(name=PROJECT, entity=ENTITY)

    print("Selecting sweeps:")
    chosen = newest_sweep_per_arm(project)

    if args.debug_keys:
        (_, _, sweep, _) = next(iter(chosen.values()))
        print("\nSummary keys on", sweep.best_run().name)
        for k in sorted(sweep.best_run().summary.keys()):
            print("   ", k)
        return

    rows = []
    for (dataset, objective), (_, merge, sweep, runs) in sorted(chosen.items()):
        rows.append(harvest_arm(dataset, objective, merge, sweep, runs))
    write_csv(rows, args.runs_out)
    print(f"\nWrote {len(rows)} arms to {args.runs_out}")

    wide = build_wide(rows, load_join(args.rcd), load_join(args.conversions))
    write_csv(wide, args.table_out)
    write_latex(wide, args.latex_out)
    print(f"Wrote {len(wide)} datasets to {args.table_out} and {args.latex_out}")

    hdr = (f"\n{'dataset':16s}{'set':4s}{'pub':>7s}{'bimach':>8s}{'fst':>8s}"
           f"{'d_pub':>8s}{'RCD':>7s}{'prod':>9s}{'min':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in wide:
        print(f"{r['dataset']:16s}{str(r['set']):4s}"
              f"{fmt(r['published_fst_f1']):>7s}{fmt(r['bimachine_f1']):>8s}"
              f"{fmt(r['fst_f1']):>8s}{fmt(r['delta_vs_published']):>8s}"
              f"{fmt(r['rcd_supported']):>7s}{fmt_int(r['product_trimmed']):>9s}"
              f"{fmt_int(r['product_minimized']):>8s}")


if __name__ == "__main__":
    main()