# Oct 5 — Sprint Day 1

*Working notes. Written with Claude.*

## Done

- Fork shared with Michael. Detaching it from the original repo: ask him Oct 13.
- Bad node fixed for real: `--exclude=bgpu-g6-u25` is now in `submit_preemptable.sh`. Verified with `scontrol` (`ExcNodeList=bgpu-g6-u25`).
- `.gitignore` now ignores `*.pyc` and `__pycache__/`.
- Set B (bpe merge) extraction finished for deu and geo. Ahead of plan (was Friday).
- Fixed the validation-loss bug (below). Relaunched swe/spa/isl and set B on swe.
- Epsilon diagnosis done (below). Guard removed; machines with ε can now be saved.

## Bugs found and fixed

Use `--exclude` on the sbatch line instead of `SBATCH_EXCLUDE`. `CUDA_VISIBLE_DEVICES=""` was only a workaround (CPU only), and it can't be used for jobs that train RNNs.

W&B now returns the loss as a flat key, `"validation.loss"`. The code only looked for the nested form, `["validation"]["loss"]`, so it got a `KeyError` right before extraction. This broke reuse for every dataset.

Fix in `src/sweep.py`:
- `_val_loss()` reads either form.
- `_best_finished_run()` picks the lowest validation loss among finished runs. It replaces `sweep.best_run()` everywhere except the extraction sweep at the end, which ranks by F1.

Best runs picked:

| Dataset | Merge | Run | Val loss |
|---|---|---|---|
| swe | right | crisp-sweep-12 | 1.0126 |
| deu | bpe | charmed-sweep-9 | 1.4004 |
| geo | bpe | rose-sweep-3 | 0.8310 |

TODO: confirm swe's pick matches the run August's swe extractions used.

**3. Epsilon guard blocked saving swe/spa/isl.**
Fix:
- `fst_to_tables` now keeps ε arcs as `delta[(state, "")]`. The existing determinism checks still forbid two ε arcs leaving the same state.
- `bimachine_to_fst` now raises `NotImplementedError` on ε instead of silently building a wrong product.
- RCD only reads ψ, so it works on these machines.

## Epsilon findings (swe, forward automaton, one extraction run)

| Question | Answer |
|---|---|
| How many? | 7 ε arcs out of 5,626 total arcs (565 states) |
| More than one ε arc per state? | No |
| Chains or loops? | None |
| Word-final? | **No.** All 7 go to the **same** state, `state-905`. It's non-final and reads `1`, `2`, `4`, `a`, `<sink>` |

- `remove_epsilon_loops.py` is never called in the pipeline. It's dead code; only its own test runs it. 
- The word-final hypothesis is wrong in its simple form. Don't predict that bpe removes ε arcs.
- Set B on deu/geo produced no ε arcs (no dumps from those jobs).

generate()` falls back to `""` when no normal transition exists. Are these arcs really a catch-all rather than true ε-moves? If so, conversion might be easy to fix.

## Jobs

| Job | What | Status |
|---|---|---|
| 28605949 | set B deu (bpe) | done |
| 28605950 | set B geo (bpe) | done |
| 28609636 | swe relaunch (right) | running |
| 28609667 | spa relaunch (right) | running |
| 28609676 | isl relaunch (right) | running |
| 28609677 | set B swe (bpe) | running |

## Tomorrow morning

- Check the 4 running jobs. New runs should have a `bimachine-<run id>` artifact and no `bimachine_serialization_error`.
- Conversion on swe/spa/isl will hit the new ε error. Have the harvest skip conversion for them and record it.
