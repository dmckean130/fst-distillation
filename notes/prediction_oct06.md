# Prediction: coverage experiments (sets B and C)

Written 2026-10-06, before viewing any set B or C results. 

## Baselines (August, right-merge)

| Dataset | F1 | Acceptance | Singly-observed % | \|Q_L\| | RCD |
|---|---|---|---|---|---|
| deu | TODO | 0.279 | 80.8 | TODO | TODO |
| swe | TODO | TODO | TODO | TODO | TODO |
| geo | TODO | TODO | TODO | TODO | TODO |

| | acc | F1 | acc. when accepted | fwd % | bwd % | out % | \|Q_L\| |
|---|---|---|---|---|---|---|---|
| deu | 0.279 | 0.215 | 0.771 | 99.1 | 0.7 | 0.2 | 24,837 |
| isl | 0.722 | 0.557 | 0.772 | 94.2 | 3.5 | 2.3 | 23,984 |
| dut | 0.002 | 0.002 | (1/1) | 83.7 | 6.9 | 9.4 | 3,869 |
| spa | 0.682 | 0.557 | 0.817 | 37.6 | 16.7 | 45.7 | 1,140 |
| fre | 0.100 | 0.096 | 0.956 | 23.5 | 10.1 | 66.4 | 370 |
| swe | 0.787 | 0.487 | 0.619 | 18.2 | 6.2 | 75.6 | 111 |
| geo | 0.200 | 0.129 | 0.645 | 10.6 | 10.0 | 79.4 | 106 |

deu forward-error share: 99.1%.

## bpe vs right (deu, swe, geo)

The RNN is retrained on the merged data. Any difference smaller than the spread across the 12-run neural sweep is seed noise, not the merge.

Output-label vocabulary will be down on deu and swe; about the same on geo. This is because `right` creates a new label for every (inserted output + following output) combination. `bpe` keeps attaching an inserted output to the same frequent partner, so fewer distinct labels. 

Singly-observed % will be down a few points on deu and swe. This follows from a smaller vocabulary: the same evidence spread over fewer labels. 

Acceptance will be up a little on deu and swe because fewer rare labels means fewer missing ψ entries. But deu's problem is forward-state sparsity. 

Since the input side is unchanged; only output labels differ, \|Q_L\ will be about the same. 

Merging left means an inserted output is emitted at the earlier input symbol, before the next one is read. That output now depends on the right context. `right` merging delays it, which is the sequential-friendly choice. The bimachine absorbs this through its backward automaton, so F1 shouldn't suffer, but RCD should increase. 

Deu, then swe move most. geo should barely move, since Georgian G2P is close to one-to-one and has few ε-inputs to merge. Treat geo as a near-control: if it moves a lot, something other than the merge changed.

Nonsensical merges expected in deu and swe, among the low-frequency merges
late in the merge order. The top-30 merges should look sensible. A nonsense
merge glues an inserted output to a neighbour it doesn't belong to, just
because that pair was frequent elsewhere.

The left-vs-right share per ε-pair should predict the RCD change. More left merges, bigger RCD increase.

## augmentation × cluster cap (deu, geo)

This dataset reuses the cached RNN, so unlike the previous dataset it has no retraining noise.

Acceptance vs augment factor (0 → 2 → 5) on deu: clear increase from 0 to 2, levelling off by 5. geo: small increase. Augmentation adds evidence for unseen transitions. Deu has the most missing rows; geo is already dense 

Singly-observed % goes down along the same axis, deu much more than geo. Same mechanism: more strings pass through each transition

F1 on deu goes up but much less than acceptance. Geo stays the same. F1 is capped by the RNN. Newly accepted strings are only correct where the RNN was right, and pseudo-labels teach ψ the RNN's errors 

 \|Q_L\ goes up modestly with augmentation. More observed transitions means more nondeterminism inside clusters for state splitting to resolve

 Cluster cap (small → current): Smaller cap: acceptance up, F1 down. Deu has a small cap that loses little F1; geo doesn't. Fewer states means each row sees more evidence, at the cost of merging states that should differ 

Augmentation helps more at the current (large) cap than at the small cap. Small caps already pack evidence into fewer rows, so extra strings add less. This is the main reason to cross the two factors.

## What would surprise me

- geo moving as much as deu under either set.
- bpe increasing the output vocabulary on deu.
- RCD going down under bpe.
- Augmentation raising acceptance on deu while F1 falls noticeably. That would
  mean pseudo-labels are actively poisoning ψ, not just failing to help.
- acceptance moves but singly-observed % doesn't, or vice versa. Then
  acceptance isn't driven by sparsity, and something else (e.g. forward-state
  count itself) is the cause.
- the smallest cluster cap fixes deu's acceptance with no F1 loss. Then the
  problem was too many states all along, not too little data, and augmentation
  is the wrong method.