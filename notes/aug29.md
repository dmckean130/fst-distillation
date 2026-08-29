## Aug 29
# error analysis

Working notes. Claude helped. 

**F1 is capped by acceptance.** `error_total` is exactly the count of rejected
strings, and a rejected string scores as wrong. The published Ginn et al. FST
numbers were computed on machines with `accepted_percentage: 1`, so comparing
raw F1 against them is comparing coverage-times-accuracy against pure accuracy.

| | acc | F1 | acc. when accepted | fwd % | bwd % | out % | \|Q_L\| |
|---|---|---|---|---|---|---|---|
| deu | 0.279 | 0.215 | 0.771 | 99.1 | 0.7 | 0.2 | 24,837 |
| isl | 0.722 | 0.557 | 0.772 | 94.2 | 3.5 | 2.3 | 23,984 |
| dut | 0.002 | 0.002 | (1/1) | 83.7 | 6.9 | 9.4 | 3,869 |
| spa | 0.682 | 0.557 | 0.817 | 37.6 | 16.7 | 45.7 | 1,140 |
| fre | 0.100 | 0.096 | 0.956 | 23.5 | 10.1 | 66.4 | 370 |
| swe | 0.787 | 0.487 | 0.619 | 18.2 | 6.2 | 75.6 | 111 |
| geo | 0.200 | 0.129 | 0.645 | 10.6 | 10.0 | 79.4 | 106 |

deu accepts 28% of test strings and gets 77% of those right, against an FST
that accepts everything and gets 20% right. These machines are **precise and
under-covering**, not inaccurate. dut's "1.000" is 1 accepted string out of 450. 

**Forward-dominated** (deu 99%, isl 94%, dut 84%): the left FSA rejects
strings it should accept. The machine's coverage is broken, not its table.
Fix direction: fewer clusters / coarser state merging.

**Output-dominated** (geo 79%, swe 76%, fre 66%): the FSAs traverse fine but
ψ has no entry for the (p, r, a) triple. The structure is right, the table is
sparse. Fix direction: more data, or a backoff when the exact triple is
missing.

Backward errors are small everywhere (0.7–17%). The right automaton is not
the bottleneck in any dataset.

Forward-error share tracks forward-state count almost monotonically:

```
24,837 -> 99%    3,869 -> 84%     370 -> 24%     106 -> 11%
23,984 -> 94%    1,140 -> 38%     111 -> 18%
```

Bigger left automaton -> more states -> more specific -> refuses more strings.
Over-fragmentation in clustering.

This is the same quantity that showed up twice before:

1. **Block 4** — sparsity exposure (frac singly observed): geo 27.5%,
   fre 36.9%, dut 65.6%, deu 80.8%. Same ordering.
2. **Block 6** — minimization compression: geo 19%, fre 55%, dut 70%,
   deu 89%. Same ordering.
3. **Block 8** — forward-error share. Same ordering.

One underlying variable — how finely the clustering fragmented the left
automaton — is driving sparsity, product redundancy, and rejection rate at
once.

- **Report accuracy-when-accepted alongside F1**, always with the acceptance
  denominator. Raw F1 vs. the published numbers is not a like-for-like
  comparison and I should stop treating it as one.
- **Number of clusters is the hyperparameter that matters.** It is currently
  swept blind. Worth an explicit ablation: sweep cluster count and watch
  acceptance, forward-error share, and RCD together.
- **RCD is confounded by fragmentation.** Block 4 already knew sparsity was a
  problem; this says the sparsity is itself a function of a hyperparameter,
  not of the language. Any RCD-vs-language claim needs cluster count
  controlled or matched across datasets.
- The forward/output split gives a cheap diagnostic for which direction to
  push per dataset. Worth logging as a ratio directly.

**caveats:**

- Best run per sweep is selected by `eval.f1`, which is also acceptance-capped.
  So the sweep may be picking machines that are good at coverage rather than
  good at the task. Worth re-selecting on accuracy-when-accepted and seeing
  whether different runs win.
- swe/spa/isl have no artifacts (epsilon guard), so no RCD or product size to
  cross-reference against their error profiles.
- Set C absent entirely — inflection has no bimachine path.


# predictions versus reality 

Does rcd_supported rank the datasets the way you predicted? fre/dut high, inflection low.

Raw RCD ranks fre, dut, geo, deu, which is very different than proposed rankings, where geo was ranked very low in comparison to fre, dut, and deu. However, when adjusting to Supported RCD (which was the predicted headline metric for RCD), the RCD is ranked fre, deu, dut, geo (weighted is the same). This is different than my rankings but isn't that suprising. Dutch g2p versus German histnorm predictions were based on *"fairly cursory knowledge of [historical spelling] reforms"*. The idea that German histnorm has higher RCD than Dutch g2p isn't something unexpected. The claimed falisifier for RCD ranking would be *"The inflection datasets outrank the high-RCD G2P datasets (`fre`, `dut`)"* - since inflection hasn't been tested, this falsifier hasn't been achieved. 

Generally, inflection having a high RCD is the falsifier here. Histnorm versus g2p isn't a result that's as problematic for my hypothesis. The geo ranking (when weighted/supported) is the real result, which is what I was looking for. I would say the RCD is broadly in line with the predicted results. 

Does RCD correlate with where bimachines beat the published FST numbers? 

This is a fairly clear negative result. The bimachines almost never beat the published FST numbers, and when they did it was by negligible amounts. The delta for FST-bimachine also doesn't correlate with RCD, even given this fact. French g2p delta is −0.104 while German histnorm delta is +0.001, dispite French being higher RCD and them both being the highest RCD datasets. 

One thing that may save this claim is that georgian g2p has the largest delta at −0.467. This does correlate with my claim, although framed differently: the bimachine was closest to performing at FST level with the high RCD data compared with the low RCD data. Still, French being so bad dents this claim as well. 

Since the claim had to do with *difference*: French, Dutch high, Georgian low, this claim is clearly falsified. The absolute value of the delta is objectively the highest for Georgian, and lowest for the histnorms. 

The clearer bifurcation in bimachine performance is histnorm vs g2p. The bimachine was consistently worse on g2p data both in raw f1 and in f1 delta.

This evidence is made much weaker by the fact that the delta is not a like-for-like comparison. The FST numbers are for FSTs that accept every string, while the bimachines have capped acceptance (in some cases, at 20%) which drastically affects f1. 

If we consider accepted strings only for the bimachines, the performance markedly improves and more closely matches claimed performance ranking. Besides g2p Dutch's 1/1 (which isn't a very useful result), French scores the highest, followed by the histnorm data and Georgian, with Georgain beating Swedish histnorm. This broadly tracks our RCD ranking - fre is highest and geo is near the bottom, which is directionally consistent. This ordering can't carry too much weight as the coverage is wildly different across data sets (geo over 20% coverage, swe over 79%). 

The spearman correlation doesn't provide any statistically significant evidence. 

Does determinization blow up on exactly the high-RCD datasets? 

The determinization test was not performed (yet). 

Some other indicators were found tracking a ranking that isn't based on RCD: 
| Quantity | geo | fre | dut | deu |
|---|---|---|---|---|
| \|Q_L\| (forward states) | 106 | 370 | 3,869 | 24,837 |
| singly-observed (p,a) pairs | 27.5% | 36.9% | 65.6% | 80.8% |
| minimization compression | 19% | 55% | 70% | 89% |
| forward-error share | 10.6% | 23.5% | 83.7% | 99.1% |
This ranking is an interesting discovery and deserves further investigation. 

-- 

We considered deu vs swe to be the independent test, but this test is not complete as we don't have final comparative numbers for the swe dataset (e.g., RCD). The bimachine underperformed Ginn's FST on swe while slightly beating Ginn's FST on deu. 

Overall, the RCD correlation claim is slightly correct, but also perhaps slightly wrong. More information may be required, and we found some other numbers that could be interesting to examine as well. The biggest limitation to these findings is that RCD isn't available for 5 of 9 datasets. More technical fixes to examine all the data is the most important change. 




