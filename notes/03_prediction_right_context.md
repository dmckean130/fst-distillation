# Right-Context Dependence: Predictions

**Written 12:00 2026-08-18 before viewing results.**
This file is not edited after commit.

---

RCD should be most prevalent in G2P datasets, followed by historical normalization, followed by inflection. Inflection has the least RCD because the inflection features sit at the left edge of the string, so a unidirectional RNN can separate states on them immediately. (See the addendum)

Predicted `rcd_supported`, highest to lowest. Published FST accuracy from Ginn et al. is included for reference. 

| Rank | Dataset | Task | Published FST acc. |
|---|---|---|---|
| 1 | `g2p/fre` | G2P | 0.200 |
| 2 | `g2p/dut` | G2P | 0.149 |
| 3 | `histnorm/deu` | Histnorm | 0.214 |
| 4 | `histnorm/swe` | Histnorm | 0.579 |
| 5 | `histnorm/isl` | Histnorm | 0.507 |
| 6 | `histnorm/spa` | Histnorm | 0.646 |
| 7 | `g2p/geo` | G2P | 0.596 |
| 8 | `inflection/czn` | Inflection | 0.666 |
| 9 | `inflection/kon` | Inflection | 0.846 |

G2P is the most right-context dependent of the three tasks, with inflection the least. The exception is Georgian, covered below.

**French and Dutch.** RCD is very high. Both languages have gone through serious phonological changes since writing was invented for their linguistic community, and so spelling reforms preserve certain morphological patterns rather than being purely grapheme → phoneme.

**Georgian.** Not like French and Dutch — it is purely grapheme → phoneme. One could read a syllable of Georgian out of context correctly (if the rest of the word were covered, for example), which is untrue for French and Dutch. This is why it is ranked below the historical normalization datasets despite being a G2P task.

**Historical normalization.** Also very right-context dependent, but not always. It generally depends on the type and amount of spelling reform in a language. The Germanic languages (German, Swedish, Icelandic) all had similar reforms that were right-context dependent in some cases but not others. *The exact ordering within these three is preliminary and based on fairly cursory knowledge of these reforms.* Spanish has a different history and less digraph reduction than the Germanic languages, which is why it is placed below them.

**Inflection.** Least right-context dependent, for the reason given in §1. Kongo is predicted lower than Zenzontepec Chatino because of the morphological regularity of Niger-Congo languages compared to Oto-Manguean languages. *Based on cursory knowledge of Swahili and Teotitlán del Valle Zapotec respectively, not of these two languages directly.*

Claims:

1. RCD ranks the datasets as predicted.
2. RCD correlates with where the bimachine beats the published FST number.
3. Determinization blows up on high-RCD datasets specifically.

On Claim 2, the quantity of interest is the *difference* between bimachine and FST, not raw bimachine accuracy. Since the FST number for Georgian is already much higher than French or Dutch, the prediction is that the bimachine improves on French and Dutch and stays roughly flat on Georgian (a large bimachine − FST gap for `fre` and `dut`, a small one for `geo`).

What would falsify these claims:

- The inflection datasets outrank the high-RCD G2P datasets (`fre`, `dut`) in bimachine performance relative to FST performance — i.e. the bimachine − FST difference is larger for inflection.
- The bimachine does not beat the published FST number in correlation with the RCD ranking (for example, the bimachine does very well on Georgian but not on French G2P).
- Determinization does not blow up on high-RCD datasets specifically.

Known Confounds:

**Sparsity.** ψ only contains triples observed in training. Small datasets have sparser output tables, which deflates measured RCD. `dut` (3,600 training examples) is more exposed to this than `swe` (8,465). Will report `rcd_supported` and `mean_support` alongside `rcd_raw`.

**Partial circularity in Claim 2.** `fre` (0.200) and `dut` (0.149) are Ginn et. al.'s worst G2P results and `geo` (0.596) is among its best, so the predicted ranking already tracks the published accuracy ordering. RCD is measured from a machine extracted from the same data the FST struggled with, so a correlation is partly expected by construction.

An independent test is histnorm `deu` (0.214) and `swe` (0.579), which are predicted adjacent in RCD. If RCD is measuring something, these two should show similar RCD despite the accuracy gap.

---
 
Addendum:

The mechanism in §1 claims inflection features sit at the left edge. The raw data files do not look that way:

```
i¹-hni²nkw-i¹-hni²V;PFV
u¹-hlya²nka¹-hlya²V;PFV
u-s-u¹kwa²nt-u-s-u¹kwa²V;HAB
```

Columns are lemma, inflected form, features — features last. Checked against `evaluate()` in `src/extract_bimachine.py`:

```python
if ex.features is not None:
    features = [f"[{f}]" for f in ex.features]
    input_string = features + input_string
    correct_output = features + correct_output
```

Features are prepended to the tokenized string, so the model reads `[V] [PFV] <sep> i ¹ - h n i ² <sink>`. 
