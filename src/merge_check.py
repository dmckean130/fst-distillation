"""Quick merge-sanity check: right vs bpe output labels.

Run from the repo root on CURC:
    python merge_sanity.py data/histnorm/deu/aligned/deu.trn.aligned
    python merge_sanity.py data/g2p/aligned/geo.trn.aligned
"""
import random
import sys
from collections import Counter

from src.data.aligned.example import load_examples_from_file

path = sys.argv[1]
random.seed(0)

for merge in ["right", "bpe"]:
    exs, _ = load_examples_from_file(path, merge_outputs=merge)
    labels = Counter(o for ex in exs for _, o in ex.aligned_chars)
    singles = sum(1 for c in labels.values() if c == 1)
    print(f"{merge:5s}  output vocab = {len(labels):5d}   seen once = {singles:5d}")

# Eyeball 20 bpe-merged pairs: does the output belong to that input?
exs, merges = load_examples_from_file(path, merge_outputs="bpe")
print(f"\n{len(merges or [])} bpe merges learned. Top 10:")
for m in (merges or [])[:10]:
    print("  ", m)

merged = [(i, o, ex) for ex in exs for i, o in ex.aligned_chars if len(o) > 1 and i != "<sink>"]
print("\n20 random merged pairs (input -> output, in word):")
for i, o, ex in random.sample(merged, min(20, len(merged))):
    word = "".join(c for c, _ in ex.aligned_chars[:-1])
    print(f"  {i} -> {o:8s}  {word}")