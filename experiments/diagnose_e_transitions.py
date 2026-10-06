"""looking at epsilon transitions in failed automata"""
import json
import sys
from collections import defaultdict

data = json.load(open(sys.argv[1]))
states = data["states"]
finals = set(data["finals"])

eps_out = defaultdict(list)    # state -> targets reached by epsilon
real_out = defaultdict(list)   # state -> (symbol, target) for real symbols
for s, arcs in states.items():
    for label, tgt in arcs:
        if label[0] == "":
            eps_out[s].append(tgt)
        else:
            real_out[s].append((label[0], tgt))

n_states = len(states)
n_arcs = sum(len(a) for a in states.values())
n_eps = sum(len(t) for t in eps_out.values())

# how many
print(f"States: {n_states}   arcs: {n_arcs}   epsilon arcs: {n_eps}")
print(f"States with at least one epsilon arc: {len(eps_out)}")

# at most one epsilon arc per state?
multi = {s: t for s, t in eps_out.items() if len(t) > 1}
print(f"\nStates with more than one epsilon arc: {len(multi)}")
for s, t in list(multi.items())[:5]:
    print(f"    e.g. {s} -> {t}")

# chains
chains = sum(1 for s, ts in eps_out.items() for t in ts if t in eps_out)
print(f"\nEpsilon arcs that lead straight into another epsilon arc: {chains}")

# loops. 
remaining = {s: set(ts) for s, ts in eps_out.items()}
changed = True
while changed:
    changed = False
    for s in list(remaining):
        remaining[s] = {t for t in remaining[s] if t in remaining}
        if not remaining[s]:
            del remaining[s]
            changed = True
print(f"States on or leading into an epsilon loop: {len(remaining)}")

# word-final arcs
dead_end_final = cont_final = nonfinal = 0
for s, ts in eps_out.items():
    for t in ts:
        if t in finals and not real_out[t]:
            dead_end_final += 1   # lands on a final state that reads nothing more
        elif t in finals:
            cont_final += 1       # lands on a final state that can also keep reading
        else:
            nonfinal += 1         # lands mid-word
print(f"\nEpsilon arcs landing on a final state with nothing after it: {dead_end_final}")
print(f"    ...on a final state that can keep reading:                  {cont_final}")
print(f"    ...on a non-final state (mid-word):                         {nonfinal}")

# is the non-epsilon part deterministic?
nondet = sum(
    1 for s, arcs in real_out.items()
    if len({sym for sym, _ in arcs}) < len(arcs)
)
print(f"\n states with two arcs on the same real symbol: {nondet}")