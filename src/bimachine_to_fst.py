from dataclasses import dataclass
from collections import deque, defaultdict

State = str

@dataclass(frozen=True)
class Bimachine:
    q_L0: State
    F_L: frozenset[State]
    delta_L: dict[tuple[State, str], State]
    q_R0: State
    F_R: frozenset[State]
    delta_R: dict[tuple[State, str], State]
    psi: dict[tuple[State, State, str], str]

def toy_R1() -> Bimachine:
    return Bimachine(
        q_L0="0",
        F_L=frozenset({"0", "1"}),
        delta_L={("0", "a"): "1",  
                 #arbitrary
                ("0", "b"): "0", ("1", "a"): "0", ("1", "b"): "0"},
        q_R0="0",
        F_R=frozenset({"0", "1"}),
        delta_R={("0", "b"): "1", 
                   #arbitrary
                   ("0", "a"): "0", ("1", "a"): "0", ("1", "b"):"1" },
        psi={("0", "0", "a"): "x", ("0", "1", "a"): "y", ("1", "0", "b"): ""}
    )

def _validate(bm: Bimachine) -> None:
    Q_L = {bm.q_L0} | {p for (p, _) in bm.delta_L} | set(bm.delta_L.values())
    Q_R = {bm.q_R0} | {r for (r, _) in bm.delta_R} | set(bm.delta_R.values())
    Sigma = {a for (_, a) in bm.delta_L}
    assert(bm.F_L <= Q_L)
    assert(bm.F_R <= Q_R)
    for (p, r, a), x in bm.psi.items(): 
        assert(p in Q_L)
        assert(r in Q_R)
        assert(a in Sigma)
    Rigma = {a for (_, a) in bm.delta_R}
    assert(Rigma == Sigma)


def build_reverse_index(delta_R):
    reverse_index = defaultdict(set)
    for (r_prime, a), r in delta_R.items():
        reverse_index[(r, a)].add(r_prime)
    return dict(reverse_index)

def bimachine_to_fst(bm: Bimachine, max_states: int = 10**6):
    pre_R = build_reverse_index(bm.delta_R)
    Sigma = {a for (_, a) in bm.delta_L}
    start = {(bm.q_L0, r) for r in bm.F_R}   # (q_L0, r) for every r in F_R
    seen = set(start)
    queue = deque(start)
    arcs, finals = [], []
    while queue:
        p, r = queue.popleft()
        if p in bm.F_L and r == bm.q_R0:     # Q2
            finals.append((p, r))

        for a in Sigma:
            p_prime = bm.delta_L.get((p, a))
            if p_prime is None: 
                continue
            for r_prime in pre_R.get((r, a), ()):
                #if (p, r_prime, a) not in bm.psi:  
                    #continue
                out  = bm.psi.get((p, r_prime, a))
                if out is None:
                    continue
                #p_prime = bm.delta_L[(p, a)]
                dest = (p_prime, r_prime)
                arcs.append(((p,r), a, out, dest))
                if dest not in seen:
                    if len(seen) >= max_states:
                        return(arcs, finals, start, "capped")
                    seen.add(dest)
                    queue.append(dest)
    return arcs, finals, start, "not capped"

#bm = toy_R1()
#_validate(bm)
#arcs, finals, start, status = bimachine_to_fst(bm)
#print("start ", sorted(start))
#print("finals", sorted(finals))
#for arc in sorted(arcs):
    #print(arc)
