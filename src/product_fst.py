import logging
 
from pyfoma.atomic import State
from pyfoma.fst import FST
 
logger = logging.getLogger(__name__)
 
ROOT_NAME = "__product_root__"
 
 
def product_to_pyfoma(arcs, finals, start):
    """Build a pyfoma FST from the raw product construction. 
       Returns pyfoma FST with a single synthetic initial state.
    """
    states: dict[tuple, State] = {}
 
    def get(key):
        if key not in states:
            states[key] = State(name=f"{key[0]}|{key[1]}")
        return states[key]
 
    # materialise every state first so add_transition always has a target.
    for src, _, _, dest in arcs:
        get(src)
        get(dest)
    for key in finals:
        get(key)
    for key in start:
        get(key)
 
    for src, a, out, dest in arcs:
        label = (a,) if a == out else (a, out)
        get(src).add_transition(other=get(dest), label=label, weight=1)
 
    final_states: set[State] = set()
    for key in finals:
        st = get(key)
        st.finalweight = 0
        final_states.add(st)
 
    # synthetic root, epsilon into each initial state.
    root = State(name=ROOT_NAME)
    for key in start:
        root.add_transition(other=get(key), label=("",), weight=0)
 
    # if any start state is also final, the empty string is accepted,
    # so the root must be final too.
    if any(key in set(finals) for key in start):
        root.finalweight = 0
        final_states.add(root)
 
    fst = FST()
    fst.states = set(states.values()) | {root}
    fst.initialstate = root
    fst.finalstates = final_states
    return fst
 
 
def describe(fst, label):
    """Print state/transition counts for one stage of the pipeline."""
    n_arcs = sum(
        len(list(tr))
        for st in fst.states
        for tr in getattr(st, "transitions", {}).values()
    )
    print(f"  {label:28s} states={len(fst.states):>8}  arcs={n_arcs:>8}")
    return fst
 
 
def convert_and_reduce(arcs, finals, start, do_determinize=True, do_minimize=True):
    """Full pipeline: build -> filter_accessible -> determinize -> minimize.
       Returns (fst, outcome_dict). Never raises on determinize/minimize failure.
    """
    outcome = {}
 
    fst = product_to_pyfoma(arcs, finals, start)
    outcome["states_built"] = len(fst.states)
    describe(fst, "built")
 
    fst = fst.filter_accessible()
    outcome["states_accessible"] = len(fst.states)
    describe(fst, "filter_accessible")
 
    if do_determinize:
        try:
            det = fst.determinize()
            outcome["determinize"] = "completed"
            outcome["states_determinized"] = len(det.states)
            fst = describe(det, "determinize")
        except NotImplementedError as e:
            outcome["determinize"] = f"not_supported: {e}"
            print(f"  determinize NOT SUPPORTED: {e}")
        except Exception as e:
            outcome["determinize"] = f"failed: {type(e).__name__}: {e}"
            print(f"  determinize FAILED: {type(e).__name__}: {e}")
 
    if do_minimize:
        try:
            mini = fst.minimize()
            outcome["minimize"] = "completed"
            outcome["states_minimized"] = len(mini.states)
            fst = describe(mini, "minimize")
        except Exception as e:
            outcome["minimize"] = f"failed: {type(e).__name__}: {e}"
            print(f"  minimize FAILED: {type(e).__name__}: {e}")
 
    return fst, outcome
 
 
def _try_apply(fst, string):
    for method in ("generate", "apply", "analyze"):
        fn = getattr(fst, method, None)
        if fn is None:
            continue
        try:
            return method, list(fn(string))
        except Exception as e:
            return method, f"raised {type(e).__name__}: {e}"
    return None, "no apply-like method found"
 
 
def main():
    from src.bimachine_to_fst import bimachine_to_fst, toy_R1
 
    print("=== toy R1 = {<a,x>, <ab,y>} ===")
    arcs, finals, start, status = bimachine_to_fst(toy_R1())
    print(f"  construction status: {status}")
    print(f"  arcs={len(arcs)}  finals={len(finals)}  start={len(start)}")
    print()
 
    fst, outcome = convert_and_reduce(arcs, finals, start)
 
    print("\n  outcome:")
    for k, v in outcome.items():
        print(f"    {k:22s} {v}")
 
    print("\n  behaviour check (expect a->x, ab->y):")
    for s in ("a", "ab"):
        method, result = _try_apply(fst, s)
        print(f"    {s!r:6s} via {method}: {result}")
 
 
if __name__ == "__main__":
    main()