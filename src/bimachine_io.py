from __future__ import annotations
import json
from pathlib import Path

from src.bimachine_to_fst import BimachineTables

FORMAT_VERSION = 1

def fst_to_tables(fst) -> tuple[str, frozenset[str], dict[tuple[str, str], str]]:
    """deterministic, epsilon-free pyfoma FSA -> (initial, finals, delta).
    """
    if fst.initialstate.name is None:
        raise ValueError("FSA has an unnamed initial state")
    initial: str = fst.initialstate.name
    finals = frozenset(s.name for s in fst.finalstates)
    if None in finals:
        raise ValueError("FSA has unnamed final states")

    delta: dict[tuple[str, str], str] = {}
    for state in fst.states:
        if state.name is None:
            raise ValueError("FSA has an unnamed state")
        for label, transitions in state.transitions.items():
            sym = label[0]
            if sym == "":
                raise ValueError(
                    f"epsilon transition out of state {state.name!r}; "
                    "delta cannot represent epsilons"
                )
            if len(transitions) > 1:
                raise ValueError(
                    f"nondeterministic: state {state.name!r} has "
                    f"{len(transitions)} transitions on input {sym!r}"
                )
            key = (state.name, sym)
            if key in delta:
                raise ValueError(
                    f"nondeterministic: state {state.name!r} has two labels "
                    f"sharing input symbol {sym!r}"
                )
            for t in transitions:
                if t.targetstate.name is None:
                    raise ValueError(f"transition from {state.name!r} to unnamed state")
                delta[key] = t.targetstate.name
    return initial, finals, delta

def dump_tables(tables: BimachineTables, path: str | Path) -> Path:
    """Serialize a BimachineTables to JSON. Returns the path written."""
    path = Path(path)
    obj = {
        "meta": {"format_version": FORMAT_VERSION},
        "forward": {
            "initial": tables.q_L0,
            "finals": sorted(tables.F_L),
            "delta": sorted(
                [p, a, p_prime] for (p, a), p_prime in tables.delta_L.items()
            ),
        },
        "backward": {
            "initial": tables.q_R0,
            "finals": sorted(tables.F_R),
            "delta": sorted(
                [r, a, r_prime] for (r, a), r_prime in tables.delta_R.items()
            ),
        },
        "psi": sorted([p, r, a, out] for (p, r, a), out in tables.psi.items()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_tables(path: str | Path) -> BimachineTables:
    """Inverse of dump_tables."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    version = obj["meta"]["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version {version}, this code reads {FORMAT_VERSION}"
        )
    fwd, bwd = obj["forward"], obj["backward"]
    return BimachineTables(
        q_L0=fwd["initial"],
        F_L=frozenset(fwd["finals"]),
        delta_L={(p, a): p_prime for p, a, p_prime in fwd["delta"]},
        q_R0=bwd["initial"],
        F_R=frozenset(bwd["finals"]),
        delta_R={(r, a): r_prime for r, a, r_prime in bwd["delta"]},
        psi={(p, r, a): out for p, r, a, out in obj["psi"]},
    )

if __name__ == "__main__":
    import tempfile

    from src.bimachine_to_fst import toy_R1

    with tempfile.TemporaryDirectory() as tmpdir:
        p = dump_tables(toy_R1(), Path(tmpdir) / "toy_R1.json")
        print(p.read_text(encoding="utf-8"))
        assert load_tables(p) == toy_R1(), "round trip changed the machine"
        q = dump_tables(load_tables(p), Path(tmpdir) / "toy_R1_again.json")
        assert p.read_bytes() == q.read_bytes(), "not byte-stable"
    print("round trip OK")