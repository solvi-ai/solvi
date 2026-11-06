"""Author-defined finite fixtures. Fixture generation is outside participant units."""
from random import Random

FAMILIES = ("reconcile", "invoice", "inventory")
ARMS = ("DIRECT", "DIRTY", "FIFO", "CHEAP", "DISPATCH", "NO_CACHE", "NO_LEARN")
LIMITS = (400, 1200, 4000)

def generate(seed, family):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    rng = Random(seed)
    layouts = {
        "reconcile": [("fold", "blend", "a", "b"), ("total", "add", "c", "d"),
                      ("delta", "sub", "fold", "total"), ("amount", "max", "delta", "e"),
                      ("flag", "lt", "amount", "f")],
        "invoice": [("rate", "blend", "b", "c"), ("line", "mul", "a", "rate"),
                    ("discount", "add", "d", "e"), ("amount", "sub", "line", "discount"),
                    ("flag", "lt", "amount", "f")],
        "inventory": [("reserved", "blend", "c", "d"), ("stock", "add", "a", "b"),
                      ("available", "sub", "stock", "reserved"), ("amount", "min", "available", "e"),
                      ("flag", "lt", "amount", "f")],
    }
    nodes = [{"name": n, "kind": k, "deps": [x, y]} for n, k, x, y in layouts[family]]
    nodes.append({"name": "noise", "kind": "mul", "deps": ["e", "f"]})
    initial = {k: rng.randrange(4) for k in "abcdef"}
    current = dict(initial)
    updates = []
    for epoch in range(12):
        changes = {}
        if epoch and epoch not in (4, 9):
            key = rng.choice(tuple(current))
            value = (current[key] + rng.randrange(1, 4)) % 4
            changes[key] = value
            current[key] = value
        updates.append({"epoch": epoch, "changes": changes,
                        "revision": {"kind": "add", "revision": "r2"} if epoch == 6 else None})
    return {"spec": {"input_names": list("abcdef"), "nodes": nodes, "outputs": ["amount", "flag"]},
            "initial_inputs": initial, "updates": updates}
