# Explicit mutation sites

`solvi.testing.mutants.generate` and `run` mutate numeric literal match patterns
(two threshold nudges), string literal match patterns (a changed literal), and
boolean singleton patterns (True/False flip). Captured/sequence/class/OR patterns
and the None singleton are not pattern mutation sites.

A multi-statement `if` body ending in a direct `raise`, without an `else`, can be
dropped by making that guard false. An `assert` can separately be made to pass.
These mutations deliberately change the selected guard; they do not infer arbitrary
control flow or guarantee that every exception site can be dropped.

Checks registered through a helper decorator or call form can be declared by source
name, without importing the task to discover callbacks:

```python
from solvi.testing import mutants

report = mutants.run("cases.json", check_functions=["known", "Rules.positive"])
```

`check_functions` is an explicit registration manifest, not a callable or automatic
Catalog scan. Names must uniquely identify functions in the task source; unknown,
ambiguous and malformed declarations are rejected. `functions=` still limits the
selected source scope. Direct `@cat.check` / `@cat.constraint` decorators need no
manifest. No helper or registration call is executed during generation; the normal
isolated baseline and mutant lifecycles still execute the task when `run` is used.

The report records the manifest. AST deduplication and the optional exact-builtin-int
equivalence grouping retain their existing score units and alias records. This is
coverage of these finite source constructs, not mutation completeness for Python.
