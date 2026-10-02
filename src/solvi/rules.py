"""Deprecated (0.8; removed in 0.9): the learned rule list is solvi.rulelist (`rules` also names the catalog's answer
rules, `@cat.rule`). Every name still reads from here, with a DeprecationWarning."""
from . import _deprecate

__getattr__ = _deprecate.module_getattr("solvi.rules", {n: "solvi.rulelist:" + n for n in ("RuleList", "words",
                                                                                              "literals")})
