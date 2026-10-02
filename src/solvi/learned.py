"""Deprecated (0.8; removed in 0.9): solvi.learned was split — CostBook and MeasuredCosts are in solvi.costs, the learned
order and producer policy (OrderModel, ProducerPolicy, Binary, scalar_row) in solvi.strategist. Every name still reads
from here, with a SolviDeprecationWarning."""
from . import _deprecate

__getattr__ = _deprecate.module_getattr("solvi.learned", {
    "CostBook": "solvi.costs:CostBook", "MeasuredCosts": "solvi.costs:MeasuredCosts",
    "OrderModel": "solvi.strategist:OrderModel", "ProducerPolicy": "solvi.strategist:ProducerPolicy",
    "Binary": "solvi.strategist:Binary", "scalar_row": "solvi.strategist:scalar_row"})


__all__ = []
