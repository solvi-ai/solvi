"""solvi.solutions — the ready-made systems of the high level, assembled from solvi.core and configured, not
subclassed (each part can be rebuilt from the low level):

    solvi.solutions.decisions   solvi.build(question, examples, ...) → a DecisionSystem: System 1 fitted, its
                                guarantee, a slow path and a dispatcher calibrated, the store wired
    solvi.solutions.guard       solvi.Guard: an agent's tool calls checked before they run

The top-level `solvi` exports their entry points (`solvi.build`, `solvi.Guard`)."""

__all__ = []
