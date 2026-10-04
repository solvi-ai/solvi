"""solvi.solutions — the ready-made systems of the high level, assembled from solvi.core and configured, not
subclassed (each part can be rebuilt from the low level):

    solvi.solutions.decisions   solvi.build(question, examples, ...) → a DecisionSystem: System 1 fitted, its
                                guarantee, a slow path and a dispatcher calibrated, the store wired
    solvi.solutions.guard       solvi.Guard: an agent's tool calls checked before they run
    solvi.solutions.agent       solvi.Agent(env, knowledge=km): an environment agent — System 1 on what the knowledge
                                predicts, System 2 a search, hard checks from the agenda and the action model
    solvi.solutions.knowledge   solvi.Knowledge: the knowledge store, agenda, action model and failure memory behind one
                                object that build, Guard and Agent share

The top-level `solvi` exports their entry points (`solvi.build`, `solvi.Guard`, `solvi.Agent`, `solvi.Knowledge`)."""

__all__ = []
