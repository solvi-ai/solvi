"""Single-author exact elicitation; no model checkpoint or external service."""
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.preferences import (
    Alternative, PreferenceSession, Requirement, check_regret_certificate,
)


def main():
    budget = WorkBudget(100000)
    session = PreferenceSession(
        [Alternative('fast', (10, 0, 2)), Alternative('quiet', (0, 10, 2)),
         Alternative('balanced', (4, 4, 2))],
        [(1, 0, 0), (0, 1, 0)],
        requirements=[Requirement('safety', 2, 2)], budget=budget,
    )
    before = session.choose()
    print('Before:', before.choice, 'worst-case regret:', before.regret)
    question = session.next_question(cost=1)
    if question is not None:
        print('Question:', question.left, question.right, 'net:', question.net_value)
        # Explicit simulated author prefers the first attribute.
        by_id = {a.id: a for a in session.alternatives}
        preferred, other = sorted((question.left, question.right),
                                  key=lambda x: by_id[x].attributes[0], reverse=True)
        session.answer('author-answer-1', preferred, other, source='simulated-author')
    certificate = session.choose()
    print('After:', certificate.choice, 'worst-case regret:', certificate.regret)
    print('Independently checked:', check_regret_certificate(session, certificate, budget=budget))
    restored = PreferenceSession.restore(session.snapshot(), budget=budget, authority=session.authority)
    print('Replay choice:', restored.choose().choice, 'shared work:', budget.spent)


if __name__ == '__main__':
    main()
