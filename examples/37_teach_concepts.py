"""Teach an exception-bearing finite rule, then compose and retract it."""
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.teaching import TeachingSession


def main():
    # Two possible exception points. The user's answer distinguishes them.
    grammar = Grammar(('x',), constants=(0, 1, 2, 3, 4), max_depth=8, max_nodes=128)
    library = [Program(('and', ('lt', ('input', 'x'), ('const', 4)),
                        ('not', ('eq', ('input', 'x'), ('const', exception)))))
               for exception in (1, 2)]
    budget = WorkBudget(20000)
    teacher = TeachingSession(grammar, library, [{'x': x} for x in range(6)], budget=budget)
    question = teacher.question()
    print('Question:', question.inputs, 'committee:', question.predictions)
    teacher.answer(question, False)  # x=1 is the exception
    small = teacher.admit('small-except-one', '1', [{'x': x} for x in (0, 2, 3, 4, 5)],
                          lambda p: p['x'] < 4 and p['x'] != 1,
                          source_id='separate-reference', source_revision='1')
    print('Accepted:', small)
    large = teacher.compose('complement', '1', Program(('not', ('input', 'learned'))),
                            {'learned': small}, [{'x': x} for x in (0, 2, 3, 4, 5)],
                            lambda p: p['x'] >= 4 or p['x'] == 1,
                            source_id='complement-reference', source_revision='1')
    print('Checked composition:', teacher.apply(large, {'x': 4}))
    print('Unverified point:', teacher.apply(large, {'x': 1}))
    saved = teacher.snapshot()
    restored = TeachingSession.restore(saved, budget=budget, authority=teacher.authority)
    restored.retract(small)
    print('Retracted dependency:', restored.apply(large, {'x': 4}))
    print('Full work ledger:', budget.snapshot())


if __name__ == '__main__':
    main()
