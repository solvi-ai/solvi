from solvi import Answer, Catalog, System
from solvi.check import lint
from solvi.core.catalog import Question


def test_hard_check_reading_question_answers_is_lint_error():
    cat = Catalog()
    @cat.rule('form')
    def form(seed):
        return 'classic'
    @cat.rule('tempo')
    def tempo(seed):
        return '120'
    @cat.check(hard=True, then={'tempo': '100'})
    def short_enough(form, tempo) -> bool:
        return False
    system = System(cat, [Question('form', 'Form?', Answer.choice(['classic', 'short'])),
                          Question('tempo', 'Tempo?', Answer.choice(['100', '120']))])
    findings = [f for f in lint(system).findings if f.code == 'question_as_fact']
    assert any(f.where == 'short_enough' and f.level == 'error' for f in findings)
    assert not lint(system).ok


def test_then_reader_warns_but_real_computed_fact_is_allowed():
    cat = Catalog()
    @cat.rule('pick')
    def choose(raw) -> bool:
        return True
    @cat.check(hard=True, then={'pick': lambda pick: not pick})
    def ready(raw) -> bool:
        return bool(raw)
    system = System(cat, [Question('pick', 'Pick?', Answer.yes_no())])
    assert any(f.code == 'question_as_fact' and f.where == 'ready.then[pick]' and f.level == 'error'
               for f in lint(system).findings)
    @cat.fn
    def pick(raw) -> bool:
        return bool(raw)
    assert not any(f.code == 'question_as_fact' for f in lint(system).findings)
