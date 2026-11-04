"""Finite S26 boundary: the event date is declared, never guessed from retract prose."""
import pytest

from solvi import Knowledge, Catalog, Question, Answer, System


@pytest.mark.parametrize('correction_writer', ['label', 'label-corr'])
def test_declared_date_preserves_before_and_changes_at_and_after(tmp_path, correction_writer):
    knowledge = Knowledge()
    catalog = Catalog()

    @catalog.rule('dose')
    def dose(knowledge: dict) -> str:
        return Knowledge.value(knowledge, 'drug', 'dose', 'unknown')

    system = System(catalog, [Question('dose', 'Dose?', Answer.choice(['old', 'new', 'unknown']))],
                    storage=str(tmp_path / 'decisions.jsonl'))
    old = knowledge.tell(('drug', 'dose', 'old'), source='spec', by='label',
                         valid=('2020-01-01', None))
    dates = ['2021-12-31', '2022-01-01', '2022-01-02']
    ids = {}
    for date in dates:
        response = system.ask({'knowledge': knowledge.snapshot(about='drug', valid_at=date), 'date': date})
        ids[date] = response.stored_id
    knowledge.tell(('drug', 'dose', 'new'), source='spec', by=correction_writer,
                   valid=('2022-01-01', None))
    journal = list(knowledge.store.journal)
    result = knowledge.redecide(system.storage, system, affected=[old], valid_at_key='date')
    assert {row['id'] for row in result['answer_changes']} == {ids[date] for date in dates[1:]}
    assert all(row['old']['dose'][0] == 'old' and row['new']['dose'][0] == 'new'
               for row in result['answer_changes'])
    assert Knowledge.value(knowledge.snapshot(about='drug', valid_at=dates[0]), 'drug', 'dose') == 'old'
    assert knowledge.store.journal == journal
    assert len(list(system.storage.iter())) == 3
