"""S348 finite module tuple/dict literal thresholds, without helper execution."""
import ast

from solvi.testing import mutants


def test_original_tuple_and_dict_thresholds_each_get_two_nudges():
    source = 'LO, HI = 10, 20\nCAPS = {"basic": 5000}\n'
    generated = mutants.generate(source, kinds=('constant',))
    assert len(generated) == 6
    assert {m.description for m in generated} == {'10 → 9', '10 → 11', '20 → 19', '20 → 21', '5000 → 4999', '5000 → 5001'}
    for m in generated:
        namespace = {}
        exec(compile(m.source, '<literal-test>', 'exec'), namespace)
        assert sum([namespace['LO'] != 10, namespace['HI'] != 20, namespace['CAPS']['basic'] != 5000]) == 1


def test_nested_negative_values_not_keys_and_no_opaque_call_execution():
    source = 'CAPS = {42: [-5, {"n": 0.5}], "opaque": explode()}\n'
    generated = mutants.generate(source, kinds=('constant',))
    assert len(generated) == 4
    assert {m.description for m in generated} == {'-5 → -6', '-5 → -4', '0.5 → 0.45', '0.5 → 0.55'}
    for m in generated:
        tree = ast.parse(m.source)
        assert tree.body[0].value.keys[0].value == 42


def test_lowercase_or_mixed_targets_and_function_scope_do_not_expand():
    assert mutants.generate('lo, HI = 10, 20\ncaps = {"basic": 5000}\n', kinds=('constant',)) == []
    source = 'LO, HI = 10, 20\nCAPS = {"basic": 5000}\ndef rule(x):\n return x > 7\n'
    scoped = mutants.generate(source, kinds=('constant',), functions=['rule'])
    assert scoped and all(m.line == 4 for m in scoped)
