"""Root help displays registration metadata without hiding command routing."""
import json
import re

import pytest

from solvi.cli import COMMANDS, main


def invoke(capsys, *args):
    try:
        result = main(list(args))
    except SystemExit as error:
        result = error.code
    output = capsys.readouterr()
    return result, output.out, output.err


def command_rows(help_text):
    section = help_text.split('commands:\n', 1)[1].split('\n\n', 1)[0]
    return re.findall(r'^    ([\w-]+)\s{2,}', section, re.M)


def test_root_help_has_six_primary_commands_and_discovery(capsys):
    code, out, err = invoke(capsys, '--help')
    assert code == 0 and not err
    assert command_rows(out) == ['init', 'test', 'ask', 'serve', 'check', 'models']
    assert '--all-help' in out and 'COMMAND --help' in out
    assert '==SUPPRESS' not in out
    assert '{test,honesty' not in out


def test_all_help_uses_actual_registered_commands(capsys):
    code, out, err = invoke(capsys, '--all-help')
    assert code == 0 and not err
    listed = set(command_rows(out))
    assert set(COMMANDS) <= listed
    assert {'init', 'ask', 'serve', 'check', 'models', 'verify', 'replay', 'diff', 'report', 'calibrate', 'migrate'} <= listed
    assert 'check the hash chain across stored decisions' in out


@pytest.mark.parametrize('args', [('verify', '--help'), ('migrate', '--help'), ('hook', '--help'),
                                 ('honesty', '--help'), ('mutants', '--help'), ('test', '--help')])
def test_secondary_and_delegated_command_help_stays_accessible(capsys, args):
    code, out, err = invoke(capsys, *args)
    assert code == 0 and out.startswith('usage:') and '==SUPPRESS' not in out


def test_secondary_command_really_routes_and_primary_scaffold_is_unchanged(tmp_path, capsys):
    # Verify a real empty JSONL store rather than replacing any parser/handler.
    store = tmp_path / 'decisions.jsonl'
    store.write_text('')
    code, out, err = invoke(capsys, 'verify', str(store))
    assert code == 0 and 'chain verified' in out
    project = tmp_path / 'minimal'
    code, out, err = invoke(capsys, 'init', str(project), '--template', 'minimal')
    assert code == 0 and (project / 'catalog.py').is_file()
    assert json.loads((project / 'cases.json').read_text())['task'] == 'catalog.py'


@pytest.mark.parametrize('args', [(), ('--unknown',), ('verify', '--unknown'),
                                 ('--all-help', '--unknown'), ('not-a-command',)])
def test_unknown_arguments_and_missing_command_remain_usage_errors(capsys, args):
    code, out, err = invoke(capsys, *args)
    assert code == 2 and err.startswith('usage:')
