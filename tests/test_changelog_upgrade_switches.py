"""The 1.4.0 upgrade notes name the switches that restore 1.1's behaviour of the experimental modules; each switch is a
real parameter, so the notes cannot point at one that does not exist."""
import inspect
import pathlib
import warnings

LOG = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def _params(f):
    return set(inspect.signature(f).parameters)


def test_every_switch_in_the_upgrade_notes_is_a_parameter():
    text = LOG.read_text()
    top = text.index("## 1.4.0")
    sec = text[text.index("### Upgrading from 1.1", top):text.index("\n### ", text.index("### Upgrading from 1.1", top) + 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.changes import KnownFailures, Pipeline
        from solvi.experimental.laws import LawSet
    switches = {
        'KnownFailures(rollbacks=False)': "rollbacks" in _params(KnownFailures),
        "failed=False": "failed" in _params(Pipeline.rollback),
        "Pipeline(raise_stage_errors=True)": "raise_stage_errors" in _params(Pipeline),
        'LawSet(support_on="matched")': "support_on" in _params(LawSet),
        "assume_one_law=True": "assume_one_law" in _params(LawSet.admit_by_probes),
        "fit_level=None": "fit_level" in _params(LawSet.admit_by_probes),
    }
    for s, ok in switches.items():
        assert s in sec, s
        assert ok, s
