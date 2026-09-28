"""The `solvi` command.

    solvi test PATH...        decision regression tests from cases.json files (solvi.testing)
    solvi honesty SET.json    honesty numbers of a labelled set, gated against a baseline (solvi.honesty)
    solvi --version"""
from __future__ import annotations

import sys

COMMANDS = {"test": ("solvi.testing", "decision regression tests from cases.json files"),
            "honesty": ("solvi.honesty", "honesty numbers of a labelled set, gated against a baseline")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "-V"):
        from . import __version__
        print(f"solvi {__version__}")
        return 0
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: solvi {test,honesty} ...  (solvi <command> --help for its options)\n")
        for name, (_, what) in COMMANDS.items():
            print(f"  {name:9s} {what}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    import importlib
    mod = importlib.import_module(COMMANDS[argv[0]][0])
    return mod.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
