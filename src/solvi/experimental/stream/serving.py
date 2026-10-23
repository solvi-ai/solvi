"""How a frozen version is put in front of a System (experimental; the adapter (LoRA) loop, stage 2): the `Served`
protocol, `HeadVersion` (a frozen model served as the question's answer head) and `BaseVersion` (version 0: what the
System had when the loop was built).

    from solvi.experimental.stream.serving import HeadVersion

    srv = HeadVersion(frozen, question="intent", feature="text")
    srv.install(system)                     # → {"from": the head's fingerprint before, "to": this version's}
    srv.uninstall(system)                   # back to what install replaced
    srv.save(dir, version)                  # [dir / f"{version}.pkl"]: the frozen model, pickled
    HeadVersion.load(dir, version, expect_fp=fp, question="intent")   # refused when the fingerprint is not fp

The loop never touches a System except through `Served.install`: a promotion or a rollback installs the version it
serves (version 0 is a `BaseVersion`, which puts back the head the System had, or removes it). `uninstall` is for
detaching (stage 5). A frozen model that already has `install` / `uninstall` (the LoRA version of stage 3) is its own
`Served`; any other one is wrapped in a `HeadVersion` (`solvi.core.versions.serve_head`). Unmeasured."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol


class Served(Protocol):
    def install(self, system) -> dict: ...                       # atomic → {"from", "to"} fingerprints

    def uninstall(self, system) -> None: ...                     # back to what install replaced

    def save(self, dir: Path, version: int) -> list: ...         # the files that identify it

    @classmethod
    def load(cls, dir: Path, version: int, *, expect_fp: str) -> "Served": ...


def _fp(head):
    return None if head is None else head.fingerprint()


class HeadVersion:
    """A frozen model (`stream.learner.Frozen` with `as_head`) served as the answer head of `question`, reading the
    text fact `feature`. It scores and fingerprints as the frozen model does."""

    def __init__(self, frozen, question: str, feature: str = "text"):
        self.frozen, self.question, self.feature = frozen, str(question), feature
        self._replaced = None
        self._installed = False

    def score(self, x):
        return self.frozen.score(x)

    def fingerprint(self) -> str:
        return self.frozen.fingerprint()

    def install(self, system) -> dict:
        from ...core.versions import serve_head
        before = system.heads.get(self.question)
        head = self.frozen.as_head(self.feature)
        serve_head(system, self.question, head)
        self._replaced, self._installed = before, True
        return {"from": _fp(before), "to": head.fingerprint()}

    def uninstall(self, system) -> None:
        if not self._installed:
            return
        if self._replaced is None:
            system.heads.pop(self.question, None)
        else:
            system.heads[self.question] = self._replaced
        self._replaced, self._installed = None, False

    def save(self, dir, version: int) -> list:
        p = Path(dir) / f"{int(version)}.pkl"
        with open(p, "wb") as f:
            pickle.dump(self.frozen, f)
        return [p]

    @classmethod
    def load(cls, dir, version: int, *, expect_fp: str, question: str = "", feature: str = "text") -> "HeadVersion":
        """The version saved by `save`; a ValueError when its fingerprint is not `expect_fp`."""
        with open(Path(dir) / f"{int(version)}.pkl", "rb") as f:
            frozen = pickle.load(f)  # noqa: S301 — the loop's own files (trusted as the calibration file is)
        got = frozen.fingerprint()
        if got != expect_fp:
            raise ValueError(f"version {version}: fingerprint {got[:12]}, recorded {str(expect_fp)[:12]}")
        return cls(frozen, question, feature)

    def __repr__(self):
        return f"HeadVersion({self.question!r}, {self.fingerprint()[:12]})"


class BaseVersion:
    """Version 0 as served on a System: the answer head the System had when the loop was built (None: no head).
    install puts it back (or removes the question's head); it saves nothing."""

    def __init__(self, system, question: str):
        self.question = str(question)
        self.head = None if system is None else system.heads.get(self.question)

    def install(self, system) -> dict:
        before = system.heads.get(self.question)
        if self.head is None:
            system.heads.pop(self.question, None)
        else:
            system.heads[self.question] = self.head
        return {"from": _fp(before), "to": _fp(self.head)}

    def uninstall(self, system) -> None:
        return None

    def save(self, dir, version: int) -> list:
        return []

    @classmethod
    def load(cls, dir, version: int, *, expect_fp: str):
        raise ValueError("version 0 is the System as attached: it is not loaded from a file")


def as_served(frozen, question: str, feature: str = "text"):
    """A frozen model's Served: itself when it has install / uninstall, else a HeadVersion over it."""
    if callable(getattr(frozen, "install", None)) and callable(getattr(frozen, "uninstall", None)):
        return frozen
    return HeadVersion(frozen, question, feature)


__all__ = ["BaseVersion", "HeadVersion", "Served", "as_served"]
