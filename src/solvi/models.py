"""solvi.models — the models a system decides with, by name, and the published deciders in the local cache.

    import solvi.models as models
    m = models.decider("solvi-base")                     # a published decider from the cache (solvi models pull ...)
    m = models.llm("http://127.0.0.1:8080/v1", "qwen2.5-7b-instruct")   # an OpenAI-compatible server
    m = models.systemone("https://...", "my-model")      # a System One service
    part = m.decision("team", "Which team?", "email", ["billing", "shipping"])

`DecideModel` (the decider's class) is here too; the low-level pieces are in solvi.core.deciders. The command
(solvi.cli._models):

    solvi models [list] [--json]                       solvi-ai/solvi-base, solvi-ai/solvi-large and every decider cached
    solvi models pull solvi-ai/solvi-base [--backend onnx|torch|all]      download (huggingface_hub; only this command does)
    solvi models check MODEL [--examples labels.jsonl --task "Which team?" [--options a,b,c]] [--json]

MODEL (here, and `solvi ask --decider`): a local checkpoint folder; a Hugging Face id already in the local cache (nothing
is downloaded outside `pull`); `systemone:URL#model` — a System One service (`--api-key`, or $SOLVI_SYSTEMONE_API_KEY);
`llm:URL#model` — an OpenAI-compatible chat-completions server (solvi.core.deciders.llm), e.g. llm:http://127.0.0.1:8080/v1#qwen2.5-7b
(`--api-key`, or $SOLVI_LLM_API_KEY); or `module:attr` / `file.py:attr` — a DecideModel your code builds.

`check` prints what the checkpoint declares (solvi_decide.json: format, question kinds, act head, questions per pass,
state serialization), its fingerprint, and — with labelled examples (JSON lines or CSV: `label` plus `text` / `input`
or the other columns as a state; the question from --task / --options or each row's "task" / "options") — the latency of
one decision (first call apart) and the accuracy, the share it escalates and the accuracy of what it answers alone.
Exit status: 0 — loaded (and, with --min-accuracy, accurate enough); 1 — cannot be loaded or run here, or below
--min-accuracy; 2 — usage errors (an id that is not downloaded, a bad file)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.deciders import DecideModel


PUBLISHED = {
    "solvi-ai/solvi-base": "typed decisions on a CPU / in ONNX (150M, preview)",
    "solvi-ai/solvi-large": "typed decisions (396M, preview)",
    "solvi-ai/solvi-large-long": "solvi-large for inputs up to 8k tokens (preview)",
}
PULL_PATTERNS = {"onnx": ["*.json", "*.md", "onnx/*"], "torch": ["*.json", "*.md", "*.safetensors"], "all": None}


class ModelError(Exception):
    """A model that cannot be named or found (exit status 2)."""


# --------------------------------------------------------------------------------------------------- the local cache
def cache_dir():
    """The Hugging Face hub cache: $HF_HUB_CACHE, else $HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub, else
    ~/.cache/huggingface/hub (huggingface_hub's own order)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "huggingface" / "hub"


def cached_path(repo_id, revision="main"):
    """The local snapshot folder of a Hugging Face model id (its `revision` ref, else the newest snapshot), or None."""
    d = cache_dir() / ("models--" + repo_id.replace("/", "--"))
    snaps = d / "snapshots"
    if not snaps.is_dir():
        return None
    ref = d / "refs" / revision
    if ref.is_file():
        p = snaps / ref.read_text().strip()
        if p.is_dir():
            return p
    got = sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return got[0] if got else None


def _size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def cached():
    """Every decider in the local cache (a snapshot with solvi_decide.json) → [{"id", "path", "bytes", "onnx", "torch"}]
    (onnx / torch: whether the snapshot holds that backend's weights)."""
    root = cache_dir()
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.glob("models--*")):
        repo = d.name[len("models--"):].replace("--", "/", 1)
        p = cached_path(repo)
        if p is not None and (p / "solvi_decide.json").is_file():
            out.append({"id": repo, "path": str(p), "bytes": _size(p),
                        "onnx": any(p.glob("onnx/*.onnx")), "torch": (p / "model.safetensors").exists()})
    return out


# --------------------------------------------------------------------------------------------------- naming a model
def kind_of(spec):
    """How a MODEL spec is read: "systemone", "llm", "folder", "code" (module:attr / file.py:attr) or "hub" (a Hugging
    Face id)."""
    if spec.startswith("systemone:"):
        return "systemone"
    if spec.startswith("llm:"):
        return "llm"
    if os.path.isdir(spec):
        return "folder"
    mod, _, attr = spec.rpartition(":")
    if mod and attr and "/" not in attr:
        return "code"
    return "hub"


def resolve(spec):
    """A MODEL spec → (kind, where): a folder for "folder" and a cached "hub" id, (url, model) for "systemone" and "llm",
    the spec for "code". A Hugging Face id that is not in the cache raises ModelError (run `solvi models pull ID`); so
    does a spec written as a path ("./x", "../x", "/x", "~/x") that is not a folder — "no such folder", never "pull"."""
    k = kind_of(spec)
    if k in ("systemone", "llm"):
        rest = spec[len(k) + 1:]
        url, sep, model = rest.rpartition("#")
        if not sep or not url or not model:
            eg = ("systemone:http://127.0.0.1:8009#kev-latest" if k == "systemone" else
                  "llm:http://127.0.0.1:8080/v1#qwen2.5-7b")
            raise ModelError(f"{spec!r}: write {k}:URL#model, e.g. {eg}")
        return k, (url, model)
    if k == "folder":
        return k, spec
    if k == "code":
        return k, spec
    if spec.startswith(("./", "../", "/", "~")) or spec in (".", ".."):
        if os.path.isdir(os.path.expanduser(spec)):
            return "folder", os.path.expanduser(spec)
        raise ModelError(f"no such folder: {spec}")
    if spec.count("/") != 1:
        raise ModelError(f"{spec!r}: not a folder, a Hugging Face id (org/name), systemone:URL#model, llm:URL#model or "
                         "module:attr")
    p = cached_path(spec)
    if p is None:
        raise ModelError(f"{spec} is not downloaded: solvi models pull {spec}")
    return k, str(p)


def declaration(path):
    """A checkpoint folder's solvi_decide.json → (meta, parsed capabilities)."""
    from .core.deciders import capabilities
    f = Path(path) / "solvi_decide.json"
    if not f.is_file():
        raise ModelError(f"{path} has no solvi_decide.json: not a solvi decider checkpoint")
    meta = json.loads(f.read_text())
    return meta, capabilities(meta)


def load(spec, backend="auto", api_key=None):
    """A MODEL spec → a DecideModel (see the module docs). Never downloads. ModelError for a name that cannot be read
    (a `file.py:attr` whose file or attribute is missing included — never SystemExit); other exceptions (a missing
    runtime, a broken checkpoint, an error inside your module) pass through."""
    from .core.deciders import DecideModel
    k, where = resolve(spec)
    if k == "systemone":
        from .core.deciders.systemone import systemone
        return systemone(where[0], where[1], api_key=api_key or os.environ.get("SOLVI_SYSTEMONE_API_KEY"))
    if k == "llm":
        from .core.deciders.llm import llm
        return llm(where[0], where[1], api_key=api_key or os.environ.get("SOLVI_LLM_API_KEY"))
    if k == "code":
        from ._loader import LoadError, load_object
        try:
            m = load_object(spec)
        except LoadError as e:
            raise ModelError(f"{spec}: {e}") from None
        if not callable(getattr(m, "decision", None)):
            raise ModelError(f"{spec}: not a decider (a DecideModel, or an object with decision(...))")
        return m
    m = DecideModel.load(where, backend=backend)
    if k == "hub":
        m.model_id = spec                                # the trace names the id, not the cache folder
    return m


# --------------------------------------------------------------------------------------------------- pull
def pull(repo_id, backend="onnx", revision=None):
    """Download a decider into the local cache (huggingface_hub) → its folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ModelError("pulling a model needs huggingface_hub: pip install 'solvi[onnx]' (or pip install "
                         "huggingface_hub)") from None
    return snapshot_download(repo_id, revision=revision, allow_patterns=PULL_PATTERNS[backend])


# --------------------------------------------------------------------------------------------------- providers
SHORT = {"solvi-base": "solvi-ai/solvi-base", "solvi-large": "solvi-ai/solvi-large",
         "solvi-large-long": "solvi-ai/solvi-large-long"}


def decider(spec="solvi-base", *, backend="auto", api_key=None):
    """A decider by name → a DecideModel: "solvi-base" / "solvi-large" / "solvi-large-long" (the published ones, from
    the local cache: `solvi models pull` downloads), a Hugging Face id, a checkpoint folder, "systemone:URL#model",
    "llm:URL#model" or "module:attr" (see the module docs). Never downloads."""
    return load(SHORT.get(spec, spec), backend=backend, api_key=api_key)


def llm(base_url, model, api_key=None, **kw):
    """An OpenAI-compatible chat-completions server as a decider (solvi.core.deciders.llm.llm: every option there)."""
    from .core.deciders.llm import llm as make
    return make(base_url, model, api_key, **kw)


def systemone(base_url, model, api_key=None, **kw):
    """A model behind the System One HTTP API as a decider (solvi.core.deciders.systemone.systemone)."""
    from .core.deciders.systemone import systemone as make
    return make(base_url, model, api_key, **kw)


def __getattr__(name):
    if name == "DecideModel":                       # the decider's class, imported when asked for (torch stays lazy)
        from .core.deciders import DecideModel
        return DecideModel
    if name in ("add_parser", "cmd_models", "measure"):   # the command's code moved to solvi.cli._models in 1.0
        import importlib

        from ._deprecate import _warn_from_caller, moved_message
        _warn_from_caller(moved_message(f"solvi.models.{name}", f"solvi.cli._models.{name}"), skip=1)
        return getattr(importlib.import_module("solvi.cli._models"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cached", "cached_path", "decider", "DecideModel", "kind_of", "llm", "load", "ModelError", "PUBLISHED", "pull",
           "resolve", "systemone"]
