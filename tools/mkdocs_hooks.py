"""MkDocs hooks for the solvi documentation site (see mkdocs.yml).

The Markdown files stay written for reading on GitHub; this hook adapts them for the site without touching them:

- README, CHANGELOG, ROADMAP and the examples / gallery indexes (outside docs/) become pages of the site;
- docs/guide.md is split at its `## ` headings into one page per chapter (guide/<chapter>.md), with the headings raised
  one level, and guide/index.md keeps the introduction and the contents. The contents name the part of the docs each
  chapter belongs to (a `**Part** —` line, then its numbered chapters); a nav entry `Title: guide-part:Part` becomes a
  section with that part's chapters, and a chapter in no part fails the build. A link to `guide.md#anchor` (from any page) or
  to `#anchor` (inside the guide) goes to the chapter that has the anchor, and a visit to `guide/#anchor` is forwarded
  there by a small script, so every old anchor keeps working;
- links to repository files that are not pages (examples/*.py, gallery folders, LICENSE, ...) point to GitHub.
"""
from __future__ import annotations

import json
import logging
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File, Files
from pymdownx.slugs import slugify

ROOT = Path(__file__).resolve().parent.parent
GUIDE = "docs/guide.md"
BRANCH = "main"

# repository path -> page of the site, for pages that live outside docs/
EXTERNAL = {"README.md": "index.md", "CHANGELOG.md": "changelog.md", "ROADMAP.md": "roadmap.md",
            "examples/README.md": "examples.md", "gallery/README.md": "gallery.md"}
FOLDERS = {"examples": "examples.md", "gallery": "gallery.md"}

_slug = slugify(case="lower")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
_PART = re.compile(r"^\*\*(.+?)\*\* —")
_ITEM = re.compile(r"^\d+\. \[[^\]]*\]\(#([^)]+)\)")
PART_ENTRY = "guide-part:"

_state: dict = {}
log = logging.getLogger("mkdocs.hooks.solvi")


def _slugify(text: str) -> str:
    return _slug(text, "-")


def _lines_outside_fences(lines):
    fenced = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
            yield i, line, True
            continue
        yield i, line, fenced


def _split_guide() -> dict:
    """Split the guide at its level-2 headings: intro + [(slug, title, lines)], anchor -> chapter slug, and the parts
    (part name -> chapter slugs, from the contents in the intro)."""
    lines = (ROOT / GUIDE).read_text(encoding="utf-8").splitlines()
    intro, chapters, anchors, parts = [], [], {}, {}
    current = part = None
    for _, line, fenced in _lines_outside_fences(lines):
        m = None if fenced else _HEADING.match(line)
        if m and len(m.group(1)) == 2:
            title = m.group(2)
            current = {"slug": _slugify(title), "title": title, "lines": []}
            chapters.append(current)
        if m and current is not None:
            anchors[_slugify(m.group(2))] = current["slug"]
        if current is None:
            intro.append(line)
            if not fenced and _PART.match(line):
                part = _PART.match(line).group(1)
                parts[part] = []
            elif not fenced and part and _ITEM.match(line):
                parts[part].append(_ITEM.match(line).group(1))
        else:
            if m and len(m.group(1)) >= 2:
                line = line[1:]                     # raise each heading one level: the chapter title becomes the page's h1
            current["lines"].append(line)
    return {"intro": intro, "chapters": chapters, "anchors": anchors, "parts": parts}


def _expand_parts(nav, guide, used):
    """Replace each nav entry `Title: guide-part:Part` (at any depth) by a section of that part's chapter pages."""
    out = []
    for item in nav:
        if isinstance(item, dict):
            (title, value), = item.items()
            if isinstance(value, str) and value.startswith(PART_ENTRY):
                name = value[len(PART_ENTRY):]
                if name not in guide["parts"]:
                    raise ValueError(f"mkdocs.yml: {value!r} names no part of the guide's contents")
                used.update(guide["parts"][name])
                item = {title: [f"guide/{slug}.md" for slug in guide["parts"][name]]}
            elif isinstance(value, list):
                item = {title: _expand_parts(value, guide, used)}
        out.append(item)
    return out


def on_config(config, **kwargs):
    guide = _split_guide()
    _state["guide"] = guide
    repo = (config.get("repo_url") or "https://github.com/solvi-ai/solvi").rstrip("/")
    _state["blob"] = f"{repo}/blob/{BRANCH}/"
    _state["tree"] = f"{repo}/tree/{BRANCH}/"
    # each nav entry "Title: guide-part:Part" becomes a section with the part's chapters; every chapter is in one part
    used: set = set()
    config["nav"] = _expand_parts(config.get("nav") or [], guide, used)
    missing = [c["slug"] for c in guide["chapters"] if c["slug"] not in used]
    if missing:
        raise ValueError(f"docs/guide.md: chapters in no part of the nav (list them in the contents): {missing}")
    return config


def on_files(files: Files, config, **kwargs):
    origin = {}                                     # site page -> the repository file it comes from
    for f in list(files):
        if f.src_uri == "guide.md":
            files.remove(f)
        elif f.src_uri.endswith(".md"):
            origin[f.src_uri] = "docs/" + f.src_uri
    for repo_path, page in EXTERNAL.items():
        f = File.generated(config, page, abs_src_path=str(ROOT / repo_path))
        f.edit_uri = None
        files.append(f)
        origin[page] = repo_path
    guide = _state["guide"]
    redirect = json.dumps({a: f"{c}/" for a, c in guide["anchors"].items()})
    index = "\n".join(guide["intro"]) + (
        "\n\n<script>(function(){var m=" + redirect + ";var h=decodeURIComponent(location.hash.slice(1));"
        "if(m[h]){location.replace(m[h]+'#'+h);}})();</script>\n")
    f = File.generated(config, "guide/index.md", content=index)
    f.edit_uri = "guide.md"
    files.append(f)
    origin["guide/index.md"] = GUIDE
    for c in guide["chapters"]:
        f = File.generated(config, f"guide/{c['slug']}.md", content="\n".join(c["lines"]) + "\n")
        f.edit_uri = "guide.md"
        files.append(f)
        origin[f"guide/{c['slug']}.md"] = GUIDE
    _state["origin"] = origin
    _state["pages"] = {repo: page for page, repo in origin.items() if repo != GUIDE}
    return files


def _target(repo_path: str, anchor: str) -> tuple[str | None, str]:
    """A repository path (+ anchor) -> (site page or None, anchor); None means: not a page of the site."""
    if repo_path == GUIDE:
        chapter = _state["guide"]["anchors"].get(anchor)
        return (f"guide/{chapter}.md" if chapter else "guide/index.md"), anchor
    if repo_path in _state["pages"]:
        return _state["pages"][repo_path], anchor
    if repo_path in FOLDERS:
        return FOLDERS[repo_path], anchor
    if repo_path.startswith("docs/") and (ROOT / repo_path).is_file():
        return repo_path[len("docs/"):], anchor         # images and other assets under docs/
    return None, anchor


def _rewrite(url: str, page: str) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", url, re.I) or url.startswith("//"):
        return url
    path, _, anchor = url.partition("#")
    source = _state["origin"][page]
    if path:
        repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    else:
        repo_path = source
    target, anchor = _target(repo_path.rstrip("/"), anchor)
    suffix = f"#{anchor}" if anchor else ""
    if target is None:
        if not (ROOT / repo_path).exists():         # a warning fails `mkdocs build --strict`
            log.warning("%s links to %r (%s), which is not in the repository", source, url, repo_path)
        base = _state["tree"] if (ROOT / repo_path).is_dir() else _state["blob"]
        return base + repo_path.rstrip("/") + suffix
    if target == page:
        return suffix or "#"
    return posixpath.relpath(target, posixpath.dirname(page) or ".") + suffix


def on_page_markdown(markdown: str, page, config, files, **kwargs):
    src = page.file.src_uri
    if src not in _state.get("origin", {}):
        return markdown
    out = []
    for _, line, fenced in _lines_outside_fences(markdown.split("\n")):
        if not fenced:
            line = _LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2), src) + m.group(3), line)
        out.append(line)
    return "\n".join(out)
