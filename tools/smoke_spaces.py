"""Smoke test of the public Hugging Face Spaces after a release: open each Space in a headless browser, wait until it has
loaded (Pyodide installs solvi from PyPI in the page, so the first load takes minutes), run one preset and check the output.

    uv run --with playwright python -m playwright install chromium        # once
    uv run --with playwright python tools/smoke_spaces.py                 # every Space
    uv run --with playwright python tools/smoke_spaces.py --only playground,realms --timeout 600 --json
    uv run --with playwright python tools/smoke_spaces.py --only playground --url playground=http://localhost:8000/index.html

What is checked, per Space (the direct *.static.hf.space page, not the huggingface.co frame around it):

- playground: the first preset runs on load and prints "Trace replay: OK"; the Run button runs it again; the "solvi vs
  LLM" tab, when the deployed page has it, decides its second case and then its first one live, reorders the options
  (solvi's answers stay the same) and replays the trace ("replay ok"); the "New in 1.0" tab, when the deployed page has
  it, runs each demo: `solvi.build` + `explain()` (who answered 200 requests), `res.checks` with a `then=` function
  (the forced answer "partial", the trace replays), quotes (a quote typed otherwise accepted on the normalized view, a
  made-up one refused), the compact journal (bytes per decision full vs compact) and the two levels (23 names); the
  "New in 0.8" tab ("New in 0.7" on a page deployed before 0.8), when the deployed page has it, runs its first demo
  (its output, or a note that the solvi it loaded is too old);
- arcade: tic-tac-toe loads; "O (solvi starts)" makes solvi move and explain the move ("solvi plays …"); the "Agent and
  knowledge (1.0)" tab, when the deployed page has it: run 1 in a world, run 2 in the same world (more System 1, fewer
  steps), a new world, a retraction (exact), every decision replays, and protection vs RiskBudget (the gate held);
- realms: a world is generated ("Turn 0 · N factions alive"); "Next turn" advances it;
- pokemon (not published; only a local copy, --url pokemon=...): the "Check and report" tab replays every stored
  decision ("Replayed N of N");
- documents: Python and solvi load in the page ("+ solvi <version>: ready"). The extractor model (790 MB) is not
  downloaded unless --with-model is given; then the first use case is extracted and decided and the trace replays
  ("Replay OK").

The playground and documents also report the solvi version they loaded. A failing Space is visited once more
(--retries), since a cold load from the CDNs now and then stalls. Nothing is uploaded or changed: the test only opens
public pages (or a local copy, --url). Playwright is optional: without it the script prints how to install it and exits 0
(exit 2 with --require). Exit status 1 when a Space fails."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

OWNER = "solvi-ai"
SPACES = ["playground", "arcade", "realms", "documents"]


def space_url(name, owner=OWNER):
    """The page of a static Space, served without the huggingface.co frame (the frame puts the app in a cross-origin
    iframe). The Space page itself is https://huggingface.co/spaces/{owner}/{name}."""
    return f"https://{owner}-{name}.static.hf.space/index.html"


# ---------------------------------------------------------------------------------------------- per-Space checks
def _wait_text(page, pattern, timeout_s):
    """Wait until the page's visible text matches `pattern` (a regex); → the match. Gradio-Lite renders into the page's
    DOM, so the body text is enough."""
    rx = re.compile(pattern)
    deadline = time.monotonic() + timeout_s
    while True:
        text = page.inner_text("body")
        m = rx.search(text)
        if m:
            return m
        if time.monotonic() > deadline:
            tail = " ".join(text.split())[-300:]
            raise TimeoutError(f"no {pattern!r} after {timeout_s:.0f} s; the page ends with: {tail!r}")
        page.wait_for_timeout(1000)


def _gradio_solvi_version(page):
    """The solvi version a Gradio-Lite Space shows on its About tab (playground), if any."""
    m = re.search(r"Running solvi \*?\*?([0-9][\w.+-]*)", page.inner_text("body"))
    return m.group(1) if m else None


def check_playground(page, timeout_s, with_model=False):
    notes = []
    _wait_text(page, r"Trace replay: OK", timeout_s)                     # demo.load runs the first preset
    notes.append("first preset ran on load: Trace replay OK")
    page.get_by_role("button", name="Run", exact=True).first.click()
    _wait_text(page, r"Trace replay: OK", 120)
    notes.append("Run pressed: Trace replay OK")
    notes.append(_check_vs_llm(page))
    notes += _check_new_in_1_0(page)
    # the demo tab: "New in 0.8" (its panel opens with "New in solvi 0.8"); a page deployed before 0.8 calls it "New in 0.7"
    for name, marker in (("New in 0.8", "New in solvi 0.8"), ("New in 0.7", "Features of solvi 0.7")):
        tab = page.get_by_role("tab", name=name)
        if tab.count():
            tab.first.click()
            panel = page.get_by_role("tabpanel").filter(has_text=marker)
            (panel if panel.count() else page).get_by_role("button", name="Run", exact=True).last.click()
            m = _wait_text(page, r"Escalation with a guarantee —|This demo needs solvi [\d.]+ or newer", 120)
            notes.append(f"{name} tab: " + ("act_guard demo ran" if m.group(0).startswith("Escalation")
                                             else m.group(0).rstrip()))
            break
    else:
        notes.append("no 'New in 0.8' (or 'New in 0.7') tab on the deployed page")
    page.get_by_role("tab", name="About").first.click()
    return notes, _gradio_solvi_version(page)


def _tab(page, name):
    """The tab `name` (exact), or None when the deployed page does not have it."""
    tab = page.get_by_role("tab", name=name, exact=True)
    return tab.first if tab.count() else None


def _click_and_wait(page, button, pattern, timeout_s=120):
    """Click the visible button named `button` (exact), wait for `pattern` → (the match, seconds)."""
    t0 = time.monotonic()
    page.get_by_role("button", name=button, exact=True).locator("visible=true").first.click()
    m = _wait_text(page, pattern, timeout_s)
    return m, time.monotonic() - t0


def _check_new_in_1_0(page):
    """The "New in 1.0" tab: each of its demos runs live (each its own sub-tab and button)."""
    tab = _tab(page, "New in 1.0")
    if tab is None:
        return ["no 'New in 1.0' tab on the deployed page"]
    tab.click()
    _wait_text(page, r"New in solvi 1\.0", 60)
    notes = []
    _tab(page, "build + explain").click()
    m, dt = _click_and_wait(page, "Build and ask",
                            r"Then 200 new requests in [\d,]+ ms: System 1 alone (\d+), the slow path (\d+), a person (\d+)|"
                            r"This demo failed[^\n]*|These demos need solvi[^\n]*")
    if not m.group(1):
        raise RuntimeError("New in 1.0 · build: " + m.group(0))
    notes.append(f"New in 1.0 · build + explain ({dt:.1f} s): 200 requests → System 1 {m.group(1)}, slow path "
                 f"{m.group(2)}, person {m.group(3)}")
    _tab(page, "res.checks + then=").click()
    page.get_by_label("over the limit: then computes 'partial'", exact=True).check()
    t0 = time.monotonic()
    _wait_text(page, r"refund = partial \(forced\)[\s\S]*the trace replays: OK", 60)
    notes.append(f"New in 1.0 · res.checks + then= ({time.monotonic() - t0:.1f} s): within_limit failed, then computed "
                 "'partial', the trace replays")
    _tab(page, "Quotes").click()
    m, dt = _click_and_wait(page, "Check the quote", r"needed the normalized view[\s\S]*?abstain")
    page.get_by_label("made up (not in the notes)", exact=True).check()
    _wait_text(page, r"a made-up quote is refused", 60)
    notes.append(f"New in 1.0 · quotes ({dt:.1f} s): typed otherwise → accepted on the normalized view (literal: "
                 "abstain); made up → refused")
    _tab(page, "Compact journal").click()
    m, dt = _click_and_wait(page, "Store them twice", r"Compact: ([\d,]+) bytes a decision vs ([\d,]+) full")
    notes.append(f"New in 1.0 · compact journal ({dt:.1f} s): {m.group(1)} bytes a decision compact vs {m.group(2)} full")
    _tab(page, "Two levels").click()
    m, dt = _click_and_wait(page, "Show what each level exports", r"high level, (\d+) names[\s\S]*?made of, (\d+) names")
    notes.append(f"New in 1.0 · two levels: solvi {m.group(1)} names, solvi.core {m.group(2)} names")
    return notes


def _check_vs_llm(page):
    """The "solvi vs LLM" tab: pick the second case, then the first one (each decided live: "solvi <version> · <case id>
    · N answers in X ms"), reorder the options, replay the trace."""
    tab = page.get_by_role("tab", name="solvi vs LLM")
    if not tab.count():
        return "no 'solvi vs LLM' tab on the deployed page"
    tab.first.click()
    _wait_text(page, r"Side by side", 60)
    cases = page.locator("label").filter(has_text=re.compile(r"^\s*[A-Z]-\d{3} · "))
    cases.nth(1).wait_for(state="visible", timeout=60_000)            # the tab renders when it is first opened
    ids = [cases.nth(i).inner_text().strip().split(" · ")[0] for i in (0, 1)]
    for i in (1, 0):
        cases.nth(i).click()
        m = _wait_text(page, rf"solvi [\w.+-]+ · {re.escape(ids[i])} · (\d+) answers in ([\d.]+) ms", 60)
    page.get_by_role("button", name="Reorder the options", exact=True).first.click()
    _wait_text(page, r"solvi: the same \d+ answers", 60)
    page.get_by_role("button", name="Replay solvi's trace", exact=True).first.click()
    _wait_text(page, r"replay ok", 60)
    return (f"solvi vs LLM tab: {ids[1]} then {ids[0]} decided live ({m.group(1)} answers in {m.group(2)} ms); reordered "
            "options: the same answers; replay ok")


def check_arcade(page, timeout_s, with_model=False):
    _wait_text(page, r"solvi has not moved yet|solvi plays", timeout_s)
    page.get_by_text("O (solvi starts)", exact=True).first.click()
    m = _wait_text(page, r"solvi plays [^\n]+", 120)
    return [f"tic-tac-toe: {m.group(0)[:80]}"] + _check_agent_tab(page), None


RUN = r"(\d+) steps, (\d) of 6 goals — System 1 (\d+), System 2 (\d+)"


def _check_agent_tab(page):
    """The arcade's "Agent and knowledge (1.0)" tab: run 1, run 2 in the same world (more System 1, fewer steps), a new
    world, a retraction, replay, and protection vs justified risk."""
    tab = page.get_by_role("tab", name=re.compile("Agent and knowledge"))
    if not tab.count():
        return ["no 'Agent and knowledge (1.0)' tab on the deployed page"]
    tab.first.click()
    notes = []
    m1, dt1 = _click_and_wait(page, "▶ Run 1 (empty memory)", r"run 1 in world 7 \(empty memory\): " + RUN)
    m2, dt2 = _click_and_wait(page, "▶ Run again, same world", r"run 2 in world 7 \(same memory\): " + RUN)
    s1a, s1b, st_a, st_b = int(m1.group(3)), int(m2.group(3)), int(m1.group(1)), int(m2.group(1))
    if not (s1b > s1a and st_b < st_a):
        raise RuntimeError(f"run 2 did not use its memory: run 1 {m1.group(0)}; run 2 {m2.group(0)}")
    notes.append(f"agent: run 1 ({dt1:.1f} s) {st_a} steps, System 1 {s1a} / System 2 {m1.group(4)}; run 2 ({dt2:.1f} s) "
                 f"{st_b} steps, System 1 {s1b} / System 2 {m2.group(4)}")
    m3, dt3 = _click_and_wait(page, "🗺️ A new world, same memory", r"world 8 \(new world, same memory\): " + RUN)
    rp = _wait_text(page, r"Every decision replays[^\n]*?(\d+) of (\d+)", 30)
    if rp.group(1) != rp.group(2):
        raise RuntimeError("agent: not every decision replays: " + rp.group(0))
    notes.append(f"agent: new world ({dt3:.1f} s) {m3.group(1)} steps, System 1 {m3.group(3)}; every decision replays "
                 f"({rp.group(1)} of {rp.group(2)})")
    m4, dt4 = _click_and_wait(page, "↩ Retract a person's fact", r"took back (\d+) items[\s\S]*?: (exact|NOT exact)")
    if m4.group(2) != "exact":
        raise RuntimeError("agent: the retraction was not exact")
    notes.append(f"agent: retraction ({dt4:.1f} s) took back {m4.group(1)} items, exact")
    m5, dt5 = _click_and_wait(page, "▶ Run both agents", r"(\d+) episodes per agent in [\d.]+ s — the gate held in both: (\w+)",
                              300)
    if m5.group(2) != "yes":
        raise RuntimeError("agent: the gate did not hold")
    iron = re.findall(r"iron in (\d+) of (\d+)", page.inner_text("body"))
    notes.append(f"agent: protection vs RiskBudget ({dt5:.1f} s, {m5.group(1)} episodes each): gate held; "
                 + ", ".join(f"iron {a}/{b}" for a, b in iron[-2:]))
    return notes


def check_realms(page, timeout_s, with_model=False):
    m = _wait_text(page, r"Turn ([\d,]+) · (\d+) factions alive", timeout_s)
    t0 = int(m.group(1).replace(",", ""))
    page.get_by_role("button", name="Next turn", exact=True).first.click()
    deadline = time.monotonic() + 120
    while True:
        m = _wait_text(page, r"Turn ([\d,]+) · (\d+) factions alive", 120)
        t1 = int(m.group(1).replace(",", ""))
        if t1 > t0:
            return [f"world generated (turn {t0}); Next turn → turn {t1}, {m.group(2)} factions alive"], None
        if time.monotonic() > deadline:
            raise TimeoutError(f"'Next turn' did not advance the turn (still {t1})")
        page.wait_for_timeout(1000)


def check_documents(page, timeout_s, with_model=False):
    m = _wait_text(page, r"\+ solvi ([\w.+-]+): ready|Python failed to load", timeout_s)
    if m.group(0).startswith("Python failed"):
        raise RuntimeError("Python failed to load: " + page.inner_text("#py-sub")[:200])
    notes = ["Pyodide + solvi ready"]
    if with_model:
        page.click("#run")
        _wait_text(page, r"Replay OK|Replay: \d+ mismatch", max(timeout_s, 900))
        if "Replay OK" not in page.inner_text("#replay"):
            raise RuntimeError("the trace did not replay: " + page.inner_text("#replay")[:200])
        notes.append("first use case extracted and decided: Replay OK")
    else:
        notes.append("model not loaded (use --with-model to download it and run a use case)")
    return notes, m.group(1)


def check_pokemon(page, timeout_s, with_model=False):
    """The Pokémon showcase (not published; a local copy only, --url pokemon=...): the replay tab replays every stored
    decision. Its output is a textbox, so the check reads the textareas' values."""
    _wait_text(page, r"solvi plays the world map of Pokémon Red", timeout_s)
    page.get_by_role("tab", name="Check and report", exact=True).first.click()
    page.get_by_role("button", name="Replay every stored decision", exact=True).first.click()
    rx = re.compile(r"Replayed (\d+) of (\d+) stored decisions")
    deadline = time.monotonic() + 180
    while True:
        vals = page.locator("textarea").evaluate_all("els => els.map(e => e.value)")
        m = next((rx.search(v) for v in vals if rx.search(v)), None)
        if m:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("no replay result after 180 s")
        page.wait_for_timeout(1000)
    if m.group(1) != m.group(2) or "Mismatches" in " ".join(vals):
        raise RuntimeError("not every stored decision replays: " + m.group(0))
    return [f"replay tab: {m.group(0)}"], None


CHECKS = {"playground": check_playground, "arcade": check_arcade, "realms": check_realms, "documents": check_documents,
          "pokemon": check_pokemon}


# ---------------------------------------------------------------------------------------------- runner
def _attempt(browser, name, url, timeout_s, with_model, screenshots):
    """One visit of one Space in a fresh browser context (no cache from an earlier visit) → a result dict."""
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)[:200]))
    t0 = time.monotonic()
    r = {"space": name, "url": url, "ok": False}
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        notes, version = CHECKS[name](page, timeout_s, with_model)
        r.update(ok=True, notes=notes, solvi=version)
    except Exception as e:  # noqa: BLE001 — every failure is a result, the other Spaces still run
        r["error"] = f"{type(e).__name__}: {e}"[:600]
        if screenshots:
            path = f"{screenshots.rstrip('/')}/{name}.png"
            try:
                page.screenshot(path=path, full_page=True)
                r["screenshot"] = path
            except Exception as se:  # noqa: BLE001
                r["screenshot_error"] = str(se)[:120]
    r["seconds"] = round(time.monotonic() - t0, 1)
    if errors:
        r["page_errors"] = errors[:5]
    ctx.close()
    return r


def run(names, timeout_s=480, headed=False, with_model=False, owner=OWNER, screenshots=None, retries=1, urls=None):
    """Check each Space; a Space that fails is visited again up to `retries` times (a cold load of Pyodide and the
    wheels from the CDNs now and then stalls), and the result says how many attempts it took."""
    from playwright.sync_api import sync_playwright
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        for name in names:
            url = (urls or {}).get(name) or space_url(name, owner)
            failed = []
            for _ in range(retries + 1):
                r = _attempt(browser, name, url, timeout_s, with_model, screenshots)
                if r["ok"]:
                    break
                failed.append(r["error"])
            r["attempts"] = len(failed) + (1 if r["ok"] else 0)
            if failed and r["ok"]:
                r["earlier_errors"] = failed
            results.append(r)
        browser.close()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(SPACES), help=f"comma-separated Spaces (default: {','.join(SPACES)})")
    ap.add_argument("--timeout", type=float, default=480, help="seconds to wait for a Space to load (default 480)")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    ap.add_argument("--with-model", action="store_true", help="documents: download the extractor (790 MB) and run a use case")
    ap.add_argument("--owner", default=OWNER, help=f"Hugging Face owner of the Spaces (default {OWNER})")
    ap.add_argument("--screenshots", metavar="DIR", help="save a full-page screenshot of every Space that fails")
    ap.add_argument("--url", action="append", default=[], metavar="SPACE=URL",
                    help="check a copy served elsewhere, e.g. playground=http://localhost:8000/index.html before deploying")
    ap.add_argument("--retries", type=int, default=1, help="visit a failing Space again this many times (default 1)")
    ap.add_argument("--require", action="store_true", help="exit 2 when Playwright is not installed (default: skip, exit 0)")
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        ap.error(f"unknown Space(s): {', '.join(unknown)}; known: {', '.join(CHECKS)}")
    if "pokemon" in names and not any(u.startswith("pokemon=") for u in a.url):
        ap.error("pokemon is not published: give a local copy with --url pokemon=http://127.0.0.1:PORT/index.html")
    try:
        urls = dict(u.split("=", 1) for u in a.url)
    except ValueError:
        ap.error("--url takes SPACE=URL")
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print("smoke_spaces: Playwright is not installed; skipped. Install it with\n"
              "  uv run --with playwright python -m playwright install chromium\n"
              "and run: uv run --with playwright python tools/smoke_spaces.py", file=sys.stderr)
        return 2 if a.require else 0
    results = run(names, a.timeout, a.headed, a.with_model, a.owner, a.screenshots, a.retries, urls)
    if a.json:
        print(json.dumps(results, indent=1))
    else:
        for r in results:
            head = f"{'OK  ' if r['ok'] else 'FAIL'} {r['space']:<10} {r['seconds']:6.1f} s  {r['url']}"
            if r["attempts"] > 1:
                head += f"  [attempt {r['attempts']}]"
            print(head + (f"  (solvi {r['solvi']})" if r.get("solvi") else ""))
            for n in r.get("notes", []):
                print(f"       {n}")
            if not r["ok"]:
                print(f"       {r['error']}")
            for e in r.get("earlier_errors", []):
                print(f"       an earlier attempt failed: {e[:200]}")
            for e in r.get("page_errors", []):
                print(f"       page error: {e}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
