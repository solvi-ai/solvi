# Native runtime diagnostic (frozen before execution)

Source546603c. Deterministic native System nodes linked by Process, chains3and10.
Every node maps integer x to x+1 through a registered rule with bounded answer choices.
Five warm requests followed by30profiled requests perchain (no model calls).
Record cProfile cumulative/selftimes and callcounts for run/spec/_go/_step/ask,
alloutputs/status/pathlength and pernode trace/admission parity with directSystem asks.
Validate node/processjournal integrity and request parentchain. Save raw before assertions.
No speed threshold/general performance claim; no originalqualityeval. Inspect hotpaths
before proposing code, preserve fingerprints/checks/replay/journals. No source changes.
