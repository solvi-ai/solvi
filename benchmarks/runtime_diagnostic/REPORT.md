# Audited native runtime diagnostic

Protocol commit e51832d preceded the only diagnostic launch; raw result commit61ab14a.
Source546603c, Python3.11.13, root3GBwrapper, OMP4. Native System.ask with JSONL
node/process journals; System.decide was not used. Fivewarm+30profiled requests each
at3/10nodes; all70knownanswers/status/pathlengths correct, direct System trace hashes
match every finalrequestnode (3/3 and10/10), all node/process journals verify.

For30requests at10nodes: Process.run cumulative .267580499s, _go .257402576s,
300System.ask .216120744s,300System._respond .162s,300store.save .130s,
330store._append .092s. spec30calls/.008845272s (~3.3% of Process.run),
Process._step300calls/.014845623s. At3nodes: Process.run .091780118s,
90System.ask .069635734s, spec30calls/.004643102s, _step90calls/.004282747s.
These are cProfile diagnostics, not independent walltime acceptance or a speed claim.
Harness incremental JSON saves are inside the profiler, outside Process.run, so total
profile time (.315s at10nodes) includes harness overhead; use named-function entries.

The dominant audited cost is node response serialization, canonical journal hashing
and persistence. No evidence supports replacing this with unrecorded decide, dropping
integrity checks, or silently batching commits with weaker durability. spec caching
would target only a small measured fraction and permanent caching risks mutable graph
semantics. Current code has a per-node scan of self.edges in _step; adjacency could
be a later explicitly frozen-run optimization, but graph changes inside callbacks
must be considered before changing behavior. No small source fix is justified yet.
Recommended next narrow investigation: inspect repeated JSON/response conversion in
System._respond→TraceStorage.save, with byte-identical records/trace/replay proofs;
retain admission and journal semantics. Parent approval needed before implementation.
