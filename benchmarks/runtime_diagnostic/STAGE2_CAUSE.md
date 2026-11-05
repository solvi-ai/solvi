# Serialization owner inspection (before code)

The audited path calls Response.to_dict exactly once per TraceStorage.save:
System._respond→storage.save→entry→resp.to_dict. Hash record_body separately
canonicalizes answers, content and final mark, then JSONLStorage._write emits
ordered full JSON. These representations differ by contract; removing a JSON
pass blindly would break integrity/redaction or ordered replay. _jkeys already
has an identity fastpath for native JSON keys. No repeated whole-response dump
was found by direct inspection.

Narrow proposal: record_body builds content using `k not in KEPT` where KEPT is
a21elementtuple for every field. Build a local exclusion set once per call and
use setmembership for contentfield filtering. Keep original tuple iteration for
markfields, exact keyorder/canonicalbytes/hash, redaction semantics and invalid
value errors. No global persistent cache, no changes to customstorage/save calls,
commit durability, response stats or Process. Test against the exact parent oracle,
including mixed dictkeys, nonfinitelegacyJSON, both recordformats, redactions,
full/compact, and actual frozenclock JSONLbytes/replay before any performanceclaim.
No source implementation in this stage pending parent direction.
