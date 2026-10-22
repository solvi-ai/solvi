# Stored decisions: format and replay rules

This is the specification of what solvi stores for a decision and what checking it guarantees. It is the contract a
store written today must keep, and the one a port (another language, another backend) must implement. Every guarantee
below is a testable sentence; the tests that pin them are in `tests/semantics/` (named in brackets). The
machine-readable schema of a record is `src/solvi/core/store/record.schema.json`.

## 1. Words

- **record**: one JSON object in a store (a line of a JSONL file, a row's `body` in SQLite / PostgreSQL / DuckDB).
- **trace**: the steps of one decision (`res.trace`): the input, then one **step record** per part that ran.
- **chain**: the records of a store in order, each holding the hash of the one before it.
- **canonical JSON**: `json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- **value hash** (`vhash`): the first 16 hex digits of the SHA-256 of a value's canonical form. Used inside a trace.
- **record hash**: 64 hex digits, SHA-256 of the canonical JSON of the record's body (§3). Used across the chain.

## 2. The record

### Versioning

- `v` is the record format. This version of solvi writes `v: 2` (`solvi.core.store.FORMAT`). Format 1 (solvi 0.7.1 and
  earlier) is read and verified, never written.
- `record.schema.json` describes format 2; its `x-schema-version` is `2.1.0` (2.0.0 plus the guard `no_data`, its
  result's `extra["no_data"]` and a response check's optional `missing` — a minor version: a 2.0.0 reader may ignore
  them, and the hash body does not change; `x-schema-changes` lists each version). Records of kinds `ask`, `teach`,
  `redaction` and `segment` are closed: a new field is a new schema version (minor when readers may ignore it, major
  when the hash body changes). Other kinds (`knowledge`, `dispatch`, `policy`, `refine`, `update`, `change`, `specialist`,
  `process` — the index of one run, or one segment of a run, of `solvi.experimental.process`; `human` — a person's
  answer to one of its questions) are checked for the envelope only.
  [test_sem_record_schema]
- Formats only go up along a chain: a record of format 1 after a record of format 2 is a verify problem. [test_sem_chain]

### The envelope (every kind)

| Field | Content |
|---|---|
| `v` | the format, `2` |
| `kind` | `ask` (a decision), `teach` (a correction), `redaction` (the mark of an erasure), `segment` (a JSONL rotation link), and kinds written by other modules |
| `seq` | position in the chain: 0, 1, 2, ... with no gap |
| `time` | seconds since the epoch, from the store's clock |
| `prev` | the record hash of the record before; `""` for the first record of a file |
| `hash` | this record's hash (§3) |
| `id` | `hash[:16]` |
| `meta` | optional, the caller's own JSON (`save(res, meta=...)`) |

### kind `ask`, stored in full

| Field | Content |
|---|---|
| `init_hash` | value hash of the input |
| `answers` | `{question: [answer, confidence rounded to 4 digits, status]}`; an answer as plain JSON (tuples as lists, "not stated" as `"<not stated>"`, a non-finite float as `{"$float": "inf"}`) |
| `flow` | the names of the planned steps |
| `records` | `[[step, name, step hash]]` of every step record, in order |
| `producers` | only when a fact had alternative producers: `{fact: the producer used}` |
| `guards` | `{question: the safeguard that settled it}` for the questions one settled |
| `safeguards` | `[[kind, fact, [questions]]]` |
| `models` | `[{"type", "id", "fp"}]` of every model that ran, sorted |
| `catalog` | the fingerprint of the catalog that decided |
| `run` | only inside `solvi.core.store.run_context`: `{"id": the run, "parent": the decision it follows from or null}` (schema 2.2.0; also on corrections and every other record kind) |
| `response` | `Response.to_dict()`: results, flow, the whole trace (`init`, step records), values, checks, timings |

The fields before `response` are the record's **summary**: what queries, reports and the SQL index tables read.

### kind `ask`, stored compact

As full, but `record: "compact"` and `compact` in place of `response`: the input, each answer without its
probabilities, each check's `[value or null, reasons]`, the model-backed step records and the answer heads, guarantees,
plan, text and `then` records whole. Not kept: the computed values, the other step records (only their hashes, in
`records`), the planned flow, timings, per-part fingerprints. `record="sample:N"` stores one decision in N in full and
the others compact.

### kind `teach`

`teach` (the question), `init` (the input), `answer`, and when given `source` (`outcome`, `rule`, `verified`; none
means a human), `by`, `of` (the stored id it corrects), `happened` (a note).

### kind `redaction`, kind `segment`

`redaction`: `of` (the id erased), `of_hash` (its hash), `by`, `note`. `segment`: `{"previous": file name, "count",
"hash"}` of the closed JSONL segment this file continues.

### A step record (inside a trace)

`step`, `kind`, `name` (`answer:<question>` for a rule), `inputs` (`{name: value hash}`), `value`, `quote`
(`[start, end, source]` or null), `confidence`, `error`, `prev`, `hash`, and when they apply `producer`, `tried`,
`provenance`, `model`, `probs`, `extra`, `tried_models`. The step hash is the value hash of the step's body: step,
kind, name, inputs, the value's hash, quote, error, prev, and the optional fields present. The first step's `prev` is
`init_hash`.

## 3. The hash chain

- **Record body** (format 2): the lasting fields (`KEPT`: `v kind seq time prev init_hash catalog records flow
  producers models guards safeguards teach source of of_hash by note segment`), plus `#answers` = SHA-256 of the
  canonical JSON of `answers`, plus `#content` = SHA-256 of everything else except `id`, `hash`, `redacted`. A redacted
  record carries those two digests in its mark instead of the content. Format 1: the record without `id` and `hash`.
- `hash = SHA-256(canonical JSON of the body)`; `id = hash[:16]`. The body covers `prev`, so each record commits to the
  whole chain before it.
- The store keeps a **head** `{"count", "hash"}` next to the records (`<file>.head` for JSONL; a table for SQL).
- The order in which a record's keys are written is not part of its hash: a record is written with its dicts in their
  own order (a decider read them in that order), and verify does not depend on it. [test_the_order_keys_are_written_in_is_not_damage]

## 4. What `verify()` guarantees

`store.verify(anchor=None, signature=None, deep=True)` → `{"ok", "count", "head", "legacy", "problems"}`.
`deep=False` skips guarantee 3 (the re-check of each stored response against its summary) and keeps the others; on
5,000 small records it verified once 1.15–1.41 times faster (JSONL and SQLite, full and compact records), on 41 KB records
about 1.15 times — the rest is reading and re-hashing every record, which no mode skips.

Guaranteed (each is a test):

1. Changing any value anywhere in any record, adding a field or removing one, makes `ok` false.
   [test_every_field_of_every_record_edited_alone_fails_verify, test_a_field_added_or_removed_fails_verify]
2. Deleting, inserting, duplicating or swapping records makes `ok` false; so does cutting records off the end while the
   stored head still counts them. [test_deleting_inserting_reordering_or_cutting_records_fails_verify]
3. A full or compact `ask` record whose summary disagrees with its stored response (answers, statuses and the
   confidence in `answers` — the result's confidence rounded to 4 digits —, `init_hash`, step hashes, the trace's own
   links, `guards`, `safeguards`, `models` and, for a full record, `flow`) makes `ok` false: what `query()` and the SQL
   index tables read is what the response holds. A compact record keeps no planned flow; its `flow` is compared with
   the flow planned again when it is re-derived (`replay_all`: kind `integrity` under the same catalog and questions,
   else `flow`). A record whose stored response is malformed is a problem, not an exception.
   [test_hole6_a_records_index_fields_agree_with_its_response]
4. On SQL backends, a row column or an index row (answers, safeguards, models) that differs from the record's JSON makes
   `ok` false. [test_sqlite_body_and_index_edits_fail_verify]
5. `verify(anchor=head)` with a head taken earlier and kept elsewhere makes `ok` false when the records it covers were
   rewritten — even with every hash and the stored head recomputed. [test_a_rewrite_of_the_whole_chain_is_caught_only_by_an_anchor_or_a_signature]
6. `verify(signature=sig)` with a signature taken earlier makes `ok` false on the same rewrite, and when exactly one
   signed record's content changed, names its position (`out["signature"]["index"]`) and its original content hash.

Not guaranteed:

- Without an anchor or a signature, someone who can write the store can rewrite a record and recompute every hash and
  the head: `verify()` says ok. Keep a head (and a signature) where the writer of the store cannot change it.
- The anchor covers the records up to its count; records appended later are not covered.
- A signature locates one changed record. Several changes (a rewrite that moves an erased record's hash also changes
  the redaction record naming it) are detected, not located. A signature is an error-locating code, not a MAC.
- Format-1 records that were redacted cannot be verified: they are listed under `unverified`.

## 5. What replay guarantees

`res.trace.replay(system, flow)` re-executes every step from the recorded input; `store.replay_all(system)` does it for
every stored decision (a compact one is first re-derived from its input, `rederive`).

Guaranteed:

1. The same input to the same System gives the same `init_hash`, the same step hashes and the same answers, whatever
   the order of the input's keys. [test_the_same_input_gives_the_same_hashes_and_answers]
2. A fresh decision, the same decision loaded back from JSON, and every decision of a store (full, compact, sampled)
   replays with `ok` true, `answers == "same"`; replaying twice gives the same result.
   [test_a_decision_replays_ok_fresh_reloaded_and_twice, test_every_stored_decision_replays_and_rederives_to_its_answers]
3. A compact record re-derives to the answers and step hashes of the original decision.
4. Replayed with the System, any field of a result edited after the run — the answer, its status, its confidence, why,
   guard or source; in a response's JSON, or in a store whose every hash was recomputed — is a mismatch of kind
   `answer` and the summary starts with `data damaged`. Confidence is compared within a relative (and, near 0,
   absolute) 1e-9 (`solvi.core.runtime.CONFIDENCE_TOL`): room for float rounding — a product or a calibration computed
   in another order, a model's probabilities on another BLAS — and far below the 4 digits a record keeps. The why of a
   format-1 record is not compared (its dicts were stored with sorted keys, so a dict in a reason reads in another
   order). [test_hole4_*]
5. Replayed against a catalog that changed, the summary says the catalog changed, never `data damaged`.
   [test_a_changed_rule_is_told_from_damaged_data]
6. Mismatch kinds: `integrity`, `answer` (the data was changed); `recompute`, `model_changed`, `missing_part`,
   `missing_input`, `flow` (the code or model changed); `not_restored`, `not_kept`, `error` (no verdict on the data).

Not guaranteed:

- Replayed with a bare Catalog, answers are not checked (`answers == "unchecked"`).
- A result's `probs`, `evidence`, `extra`, `repaired`, `provenance` and `kind` are not compared: edited consistently they
  still replay ok (the step records they were taken from are bound by their hashes).
- A trace rebuilt honestly from a different input is consistent; compare `init_hash` with a receipt kept elsewhere.
- Parts that call external systems must return the same values on replay, or their steps do not recompute.

### Specialist runs (kind `specialist`): the determinism contract

Replay re-executes code and re-renders from recorded model outputs; it does not re-call a model. A model need not
repeat itself (a remote LLM at temperature 0 returns other outputs across identical runs), so a run's record keeps the
proposer's output — `proposal` (the parsed spec) and `proposal_raw` (the proposer's output when it was text — a model's
reply as it came; for a blocked run `proposal` is the raw output and `proposal_error` why it failed).
`Specialist.replay` parses the recorded output again (it must give the recorded spec: "re-parsing the recorded model
output gives another spec now" otherwise), checks it against the source and renders it; the output's sha256 must be the
recorded one. A recorded invalid output must still be invalid. Only a record with no proposer output at all calls the
proposer — `Replay.from_record` is then False and `Replay.note` says so; such a replay is byte-identical only if the
proposer repeats itself. [tests/test_specialist_replay_record.py: a nondeterministic proposer's stored runs replay byte
for byte without a call; `check_specialist` fails a specialist whose replay calls the proposer] On the charts
benchmark's 50 texts with an LLM proposer every stored run replays byte for byte from its record (50 of 50).

## 6. Provenance kinds and statuses

- Every step value has a provenance kind: `given`, `computed`, `quoted`, `decided`, `learned`, `proposed`
  (`solvi.core.provenance.KINDS`); the record stores it only when it differs from the default (`quoted` with a quote,
  else `computed`).
- An answer's status is one of `ok`, `forced` (a failed hard check's `then` set it), `abstain`.
- A check's status in `res.checks` is one of `passed`, `failed`, `skipped`, `error`.
- **No data vs. cannot be checked.** A hard check that governs a question and whose inputs are not given (and cannot be
  computed) is not planned: the question abstains with guard `no_data` (source: the check; `extra["no_data"]`:
  `{check: [the given inputs it misses]}`, walked back through facts that cannot be computed), and the check is listed
  in `res.checks` after the flow's checks with status `skipped`, `missing` naming those inputs. A hard check that ran
  and raised or returned a value that is not a boolean: the question abstains with guard `hard_check`, and the check's
  status is `error`. Both keep the status `abstain` (an answer's status is unchanged, so every reader of `abstain`
  keeps working); the store tells them apart in the record's `guards` (and the SQL index's guard column). A question
  that cannot be computed for any other reason (a rule's input, a soft check) abstains with guard `None`, "cannot
  compute: ...". `no_data` is not a safeguard event (`safeguards` and `system.stats` do not count it): nothing was
  rejected, an input was missing. [test_sem_statuses]

## 7. Hard checks

1. A hard check passes only when it returns `True` (numpy's `True` counts), whether it is untyped or declared `-> bool`.
   Any other value — `False`, `0`, `1`, `None`, `[]`, `""`, a string such as `"yes"` or `"true"` — never lets the
   guarded answer through: `False` (numpy's `False`, `Fail(...)`) forces the `then` answer, a non-boolean rejects the
   step (check status `error`, reason "a check [declared `-> bool`] returns True or False, not ...") and the question
   abstains. The same holds for a guard policy: the call is not allowed. [test_hole2_*]
2. numpy's `bool_` is taken as a bool, typed or not: it is what a comparison of numpy numbers gives, and it has only
   the two values, so reading it is not a coercion (the untyped rule has accepted it since the audit fix). A check
   declared with another return type (`-> Literal[True, False]`, ...) is validated by that type alone.
3. `cat.check(f, hard=True, then=...)` with the function passed directly registers the same part as the decorator form.
   [test_hole3_*]
4. An input key named like a check or a computed fact of the catalog is refused (`ValueError`) — in `ask`, over HTTP
   (422), and as a fact given to the agent guard. Any other extra key, `answer:<question>` included, changes no
   answer. [test_hole1_*]

## 8. Redaction and signature

`store.redact(id, by=, note=, keep_answers=True)` removes a record's content (the response or compact content, the
meta; a correction's input, answer and note), keeps its lasting fields, its hash and its id, writes the mark
`redacted: {"by", "note", "content": digest[, "answers": digest]}`, and appends a `redaction` record naming the erased
id and hash.

1. After a redaction, `verify(anchor, signature)` with an anchor and a signature taken before it says ok.
2. Editing a kept answer of a redacted record makes `verify()` and `verify(anchor, signature)` fail.
3. Turning a record into a "redacted" one with other answers (content dropped, a mark with the real content digest,
   a redaction record, every hash and the head recomputed) is caught by an anchor and a signature taken before, and the
   signature names the record. [test_hole5_*]
4. A redacted record still holding content, a mark without its digests, or a mark without a redaction record that
   names it (or with another who / why) is a verify problem.
5. Not done: copies made before (backups, exports) and what was learned from the record are not touched.

## 9. Decisions (the author's)

Done:

- **Every field of a result is bound to the trace.** Replay compares confidence (within 1e-9), why, guard and source
  besides the answer and its status (§5.4). [test_hole4_a_results_other_fields_are_bound_to_the_trace]
- **A record's summary is checked against its response.** `verify()` compares `guards`, `safeguards`, `models`,
  `flow` and the confidence in `answers` with the stored response (§4.3). [test_hole6_*]
- **"No data" is stored apart from "cannot be checked".** Guard `no_data`, the check `skipped` in `res.checks` with
  its missing inputs; the status stays `abstain` (§6); record schema 2.1.0. [test_sem_statuses]
- **Strict typed checks.** A `-> bool` check passes only on a real boolean, as an untyped one does; numpy's bools count
  (§7). [test_hole2_*]

Decided, not done:

- **Wider value hashes in a format 3.** Step and input hashes are 64 bits today (16 hex digits of SHA-256): enough
  against accidents, short for a port or an adversary. A future record format 3 will hash steps and inputs with
  **BLAKE3**, 256 bits (fast, and one function for every length of value); if BLAKE3 is not acceptable as a dependency,
  with full-length SHA-256 (64 hex digits) instead. Formats 1 and 2 keep their 16-digit hashes and are still read and
  verified; nothing written today changes.
