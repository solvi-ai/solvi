# Process journal reads: JSONL and SQLite

Process already maintains an incremental ticket/run index. `_ticket_seen`, human
answer lookup, run segment lookup and `records()` share that index. It requests
`store.iter(None, after=last_seq)` rather than rescanning records per ticket.

For an owner-managed append-only SQL journal, choose the existing native backend:

```python
from solvi.core.store import SQLiteStorage
from solvi.experimental.process import Process

store = SQLiteStorage("process.db")
process = Process("review", storage=store)
# Declare nodes and the human channel as usual.
```

A new `.db`, `.sqlite` or `.sqlite3` Process storage path selects SQLiteStorage
already. SQLite reads the head from its metadata row and retrieves the new suffix
with `WHERE seq > ? ORDER BY seq`, using the records table's integer primary key.
A finite operation-count test reads zero bodies for five unchanged-index calls
and exactly one newly appended body for ticket/human lookup after a 96-row prefix.
This is an indexed-read mechanism proof, not a timing benchmark or a global SLA.
Initial indexing and explicit `verify()` still read the relevant/full history.

The SQL index assumes append-only ownership of the history. It does not promise
that a cached Process index automatically notices arbitrary administrative SQL
rewrites of old rows. Explicit storage verification detects invalid record/hash
edits; it does not magically authenticate a completely rewritten valid chain
without a separately held anchor. Choose and enforce the storage ownership and
verification policy appropriate to the application.

JSONL retains its existing stronger byte-freshness check for arbitrary in-place
prefix changes. Even an unchanged incremental read validates the full existing
byte prefix; parsed-record and Process caches do not remove that O(file) read.
This protection is deliberately retained. Strict O(new) together with detection
of an arbitrary edit to an unread untrusted prefix needs an external trusted
append-only contract; switching paths or omitting hashes is not a fix.

This recipe selects a backend for a new journal. It does not migrate, rename or
relax the trust policy of an existing JSONL journal, and it changes no defaults.
Human-channel persistence and normal replay/chain-verification contracts remain
separate. Short native human handoff/reopen tests cover both backends and confirm
that explicit verify rejects tampered temporary test journals.
