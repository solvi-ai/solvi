# Opt-in LawSet consolidation

`LawSet.consolidate(candidate, buffer, exclude=None, commit=True)` addresses a
patchwork of at least two strictly narrower laws with the same action and effect.
It returns the replacement law, or `None`. Ordinary `test()` still requires
positive net error reduction against the original active book.

For example, a book containing `leap & on -> yes` and `leap & off -> yes`
already answers correctly on 60 held-out leap rows. `leap -> yes` has original
net zero, so `test()` refuses it. Explicit consolidation may replace the two
laws after testing the general candidate on all 60 matched rows. It then also
answers `yes` for `leap & mid`, which the old book did not cover.

This is empirical compaction, not additional evidence of prediction improvement
and not a guarantee for unseen inputs. The caller opts into extending the answer
to the candidate's declared condition. If that extension requires a domain
guarantee, the caller must establish it separately; `confirm(candidate)` can
enforce an application validator. Observed matching counterexamples are included
even if an existing exception would hide them under `support_on="applied"`.
In particular, a candidate with condition `()` must pass on every held-out row of
its action. No claim is made about historical multi-seed quality results.

The transaction uses a private trial book with the proposed subsumed laws removed.
It forces matched support for this candidate and uses ordinary `test()` gates:
grammar, readmission, cap, positive counterfactual net, minimum support,
precision, sized binomial bound/null, rivals, confirmation and the post-admission
check. Existing leftover-probe falsification is also enforced. Buffer candidate
accounting retains the same candidate identity and multiplicity budget; a
committed attempted test spends its candidate in `Buffer.tested` even if refused.
A full book can compact when removal makes room; the resulting book obeys cap.

Before committing, every previously correct allowed held-out row must remain
correct, the replacement must remain active, and every other active law must
survive. This prevents an overlapping contrary rule from silently making the
replacement lose correct predictions when priorities change. Rejection or a
confirmation exception changes no active or retired laws. Callbacks have their
usual external side effects; those cannot be rolled back. Failed committed
attempts may update test accounting, as ordinary attempted admission does.

On success the same original Law objects become retired with reason
`subsumed by <replacement-id>`. The replacement's `admitted` is
`"consolidation"`; its `consolidation` dictionary records the replaced IDs,
held-out source IDs, original net, counterfactual net, old/new correct counts,
excluded group and `domain_guarantee=False`. Its ordinary `net` field is the
counterfactual net from the trial book, not error reduction against the original
book. Original source IDs are not recycled as new independent evidence. Normal
independent observation, checking, suspension/retirement and pickle persistence
continue to apply to the replacement.

`commit=False` performs the same evidence checks on a private buffer, returns a
`status="proposed"` law, skips the external confirmation callback, and changes
neither the original book nor its candidate budget. It can be replayed to inspect
the proof before committing. A proposed law is conditional on later confirmation.

The initial API intentionally refuses compaction when the candidate has no
positive net even against the book with the subsumed laws removed (for example,
an already-correct base). It does not implement arbitrary semantic implication,
effect equivalence, automatic consolidation during `step()`, or a statistical
comparison proving that a broader rule is better than its patchwork.

Deterministic regression coverage is in `tests/test_laws_consolidation.py`:
the original leap/on/off witness under both support modes, unseen mid behavior,
contrary matched rows and defaults, retained overlaps, sized/support/net gates,
readmit/falsification and exclusion, cap, confirmation rejection/exception,
read-only replay, pickle and independent falsifying observations.
