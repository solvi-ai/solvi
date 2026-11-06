"""Bounded reuse of whole parameterized Programs from host-verified traces.

ExternalExample is a trusted host assertion, not trace authentication. Two
distinct accepted input bindings discover a template; exhaustive independent
finite-domain admission remains mandatory. This does not invent an algorithm.
The DAG prepays the returned operator's static costs before its callbacks.
Compilation removes recursive AST dispatch, never charges fewer primitives.

Snapshots bind registry evidence; restore recompiles paid pure instructions.
Callables are never serialized; hosts supply versioned live verifiers again.
"""
from __future__ import annotations

from collections.abc import Mapping
from weakref import WeakKeyDictionary

from ._common import canonical, digest, identifier, natural, portable
from .competences import AdmissionRecord, Candidate, CompetenceRegistry, ExternalExample
from .domain_checks import FiniteDomain
from .programs import Program, _primitive, _scalar


class OperationCatalog:
    def __init__(self, registry: CompetenceRegistry, *, max_templates=16, max_samples=32):
        natural(max_templates, "max_templates", positive=True)
        natural(max_samples, "max_samples", positive=True)
        if max_templates > 16 or max_samples > 32 or registry.store is not None:
            raise ValueError("bounded catalog requires an unbound registry")
        self.registry = registry
        self.max_templates, self.max_samples = max_templates, max_samples
        self._templates = {}
        self._session_numbers = WeakKeyDictionary()
        self._next_session_number = 0

    @staticmethod
    def _compile(program, budget):
        instructions = []

        def visit(node):
            budget.charge("operation_compile")
            if node[0] in ("input", "const"):
                instructions.append(node)
            else:
                for child in node[1:]:
                    visit(child)
                instructions.append((node[0], len(node) - 1))

        visit(program.ast)
        return tuple(instructions)

    def snapshot(self, *, budget):
        budget.charge("operation_snapshot", 1 + len(self._templates))
        entries = []
        for key, template in self._templates.items():
            candidate = template["candidate"]
            budget.charge("operation_snapshot", candidate.program.nodes + len(candidate.proposal_examples))
            admission = template.get("admission")
            entries.append(dict(key=key, candidate_id=candidate.candidate_id,
                                admission=None if admission is None else dict(admission.__dict__),
                                domain=None if admission is None else template["domain"].to_dict(budget=budget)))
        body = dict(v=1, grammar_fingerprint=self.registry.grammar.fingerprint,
                    max_templates=self.max_templates, max_samples=self.max_samples,
                    next_session_number=self._next_session_number, templates=entries)
        return {**body, "digest": digest(body)}

    @classmethod
    def restore(cls, snapshot, *, registry, budget):
        """Recompile existing checked evidence; never perform or grant admission.

        Registry restoration validates its proof transcript first. This method
        checks catalog bindings against that registry's public snapshot. No
        registry writes or oracle calls occur. Persist the shared ledger after
        rebuilding, before reopening graph execution.
        """
        budget.charge("operation_restore")
        fields = {"v", "grammar_fingerprint", "max_templates", "max_samples",
                  "next_session_number", "templates", "digest"}
        if type(snapshot) is not dict or set(snapshot) != fields:
            raise ValueError("invalid catalog snapshot schema")
        if (type(snapshot["v"]) is not int or snapshot["v"] != 1 or
                type(snapshot["templates"]) is not list or len(snapshot["templates"]) > 16):
            raise ValueError("invalid catalog snapshot version or bounds")
        budget.charge("operation_restore", len(snapshot["templates"]) + 1)
        data = portable(snapshot)
        checksum = data.pop("digest")
        if checksum != digest(data) or data["grammar_fingerprint"] != registry.grammar.fingerprint:
            raise ValueError("catalog snapshot integrity or grammar mismatch")
        catalog = cls(registry, max_templates=data["max_templates"], max_samples=data["max_samples"])
        natural(data["next_session_number"], "session counter")
        catalog._next_session_number = data["next_session_number"]
        if len(data["templates"]) > catalog.max_templates:
            raise ValueError("template limit")
        budget.charge("operation_restore_registry", 1 + catalog.max_templates * catalog.max_samples)
        evidence = registry.snapshot()
        for entry in data["templates"]:
            budget.charge("operation_restore")
            if type(entry) is not dict or set(entry) != {"key", "candidate_id", "admission", "domain"}:
                raise ValueError("invalid catalog template schema")
            cid, key = entry["candidate_id"], entry["key"]
            if type(cid) is not str or type(key) is not str or cid not in evidence["candidates"]:
                raise ValueError("missing catalog candidate")
            raw = evidence["candidates"][cid]
            budget.charge("operation_restore_candidate", registry.grammar.max_nodes + catalog.max_samples)
            candidate = Candidate.from_dict(raw)
            if (candidate.candidate_id != cid or candidate.program.fingerprint != key or
                    candidate.grammar_fingerprint != registry.grammar.fingerprint or candidate.dependencies or
                    not 1 <= len(candidate.proposal_examples) <= catalog.max_samples or key in catalog._templates):
                raise ValueError("catalog candidate mismatch or sample bound")
            template: dict = {"candidate": candidate}
            admission = entry["admission"]
            if admission is None:
                if entry["domain"] is not None or cid in evidence["admissions"]:
                    raise ValueError("missing catalog admission")
            else:
                if type(admission) is not dict or set(admission) != {"accepted", "reason", "competence_id"}:
                    raise ValueError("invalid catalog admission")
                saved = evidence["admissions"].get(cid)
                proof = evidence["checked"].get(cid)
                if (saved is None or saved["mode"] != "checked" or saved["result"] != admission or
                        proof is None or proof["binding"]["candidate_id"] != cid or
                        len(saved["event_ids"]) + len(candidate.proposal_examples) > catalog.max_samples):
                    raise ValueError("catalog checked admission mismatch")
                domain = FiniteDomain.from_dict(entry["domain"], registry.grammar, budget=budget)
                if proof["binding"]["domain"] != entry["domain"]:
                    raise ValueError("catalog domain binding mismatch")
                if admission["accepted"]:
                    competence = evidence["competences"].get(cid)
                    if (admission["competence_id"] != cid or competence is None or
                            competence["status"] != "active" or proof["result"] is None or
                            proof["result"]["status"] != "verified" or
                            proof["registry_proof_id"] is None or len(candidate.proposal_examples) < 2):
                        raise ValueError("catalog competence not active and verified")
                template.update(admission=AdmissionRecord(**admission), domain=domain,
                                instructions=catalog._compile(candidate.program, budget))
            catalog._templates[key] = template
        return catalog

    def record_session(self, session, node_programs: Mapping[str, Program], budget, *,
                       scope, mechanism, schema, task_prefix):
        """Extract accepted DAG traces, with host-declared Program/callback tags.

        Grammar inputs bind positionally to each node's dependency ports. The
        declared AST is checked against accepted outputs; opaque callbacks are
        not authenticated. Independent domain admission is still mandatory.
        Duplicate bindings with identical typed outputs are paid, then skipped.
        """
        from .computation import VerifiedComputation

        budget.charge("operation_trace")
        if not isinstance(session, VerifiedComputation) or not isinstance(node_programs, Mapping):
            raise ValueError("verified computation and node Program mapping required")
        for label in (scope, mechanism, schema, task_prefix):
            identifier(label)
        plan = session.plan
        budget.charge("operation_trace", len(plan.nodes) + len(plan.input_names) + len(node_programs) + 1)
        nodes = {node.name: node for node in plan.nodes}
        port_count = 0
        for node in plan.nodes:
            budget.charge("operation_trace_metadata")
            port_count += len(node.dependencies)
        if not set(node_programs) <= nodes.keys():
            raise ValueError("unknown trace node")
        for name, program in node_programs.items():
            budget.charge("operation_trace_metadata")
            if not isinstance(program, Program):
                raise ValueError("trace metadata requires Program")
            budget.charge("operation_trace_metadata", program.nodes)
            if len(nodes[name].dependencies) != len(self.registry.grammar.inputs):
                raise ValueError("trace port schema mismatch")
        # Prepay the bounded public views and portable snapshot traversal.
        budget.charge("operation_trace_snapshot",
                      16 * (len(plan.nodes) + len(plan.input_names) + port_count + 1))
        accepted = dict(session.values)
        state = session.snapshot()
        if state["pending"] is not None or state["stopped"]:
            raise ValueError("interrupted computation cannot supply learning traces")
        budget.charge("operation_trace_identity")
        if session not in self._session_numbers:
            self._next_session_number += 1
            self._session_numbers[session] = self._next_session_number
        serial = self._session_numbers[session]
        keys = []
        for name, program in node_programs.items():
            budget.charge("operation_trace")
            if name not in accepted:
                continue
            node = nodes[name]
            bindings = {}
            for parameter, dependency in zip(self.registry.grammar.inputs, node.dependencies):
                budget.charge("operation_trace_port")
                if dependency in accepted:
                    value = accepted[dependency]
                elif dependency in state["inputs"]:
                    value = state["inputs"][dependency]["value"]
                else:
                    raise ValueError("accepted trace lacks a verified dependency")
                bindings[parameter] = value
            expected = accepted[name]
            budget.charge("operation_trace_metadata", program.nodes + len(bindings) + 1)
            key = program.fingerprint
            template = self._templates.get(key)
            duplicate = False
            if template is not None:
                for row in template["candidate"].proposal_examples:
                    budget.charge("operation_trace_compare", len(bindings) + 1)
                    if canonical(row.inputs) == canonical(bindings):
                        if canonical(row.expected) != canonical(expected):
                            raise ValueError("accepted traces disagree at the same binding")
                        if (row.scope, row.mechanism, row.schema) != (scope, mechanism, schema):
                            raise ValueError("trace context mismatch")
                        duplicate = True
                        break
            if not duplicate:
                identity = "trace:" + digest([task_prefix, serial, name, key, bindings, expected])
                example = ExternalExample(identity, identity, "environment", mechanism, schema,
                                          scope, bindings, expected, "proposal")
                key = self.record(program, example, budget)
            if key not in keys:
                keys.append(key)
        return tuple(keys)

    def record(self, program: Program, example, budget):
        """Record a proposal example asserted by the host to be an accepted trace."""
        budget.charge("operation_discovery")
        if not isinstance(program, Program) or not isinstance(example, ExternalExample):
            raise ValueError("Program and external example required")
        key = program.fingerprint
        old = self._templates.get(key)
        if old is None and len(self._templates) >= self.max_templates:
            raise ValueError("template limit")
        if old is not None and old.get("admission") is not None:
            raise ValueError("admission freezes template")
        rows = [] if old is None else list(old["candidate"].proposal_examples)
        if len(rows) >= self.max_samples:
            raise ValueError("sample limit")
        for row in rows:
            budget.charge("operation_discovery", len(self.registry.grammar.inputs) + 1)
            if canonical(row.inputs) == canonical(example.inputs):
                raise ValueError("distinct accepted bindings required")
        candidate = self.registry.propose_program(program, [*rows, example], budget)
        self._templates[key] = {"candidate": candidate}
        return key

    def acquire(self, key, admission_examples, domain, oracle, budget, *, min_examples=2):
        """Compile, then independently check every declared finite-domain point."""
        budget.charge("operation_lookup")
        template = self._templates[key]
        candidate = template["candidate"]
        if template.get("admission") is not None:
            raise ValueError("admission freezes template")
        if len(candidate.proposal_examples) < 2 or not candidate.program.input_names:
            raise ValueError("two distinct parameterized trace examples required")
        if type(domain) is not FiniteDomain:
            raise ValueError("finite domain required")
        rows = []
        for example in admission_examples:
            budget.charge("operation_validation")
            if len(rows) + len(candidate.proposal_examples) >= self.max_samples:
                raise ValueError("sample limit")
            rows.append(example)
        instructions = self._compile(candidate.program, budget)
        admission = self.registry.admit_checked(candidate, rows, budget,
                                                domain=domain, oracle=oracle,
                                                min_examples=min_examples)
        template.update(admission=admission, domain=domain, instructions=tuple(instructions))
        return admission

    def operation(self, key, *, verify, verify_revision, budget, verify_cost=8):
        """Return a domain/revision-bound live operator with an independent verifier.

        The caller declares verifier cost. Direct invocation of callbacks is an
        internal prepaid interface: use ComputeSession to debit execution work.
        Rebuild the operator if unrelated registry admissions increase its size.
        """
        from .computation import ComputeOperator

        budget.charge("operation_lookup")
        natural(verify_cost, "verify_cost", positive=True)
        identifier(verify_revision, "verify_revision")
        if not callable(verify):
            raise ValueError("independent verifier required")
        template = self._templates[key]
        admission = template.get("admission")
        if admission is None or not admission.accepted:
            raise ValueError("operation not independently admitted")
        cid = admission.competence_id
        domain = template["domain"]
        grammar = self.registry.grammar
        candidate = template["candidate"]
        instructions = template["instructions"]
        registry_size = len(self.registry._competences)
        budget.charge("operation_compile", len(domain.points) * (len(grammar.inputs) + 1))
        points = frozenset(canonical(list(row)) for row in domain.points)
        revision = digest([candidate.fingerprint, domain.fingerprint, "postfix-v1", verify_revision])

        def is_valid():
            try:
                self.registry._guard()
            except RuntimeError:
                return False
            record = self.registry._competences.get(cid)
            return (self.registry.grammar.fingerprint == grammar.fingerprint and
                    len(self.registry._competences) <= registry_size and
                    record is not None and record["status"] == "active")

        def guard(values):
            if not is_valid():
                raise ValueError("operation revoked or registry changed; rebuild required")
            if type(values) is not tuple or len(values) != len(grammar.inputs):
                raise ValueError("operation input schema mismatch")
            for value, kind in zip(values, domain.input_types):
                if type(value) is not (int if kind == "int" else bool):
                    raise ValueError("operation input type mismatch")
                _scalar(value, grammar.max_abs)
            if canonical(list(values)) not in points:
                raise ValueError("operation input outside admitted domain")

        def execute(values):
            guard(values)
            bindings = dict(zip(grammar.inputs, values))
            stack = []
            for op, arg in instructions:
                if op == "input":
                    stack.append(bindings[arg])
                elif op == "const":
                    stack.append(arg)
                else:
                    args = stack[-arg:]
                    del stack[-arg:]
                    stack.append(_primitive(op, args, grammar.max_abs))
            return stack[0]

        def checked(values, proposal):
            guard(values)
            result = verify(values, proposal)
            if type(result) is not bool:
                raise ValueError("live verifier must return strict bool")
            if result is False:
                self.registry.invalidate(cid, "live_counterexample", "live:" + revision)
            if result:
                guard(values)
            return result

        # Same input and AST costs as Program.evaluate, plus paid live guards.
        # Quadratic reservation covers registry invalidation's descendant scan.
        guard_cost = 3 + len(grammar.inputs) + len(points)
        return ComputeOperator(name="learned:" + key, revision=revision, execute=execute,
                               verify=checked, cost=candidate.program.nodes + len(grammar.inputs) + guard_cost,
                               verify_cost=verify_cost + 2 * guard_cost + registry_size * registry_size + 1,
                               is_valid=is_valid)
