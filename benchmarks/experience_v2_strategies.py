"""Isolated operational r2 arms. Engineering seeds only; no batch launcher."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from experience_v2 import FAMILIES, Wire, NestedBudget, roles, points, require_engineering_seed
from solvi.experimental.experience import programs
from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.competences import CompetenceRegistry, Candidate, ExternalExample
from solvi.experimental.experience.domain_checks import FiniteDomain
from solvi.experimental.experience.tabular import TabularEnvironment, TabularOracle

ARMS = ('OLD_EMPIRICAL', 'NEW_EMPIRICAL', 'OLD_CHECKED', 'NEW_CHECKED',
        'FIXED_MEASURE', 'FORMULA', 'FORMULA_CHECKED')


def formula(family, names):
    i = lambda n: ('input', names[n])
    if family == 'inventory':
        return ('max', ('const', 0), ('sub', ('sub', i(0), i(1)), i(2)))
    if family == 'eligibility':
        return ('and', ('lt', i(0), i(1)), ('not', i(2)))
    return ('add', ('max', ('add', i(0), i(1)), i(2)), i(3))


class Registry(CompetenceRegistry):
    """One registry contract; pinned OLD search and interpreter remain OLD."""
    def __init__(self, grammar, engine):
        super().__init__(grammar, allowed_sources=('tabular', 'sql:inventory', 'sql:eligibility', 'sql:schedule'))
        self.engine = engine
        self.engine_grammar = engine.Grammar.from_dict(grammar.to_dict())
        self.search = None
        self.trace = None
        self.search_trace = None

    def propose(self, examples, budget):
        self._guard()
        budget.charge('competence_propose')
        rows = self._batch(examples, budget, 'proposal')
        self._remember(rows)
        engine_hash = hashlib.sha256(Path(self.engine.__file__).read_bytes()).hexdigest()
        ledger = budget.parent if hasattr(budget, 'parent') else budget
        trace = dict(engine_sha256=engine_hash, grammar_fingerprint=self.grammar.fingerprint,
                     proposal=[dict(inputs=e.inputs, expected=e.expected, event_id=e.event_id, task_id=e.task_id) for e in rows])
        if self.trace is not None:
            self.trace('search_intent', trace, budget)
        start, counts = ledger.spent, ledger.counts.copy()
        self.search = self.engine.synthesize([(e.inputs, e.expected) for e in rows], self.engine_grammar, budget)
        self.last_search_reason = self.search.reason
        self.search_trace = dict(**trace, start=start, end=ledger.spent, tried=self.search.tried,
                                 reason=self.search.reason,
                                 ast=None if self.search.candidate is None else self.search.candidate.to_dict(),
                                 counts={k: v-counts.get(k, 0) for k, v in ledger.counts.items() if v-counts.get(k, 0)})
        if self.trace is not None:
            self.trace('search_receipt', self.search_trace, budget)
            self.trace('search_terminal', dict(engine_sha256=engine_hash, reason=self.search.reason,
                                               tried=self.search.tried, receipt_digest=digest(self.search_trace)), budget)
        if self.search.candidate is None:
            return None
        program = programs.Program(self.search.candidate.ast)
        first = rows[0]
        candidate = Candidate(program, rows, first.scope, first.mechanism, first.schema, self.grammar.fingerprint)
        self._candidates[candidate.fingerprint] = candidate.to_dict()
        return candidate

    def _evaluate(self, program, inputs, budget):
        # Baseline interpreter has the same strict bounded scalar ABI.
        return self.engine.Program(program.ast).evaluate(inputs, budget, self.engine_grammar)


class EncodedOracle:
    def __init__(self, family, codec):
        self.oracle, self.codec = TabularOracle(family), codec
        self.source_id, self.source_revision = self.oracle.source_id, self.oracle.source_revision
        self.calls = 0

    def read(self, request_id, source_revision, point):
        from solvi.experimental.experience.domain_checks import DomainReceipt
        row = self.codec.decode(point)
        receipt = self.oracle.read(request_id, source_revision, dict(zip(self.codec.names, row)))
        self.calls += 1
        return DomainReceipt(receipt.request_id, receipt.event_id, receipt.source_id,
                             receipt.source_revision, digest(point), receipt.expected)


class Boundary:
    """Messages cross a real encode/serialize/parse/decode boundary before SQL."""
    def __init__(self, environment, codec, log):
        self.environment, self.codec, self.log = environment, codec, log
        self.calls = {'observe': 0, 'act': 0}

    def execute(self, task, kind, answer, attempt, budget, charge_kind):
        wire = self.codec.enabled
        target = 'wt' if wire else 'answer'
        probe = 'w00' if wire else 'measure'
        payload = {target: answer} if kind == 'act' else {'probe': probe, 'field': target}
        public = self.environment.task(task)
        input_message = self.codec.encode([public.inputs[n] for n in self.codec.names])
        message = dict(task=task, attempt=attempt, kind=kind, inputs=input_message, payload=payload)
        raw = json.dumps(message, sort_keys=not wire, separators=(',', ':'))
        self.log('intent', {'message_bytes': raw}, budget)
        budget.charge(charge_kind, 8)
        parsed = json.loads(raw)
        if dict(zip(self.codec.names, self.codec.decode(parsed['inputs']))) != public.inputs:
            raise ValueError('wire input roundtrip mismatch')
        if kind == 'act':
            decoded = {'answer': parsed['payload'][target]}
        else:
            if parsed['payload'] != {'probe': probe, 'field': target}:
                raise ValueError('wire probe mismatch')
            decoded = {'probe': 'measure', 'field': 'answer'}
        receipt = self.environment.execute(parsed['task'], parsed['kind'], decoded, parsed['attempt'])
        self.calls[kind] += 1
        result = receipt.to_dict()
        if kind == 'observe':
            result['value']['field'] = target
        elif wire:
            result['value']['wt'] = result['value'].pop('answer')
        raw_receipt = json.dumps(result, sort_keys=not wire, separators=(',', ':'))
        self.log('receipt', {'message_bytes': raw_receipt}, budget)
        received = json.loads(raw_receipt)
        if kind == 'observe':
            received['value']['field'] = 'answer'
        elif wire:
            received['value']['answer'] = received['value'].pop('wt')
        self.log('terminal', {'attempt': attempt, 'event_id': received['event_id'], 'kind': kind}, budget)
        return received


def run_trial(directory, seed, family, arm, *, old=None, wire=False, limit=200000, approved_phase=None):
    """Run one isolated trial; retain every refusal/error/budget in result.json.

    OLD must be the programs module returned by load_baseline with pinned hashes.
    No final/development seed is accepted by this engineering entry point.
    """
    if approved_phase is None:
        require_engineering_seed(seed)
    else:
        require_engineering_seed(seed, approved_phase=approved_phase)
    if arm not in ARMS or family not in FAMILIES:
        raise ValueError('unknown arm/family')
    if arm.startswith('OLD') and old is None:
        raise ValueError('pinned OLD programs adapter required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    budget = WorkBudget(limit)
    codec = Wire(seed, family, wire)
    groups = roles(seed, family)
    names = tuple('w' + str(i) for i in range(len(codec.names))) if wire else codec.names
    grammar = programs.Grammar(names, constants=(0, 1), operators=FAMILIES[family][2],
                               max_depth=4, max_nodes=7, max_candidates=10000, max_abs=32)
    registry = Registry(grammar, old if arm.startswith('OLD') else programs)
    events = []

    def log(kind, payload, paid=budget):
        paid.charge('journal_event')
        event = {'kind': kind, 'payload': payload, 'spent': budget.spent}
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        events.append(event)

    rows = [(role, index, row) for role, values in groups.items() for index, row in enumerate(values)]
    registry.trace = log
    csv_path = directory / 'tasks.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task_id', 'family', *codec.names])
        writer.writeheader()
        for role, index, row in rows:
            writer.writerow(dict(task_id=codec.identity('task', role + ':' + str(index)),
                                 family=family, **dict(zip(codec.names, [int(v) if type(v) is bool else v for v in row]))))
    try:
        environment = TabularEnvironment(directory / 'environment.db', csv_path)
    except Exception as error:  # noqa: BLE001 - persist infrastructure censor, retaining every assigned task
        # This is scorer-side infrastructure evidence, never a free semantic
        # completion or an exclusion from the preassigned transfer denominator.
        censored = dict(seed=seed, family=family, arm=arm, wire=wire, status='censored',
                        error=type(error).__name__ + ': ' + str(error), spent=budget.spent,
                        counts=budget.counts, external_calls={'observe': 0, 'act': 0}, oracle_calls=0,
                        effects={}, registry=registry.snapshot(), candidate_id=None, proof_id=None,
                        search=None, technical_censor=True, wall_seconds=time.monotonic()-started,
                        transfers=[dict(role=role, task_id=codec.identity('task', role + ':' + str(i)),
                                        status='unexecuted', useful=False)
                                   for role in ('transfer_R', 'transfer_H') for i, _ in enumerate(groups[role])])
        with (directory / 'result.json').open('w') as stream:
            json.dump(censored, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return censored
    boundary = Boundary(environment, codec, log)
    candidate = None
    admission = None
    domain = None
    oracle = EncodedOracle(family, codec)
    result = dict(seed=seed, family=family, arm=arm, wire=wire, status='completed',
                  transfers=[], search=None, candidate_id=None, proof_id=None)
    search_budget = None
    search_start_counts = None

    def label(role, index, row, paid):
        task = codec.identity('task', role + ':' + str(index))
        received = boundary.execute(task, 'observe', None, codec.identity('attempt', role + ':' + str(index)),
                                    paid, role + '_label')
        return ExternalExample(received['event_id'], task, environment.identity, 'v1', family + '-v1',
                               family, codec.encode(row), received['value']['value'], role)

    try:
        if arm not in ('FIXED_MEASURE', 'FORMULA'):
            search_budget = NestedBudget(budget)
            search_start_counts = budget.counts.copy()
            proposal = [label('proposal', i, row, search_budget) for i, row in enumerate(groups['proposal'])]
            if arm == 'FORMULA_CHECKED':
                log('formula_registration_intent', {'grammar_fingerprint': grammar.fingerprint}, search_budget)
                search_budget.charge('competence_propose')
                validated = registry._batch(proposal, search_budget, 'proposal')
                registry._remember(validated)
                candidate = Candidate(programs.Program(formula(family, names)), validated, family,
                                      'v1', family + '-v1', grammar.fingerprint)
                registry._candidates[candidate.fingerprint] = candidate.to_dict()
                log('formula_registration_receipt', {'candidate_id': candidate.fingerprint, 'program': candidate.program.to_dict()}, search_budget)
                log('formula_registration_terminal', {'candidate_id': candidate.fingerprint}, search_budget)
            else:
                candidate = registry.propose(proposal, search_budget)
            result['search'] = dict(spent=search_budget.spent, reason=registry.last_search_reason,
                                    tried=None if registry.search is None else registry.search.tried,
                                    ast=None if candidate is None else candidate.program.to_dict(),
                                    engine_reason=registry.last_search_reason,
                                    counts={k: v-search_start_counts.get(k, 0) for k, v in budget.counts.items()
                                            if v-search_start_counts.get(k, 0)})
            if candidate is None:
                result['status'] = 'budget' if registry.last_search_reason == 'budget_exhausted' else 'unknown'
            else:
                result['candidate_id'] = candidate.fingerprint
                labels = [label('admission', i, row, budget) for i, row in enumerate(groups['admission'])]
                if arm.endswith('CHECKED'):
                    domain = FiniteDomain(family, 'v1', family + '-v1', grammar, oracle.source_id,
                                          oracle.source_revision, (codec.encode(row) for row in points(family)),
                                          budget=budget, input_types={n: 'bool' if canonical == 'expired' else 'int'
                                          for n, canonical in zip(names, codec.names)},
                                          result_type='bool' if family == 'eligibility' else 'int')
                    def domain_event(kind, payload):
                        log(kind, payload)
                        if kind == 'domain_read_receipt':
                            log('terminal', {'kind': 'domain_read', 'request_id': payload['request_id']})
                    admission = registry.admit_checked(candidate, labels, budget, domain=domain, oracle=oracle,
                                                        min_examples=8, on_event=domain_event)
                else:
                    admission = registry.admit(candidate, labels, budget, min_examples=8)
                if not admission.accepted:
                    result['status'] = 'rejected'
                if domain is not None and admission.accepted:
                    result['proof_id'] = registry.snapshot()['checked'][candidate.fingerprint]['registry_proof_id']
        for role in ('transfer_R', 'transfer_H'):
            for index, row in enumerate(groups[role]):
                task = codec.identity('task', role + ':' + str(index))
                entry = dict(role=role, task_id=task, status='abstain', useful=False,
                             candidate_id=None, proof_id=None, application_id=None, receipt=None)
                answer = None
                if arm == 'FIXED_MEASURE':
                    measured = boundary.execute(task, 'observe', None, codec.identity('attempt', task + ':measure'), budget, 'observation')
                    answer = measured['value']['value']
                elif arm == 'FORMULA':
                    answer = programs.Program(formula(family, names)).evaluate(codec.encode(row), budget, grammar)
                elif admission is not None and admission.accepted:
                    app = registry.predict(codec.encode(row), task_id=task, scope=family, mechanism='v1',
                                           schema=family + '-v1', budget=budget,
                                           **({'domain': domain, 'require_checked': True} if domain else {}))
                    if app.status == 'applied':
                        answer = app.value
                        entry.update(candidate_id=candidate.fingerprint, proof_id=result['proof_id'], application_id=app.application_id)
                if answer is not None:
                    receipt = boundary.execute(task, 'act', answer, codec.identity('attempt', task + ':act'), budget, 'external_act')
                    entry.update(status='correct' if receipt['value']['ok'] else 'wrong', receipt=receipt)
                    entry['useful'] = bool(arm == 'NEW_CHECKED' and entry['proof_id'] and entry['application_id'] and receipt['value']['ok'])
                    log('application_receipt', dict(task_id=task, candidate_id=entry['candidate_id'], proof_id=entry['proof_id'],
                                                   application_id=entry['application_id'], event_id=receipt['event_id']))
                    if entry['application_id'] is not None:
                        observation = ExternalExample(receipt['event_id'], receipt['task_id'], receipt['source'],
                                                      receipt['mechanism'], family + '-v1', family,
                                                      codec.encode(row), receipt['value']['answer'], 'evaluation')
                        revoked = registry.observe(entry['application_id'], observation, budget)
                        log('registry_revision', dict(application_id=entry['application_id'],
                                                      event_id=receipt['event_id'], revoked=list(revoked)))
                result['transfers'].append(entry)
    except BudgetStop as error:
        result.update(status='budget', error=str(error))
    except Exception as error:  # noqa: BLE001 - retain semantic/persistence failures as explicit trial errors
        result.update(status='error', error=type(error).__name__ + ': ' + str(error))
    finally:
        if search_budget is not None and result['search'] is None:
            # A paid prefix exists even if the engine or its paid semantic
            # receipt/terminal could not finish. Preserve it before any later
            # recovery/finalization debit, without granting an admission.
            actual = registry.search
            result['search'] = dict(spent=search_budget.spent,
                                    reason='budget_exhausted' if result['status'] == 'budget' else 'error',
                                    engine_reason=None if actual is None else actual.reason,
                                    tried=None if actual is None else actual.tried,
                                    ast=None if actual is None or actual.candidate is None else actual.candidate.to_dict(),
                                    counts={k: v-search_start_counts.get(k, 0) for k, v in budget.counts.items()
                                            if v-search_start_counts.get(k, 0)},
                                    terminal_censored=True)
            technical = dict(kind='search_prefix_censor', status=result['status'], search=result['search'],
                             error=result.get('error'), spent=budget.spent)
            with (directory / 'technical.jsonl').open('a') as stream:
                stream.write(json.dumps(technical, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
    # A successful SQL transaction can precede an unaffordable receipt/revision
    # journal write. Settle the known attempt IDs without executing or purchasing
    # another observation. These technical records never earn useful credit.
    existing = {entry['task_id']: entry for entry in result['transfers']}
    applications = registry.snapshot()['applications']
    for role in ('transfer_R', 'transfer_H'):
        for index, _ in enumerate(groups[role]):
            task = codec.identity('task', role + ':' + str(index))
            if task in existing and existing[task].get('receipt') is not None:
                continue
            recovered = environment.reconcile(codec.identity('attempt', task + ':act'))
            if recovered is None:
                continue
            app_pair = next(((aid, app) for aid, app in applications.items() if app['task_id'] == task), None)
            entry = dict(role=role, task_id=task, status='committed_unlinked', useful=False,
                         receipt=recovered.to_dict(), technical_reconciliation=True,
                         candidate_id=None if app_pair is None else app_pair[1]['competence_id'],
                         application_id=None if app_pair is None else app_pair[0],
                         prediction=None if app_pair is None else app_pair[1]['prediction'],
                         proof_id=result['proof_id'])
            if task in existing:
                result['transfers'].remove(existing[task])
            result['transfers'].append(entry)
    # Assigned but unreached tasks remain explicit denominator failures.
    done = {e['task_id'] for e in result['transfers']}
    for role in ('transfer_R', 'transfer_H'):
        for i, _ in enumerate(groups[role]):
            task = codec.identity('task', role + ':' + str(i))
            if task not in done:
                result['transfers'].append(dict(role=role, task_id=task, status='unexecuted', useful=False))
    try:
        log('trial_terminal', {'status': result['status']})
    except BudgetStop:
        result['terminal_censored'] = True
    result.update(spent=budget.spent, counts=budget.counts, external_calls=boundary.calls,
                  oracle_calls=oracle.calls, effects=environment.state,
                  wall_seconds=time.monotonic()-started, registry=registry.snapshot())
    (directory / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2))
    return result
