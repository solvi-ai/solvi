"""Paid actors for the frozen finite computation experiment; no comparative runner."""
from decimal import Decimal

from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.computation import ComputeNode, ComputeOperator, ComputationPlan, VerifiedComputation
from solvi.experimental.experience.programs import Program, Grammar
from solvi.experimental.experience.operation_learning import OperationCatalog
from solvi.experimental.experience.competences import CompetenceRegistry, ExternalExample
from solvi.experimental.experience.domain_checks import FiniteDomain, DomainReceipt

try:
    from .compute_tasks import generate, ARMS
except ImportError:
    from compute_tasks import generate, ARMS

ASTS = {k: (k, ("input", "x"), ("input", "y")) for k in ("add", "sub", "mul", "min", "max", "lt", "eq")}
ASTS["blend"] = ("add", ("input", "x"), ("mul", ("input", "y"), ("input", "y")))
GRAMMAR = Grammar(("x", "y"))
AST_NODES = {k: 5 if k == "blend" else 3 for k in ASTS}

def reference(kind, args):
    """Verifier uses Decimal arithmetic, independently of Program/postfix code."""
    x, y = (Decimal(v) for v in args)
    if kind == "blend":
        return int(x + y ** 2)
    if kind == "add":
        return int(x + y)
    if kind == "sub":
        return int(x - y)
    if kind == "mul":
        return int(x * y)
    if kind == "min":
        return int(min(x, y))
    if kind == "max":
        return int(max(x, y))
    if kind == "lt":
        return bool(x < y)
    if kind == "eq":
        return bool(x == y)
    raise ValueError(kind)

def _compiled(kind):
    # Author supplies straight-line formulas; preparation is paid by DIRTY.
    return {"blend": lambda x,y:x+y*y, "add": lambda x,y:x+y,
            "sub":lambda x,y:x-y, "mul":lambda x,y:x*y,
            "min":lambda x,y:min(x,y), "max":lambda x,y:max(x,y),
            "lt":lambda x,y:x<y, "eq":lambda x,y:x==y}[kind]

def trial(seed, family, limit, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    fixture = generate(seed, family)
    events, charges, epochs = [], [], []
    previous = {}
    def sink(snapshot):
        changed = [(k, v - previous.get(k, 0)) for k,v in snapshot["counts"].items() if v != previous.get(k, 0)]
        kind, amount = changed[0]
        charges.append({"kind":kind, "amount":amount, **snapshot})
        previous.update(snapshot["counts"])
    budget = WorkBudget(limit, on_charge=sink)
    phase = "setup"
    programs, operators = {}, {}
    def op(kind, revision="r1", compiled=False):
        program = programs[kind]
        cost, verify_cost = program.nodes + 2, program.nodes + 3
        function = _compiled(kind) if compiled else None
        def execute(args):
            if function is not None:
                return function(*args)
            scratch = WorkBudget(cost)
            value = program.evaluate(dict(zip(("x", "y"), args)), scratch, GRAMMAR)
            if scratch.spent != cost:
                raise RuntimeError("static AST tariff mismatch")
            return value
        def verify(args, proposal):
            expected = reference(kind, args)
            accepted = type(proposal) is type(expected) and proposal == expected
            events.append({"args":list(args), "proposal":proposal, "accepted":accepted,
                           "opname":kind, "kind":kind, "cost":cost, "verify_cost":verify_cost, "phase":phase})
            return accepted
        return ComputeOperator(kind, revision, execute, verify, cost, verify_cost)

    acquisition = {"status":"disabled", "spent":0, "learned_name":None}
    session = None
    cache = {}
    current = {}
    stopped = False
    try:
        # Paid before runtime graph/program construction; fixture layout itself is public host data.
        nodes = fixture["spec"]["nodes"]
        budget.charge("host_construction", 2 + len(nodes) + sum(len(n["deps"]) + AST_NODES[n["kind"]] for n in nodes))
        programs = {k: Program(ASTS[k]) for k in {n["kind"] for n in nodes}}
        if arm == "DIRTY":
            budget.charge("dirty_compile", sum(AST_NODES[n["kind"]] + len(n["deps"]) for n in nodes))
        operators = {k:op(k, compiled=arm == "DIRTY") for k in programs}
        if arm in ("DISPATCH", "NO_CACHE"):
            phase = "training"
            start = budget.spent
            acquisition["status"] = "budget"
            try:
                budget.charge("host_training_construction", 2+1+2+5)
                catalog = OperationCatalog(CompetenceRegistry(GRAMMAR, allowed_sources=("environment","audit","decimal")))
                training = ComputationPlan(("x", "y"), (ComputeNode("accepted", ("x", "y"), ("blend",)),), ("accepted",))
                sessions = []
                for index, (x,y) in enumerate(((0,1),(1,2))):
                    budget.charge("source_read",16)
                    trained = VerifiedComputation(training, [operators["blend"]], budget, inputs={"x":x,"y":y})
                    report = trained.run()
                    if report.status != "complete":
                        raise BudgetStop("training incomplete")
                    keys = catalog.record_session(trained, {"accepted":programs["blend"]}, budget,
                                                  scope="blend",mechanism="arithmetic",schema="xy",task_prefix="train")
                    sessions.append(trained)
                budget.charge("source_read",32)
                rows = [ExternalExample("admit"+str(i),"admit"+str(i),"environment","arithmetic","xy","blend",
                                        {"x":x,"y":y},reference("blend",(x,y)),"admission")
                        for i,(x,y) in enumerate(((2,0),(3,3)))]
                domain = FiniteDomain("blend","arithmetic","xy",GRAMMAR,"decimal","r1",
                                      [{"x":x,"y":y} for x in range(4) for y in range(4)],budget=budget)
                class Oracle:
                    source_id, source_revision = "decimal", "r1"
                    def read(self, request_id, source_revision, point):
                        budget.charge("source_read",16)
                        return DomainReceipt(request_id,"oracle:"+request_id,self.source_id,source_revision,
                                             digest(point),reference("blend",(point["x"],point["y"])))
                admission = catalog.acquire(keys[0],rows,domain,Oracle(),budget)
                if admission.accepted:
                    learned = catalog.operation(keys[0],verify=lambda a,p:type(p) is int and p==reference("blend",a),
                                                verify_revision="decimal-r1",budget=budget,verify_cost=8)
                    raw_execute, raw_verify = learned.execute, learned.verify
                    def learned_verify(args, proposal):
                        accepted = raw_verify(args,proposal)
                        events.append({"args":list(args),"proposal":proposal,"accepted":accepted,"opname":learned.name,
                                       "kind":"blend","cost":learned.cost,"verify_cost":learned.verify_cost,"phase":phase})
                        return accepted
                    learned = ComputeOperator(learned.name,learned.revision,raw_execute,learned_verify,
                                              learned.cost,learned.verify_cost,learned.is_valid)
                    operators[learned.name] = learned
                    acquisition.update(status="acquired",learned_name=learned.name)
                else:
                    acquisition["status"]="rejected"
            except BudgetStop:
                pass
            acquisition["spent"] = budget.spent-start
        active = set(fixture["spec"]["outputs"])
        for n in reversed(nodes):
            if n["name"] in active:
                active.update(n["deps"])
        active_nodes = [n for n in nodes if n["name"] in active]
        plan = ComputationPlan(tuple("abcdef"), tuple(ComputeNode(n["name"],tuple(n["deps"]),
                    (n["kind"], acquisition["learned_name"]) if n["kind"]=="blend" and acquisition["learned_name"] else (n["kind"],))
                    for n in nodes), tuple(fixture["spec"]["outputs"]))
    except BudgetStop:
        stopped = True
    for update in fixture["updates"]:
        epoch = update["epoch"]
        phase = "epoch:"+str(epoch)
        output, status = {}, "budget"
        epoch_cache_valid = False
        if not stopped:
            try:
                changes = fixture["initial_inputs"] if epoch==0 else update["changes"]
                budget.charge("source_read",8*len(changes))
                current.update(changes)
                if update["revision"]:
                    budget.charge("host_revision_validation",AST_NODES["add"]+3)
                    if arm=="DIRTY":
                        budget.charge("dirty_compile",AST_NODES["add"]+2)
                    operators["add"] = op("add","r2",arm=="DIRTY")
                if arm in ("DIRECT","DIRTY"):
                    if arm=="DIRECT":
                        cache={}
                    elif epoch and (changes or update["revision"]):
                        budget.charge("dirty_scan",len(nodes)+sum(len(n["deps"]) for n in nodes))
                        dirty=set(changes)
                        for n in nodes:
                            if set(n["deps"]) & dirty or (update["revision"] and n["kind"]=="add"):
                                dirty.add(n["name"])
                        budget.charge("dirty_invalidation",len(dirty)+2)
                        cache={k:v for k,v in cache.items() if k not in dirty}
                    epoch_cache_valid=True
                    for n in (() if all(k in cache for k in fixture["spec"]["outputs"]) else active_nodes):
                        budget.charge("baseline_lookup",1+len(n["deps"]))
                        if n["name"] not in cache:
                            operator=operators[n["kind"]]
                            budget.charge("baseline_attempt",operator.cost+operator.verify_cost+2)
                            args=tuple(cache[k] if k in cache else current[k] for k in n["deps"])
                            value=operator.execute(args)
                            if operator.verify(args,value):
                                cache[n["name"]]=value
                    status="complete"
                    output={k:cache[k] for k in fixture["spec"]["outputs"] if k in cache}
                else:
                    if session is None:
                        session=VerifiedComputation(plan,list(operators.values()),budget,inputs=current)
                    else:
                        if arm=="NO_CACHE":
                            session.update_inputs(current,"cache-reset:"+str(epoch))
                        elif changes:
                            session.update_inputs(changes,"epoch:"+str(epoch))
                        if update["revision"]:
                            session.update_operators(list(operators.values()))
                    status=session.run({"FIFO":"fifo","CHEAP":"cheap"}.get(arm,"critical")).status
                    output=dict(session.outputs)
            except BudgetStop:
                stopped=True
                if arm in ("DIRECT","DIRTY") and epoch_cache_valid:
                    output={k:cache[k] for k in fixture["spec"]["outputs"] if k in cache}
            if status=="budget":
                stopped=True
        epochs.append({"epoch":epoch,"outputs":output,"status":status,"spent":budget.spent})
    return {"seed":seed,"family":family,"limit":limit,"arm":arm,**fixture,"epochs":epochs,
            "ledger":budget.snapshot(),"callbackevents":events,"charges":charges,"acquisition":acquisition}
