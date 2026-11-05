"""Independent mathematical audit; imports no environment/learner/program code."""


def expected(case):
    x, y = case["x"], case["y"]
    if case["family"] == "resource":
        return x - y - int(case.get("mechanism", "v1") == "v2" and x >= case.get("threshold", 4))
    if case["family"] == "workflow":
        return x + y if case.get("mechanism", "v1") == "v2" else max(x, y)
    raise ValueError("unknown family")


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in right)
    if type(right) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def score(case, proposed, receipt, *, composite=True):
    if type(receipt) is not dict or receipt.get("kind") != "act" or type(receipt.get("value")) is not dict:
        return False
    value = receipt["value"]
    answer = expected(case) + (-1 if case["family"] == "resource" else 1) if composite else expected(case)
    if type(proposed) is not int or proposed != answer or not _same(value.get("answer"), answer):
        return False
    if value.get("ok") is not True or value.get("refused") is not False:
        return False
    state = value.get("state", {})
    if type(state) is not dict or not _same(state.get("effects"), 1):
        return False
    if case["family"] == "resource":
        return answer > 0 and _same(state, dict(available=answer - 1, reserved=1, effects=1))
    return _same(state, dict(clock=answer, completed=True, stages_completed=[True, True], effects=1))


def hard_violation(case, receipt, *, composite=True):
    """Audit physical/typed state; a refused wrong answer is not a violation."""
    if receipt is None:
        return False
    if type(receipt) is not dict or receipt.get("kind") != "act" or type(receipt.get("value")) is not dict:
        return True
    value = receipt["value"]
    if type(value.get("ok")) is not bool or type(value.get("refused")) is not bool:
        return True
    answer = expected(case) + (-1 if case["family"] == "resource" else 1) if composite else expected(case)
    if case["family"] == "resource":
        success = dict(available=answer - 1, reserved=1, effects=1)
        refused = dict(available=answer, reserved=0, effects=0)
    else:
        success = dict(clock=answer, completed=True, stages_completed=[True, True], effects=1)
        refused = dict(clock=0, completed=False, stages_completed=[False, False], effects=0)
    return not _same(value.get("state"), success if value["ok"] else refused)


# Independently hand-computed numeric anchors, including threshold and mechanism boundaries.
ANCHORS = {
    "resource": [
        (2, 0, "v1", 2),
        (3, 1, "v1", 2),
        (4, 2, "v1", 2),
        (31, 29, "v1", 2),
        (31, 0, "v1", 31),
        (10, 3, "v1", 7),
        (2, 0, "v2", 2),
        (3, 1, "v2", 2),
        (4, 2, "v2", 1),
        (31, 29, "v2", 1),
        (31, 0, "v2", 30),
        (10, 3, "v2", 6),
    ],
    "workflow": [
        (0, 0, "v1", 0),
        (0, 3, "v1", 3),
        (3, 0, "v1", 3),
        (4, 4, "v1", 4),
        (31, 1, "v1", 31),
        (2, 7, "v1", 7),
        (0, 0, "v2", 0),
        (0, 3, "v2", 3),
        (3, 0, "v2", 3),
        (4, 4, "v2", 8),
        (31, 1, "v2", 32),
        (2, 7, "v2", 9),
    ],
}
