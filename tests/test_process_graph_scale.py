"""Graph preparation visits edges a bounded number of times; deep chains do not recurse."""
import sys
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (Process, ProcessSpecError, _check_acyclic, _check_graph,
                                            _check_merges, _topological)


class CountedEdges(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.visits = 0

    def __iter__(self):
        for edge in super().__iter__():
            self.visits += 1
            yield edge


def chain(n):
    nodes = {str(i): {"end": i == n - 1} for i in range(n)}
    edges = CountedEdges({"from": str(i), "to": str(i + 1)} for i in range(n - 1))
    return nodes, edges


def test_topological_edge_walk_once_and_exact_legacy_order():
    nodes = dict.fromkeys("abcde")
    edges = CountedEdges([{"from": "a", "to": "b"}, {"from": "a", "to": "c"},
                          {"from": "b", "to": "d"}, {"from": "c", "to": "d"},
                          {"from": "d", "to": "a", "loop": True}])
    assert _topological(nodes, edges, set(nodes)) == list("eacbd")
    assert edges.visits == len(edges)
    assert _topological(nodes, edges, {"b", "d"}) == ["b", "d"]


@pytest.mark.parametrize("start", ["0", None])
def test_valid_chain_graph_validation_visits_each_edge_a_constant_number_of_times(start):
    nodes, edges = chain(64)
    _check_graph(nodes, edges, {}, start)
    assert edges.visits == (2 + (start is None)) * len(edges)


def test_declared_single_input_all_merges_scan_edges_once():
    nodes, edges = chain(64)
    _check_merges(nodes, edges, {n: "all" for n in nodes})
    assert edges.visits == len(edges)


@pytest.mark.parametrize("operation", ["acyclic", "topological", "public_spec"])
def test_chain_beyond_recursion_limit_without_changing_limit(operation):
    limit = sys.getrecursionlimit()
    nodes, edges = chain(limit + 100)
    if operation == "acyclic":
        _check_acyclic(nodes, edges)
    elif operation == "topological":
        assert _topological(nodes, edges, set(nodes)) == list(nodes)
    else:
        process = Process("long", start="n0")
        for n, spec in nodes.items():
            process.node("n" + n, kind="human", question="ok", end=spec["end"])
        for edge in edges:
            process.edge("n" + edge["from"], "n" + edge["to"])
        spec = process.spec()
        assert len(spec["nodes"]) == len(nodes)
    assert sys.getrecursionlimit() == limit


def test_cycle_path_and_first_bad_merge_remain_in_declared_order():
    nodes = dict.fromkeys("abcd")
    with pytest.raises(ProcessSpecError, match="cycle without max_iter: b -> c -> b"):
        _check_acyclic(nodes, [{"from": "a", "to": "b"}, {"from": "b", "to": "c"},
                                {"from": "c", "to": "b"}])
    with pytest.raises(ProcessSpecError, match="node 'd' merges"):
        _check_merges(nodes, [{"from": "a", "to": n} for n in "bdbd"], {"d": "all", "b": "all"})


def test_loop_edges_are_traversed_for_reachability_not_for_acyclicity():
    nodes = {"a": {}, "b": {"end": True}}
    edges = [{"from": "a", "to": "b", "loop": True}]
    _check_graph(nodes, edges, {}, "a")
    with pytest.raises(ProcessSpecError, match="nodes not reachable.*b"):
        _check_graph(nodes, [], {}, "a")


def test_topological_materializes_list_membership_once():
    class Keep(list):
        scans = 0

        def __contains__(self, key):
            self.scans += 1
            return super().__contains__(key)

    nodes, edges = chain(50)
    keep = Keep(nodes)
    assert _topological(nodes, edges, keep) == list(nodes)
    assert keep.scans == 0
