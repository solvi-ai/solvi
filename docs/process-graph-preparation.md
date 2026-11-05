# Preparing large process graphs

Process specification validates reachability, unbounded cycles and merge rules before a run.
These checks build adjacency or incoming-count tables once per pass. Topological preparation
preserves the existing depth-first ordering and the priority of declared edges, while using
explicit traversal frames instead of Python recursion. Long chains do not require changing
Python's recursion limit. Declared bounded loop edges retain their existing meaning.

This improves graph preparation, not arbitrary System execution or end-to-end request latency.
Public specification, canonical hashes, unreachable-node errors and the first reported invalid
merge/cycle retain their contracts. Runtime callbacks and journal trust policies are unchanged.
