#!/usr/bin/env python
"""Parse a CMake --graphviz dot file and return the static-lib targets
reachable from a given root target, in topological (link) order.

Usage:
    python tools/cmake_graphviz_targets.py <root_label> <dot_file>

Example:
    cmake -S . -B /tmp/bg --graphviz=/tmp/crispasr.dot \
          -DBUILD_SHARED_LIBS=OFF -DCRISPASR_BUILD_TESTS=OFF \
          -DCRISPASR_BUILD_EXAMPLES=OFF -DCRISPASR_BUILD_SERVER=OFF
    python tools/cmake_graphviz_targets.py crispasr-lib /tmp/crispasr.dot

Prints one library name per line (no lib prefix, no .a suffix).
"""

import re
import sys
from collections import defaultdict, deque


def parse_dot(path):
    """Return (static_shape, nodes {id: (label, shape)}, graph {id: [id]})."""
    static_shape = None
    nodes = {}
    graph = defaultdict(list)

    re_legend = re.compile(
        r'\[\s*label\s*=\s*"Static Library"\s*,\s*shape\s*=\s*(\w+)\s*\]'
    )
    re_node = re.compile(
        r'^\s*"(\w+)"\s*\[\s*label\s*=\s*"(\S+)"\s*,\s*shape\s*=\s*(\w+)\s*\]'
    )
    re_edge = re.compile(r'^\s*"(\w+)"\s*->\s*"(\w+)"')

    with open(path) as f:
        for line in f:
            m = re_legend.search(line)
            if m:
                static_shape = m.group(1)
                continue
            m = re_node.match(line)
            if m:
                nid, label, shape = m.group(1), m.group(2), m.group(3)
                # Strip CMake alias annotations like \n(ggml::ggml)
                label = re.sub(r"\\n\([^)]+\)", "", label)
                nodes[nid] = (label, shape)
                continue
            m = re_edge.match(line)
            if m:
                graph[m.group(1)].append(m.group(2))

    return static_shape, nodes, graph


def reachable_from(roots, graph):
    """BFS from root node ids, return set of reachable node ids."""
    visited = set()
    queue = deque(roots)
    while queue:
        n = queue.popleft()
        if n in visited:
            continue
        visited.add(n)
        for child in graph.get(n, []):
            queue.append(child)
    return visited


def topo_sort(node_ids, graph):
    """Kahn's algorithm over the subgraph induced by node_ids."""
    in_degree = defaultdict(int)
    adj = defaultdict(list)
    for n in node_ids:
        if n not in in_degree:
            in_degree[n] = 0
        for child in graph.get(n, []):
            if child in node_ids:
                adj[n].append(child)
                in_degree[child] = in_degree.get(child, 0) + 1

    queue = deque(n for n in node_ids if in_degree[n] == 0)
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for child in adj[n]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def get_static_libs(dot_path, root_labels):
    """Return list of static-lib target names reachable from root_labels,
    in reverse-topological order (leaves first — correct for linker)."""
    static_shape, nodes, graph = parse_dot(dot_path)
    if static_shape is None:
        raise RuntimeError("Could not find Static Library shape in dot legend")

    # Find root node ids by label match
    roots = []
    for nid, (label, _shape) in nodes.items():
        for rl in root_labels:
            if label == rl or label.startswith(rl + "\\n"):
                roots.append(nid)
                break

    if not roots:
        raise RuntimeError(
            f"No nodes found matching labels: {root_labels}. "
            f"Available: {[v[0] for v in nodes.values()]}"
        )

    reach = reachable_from(roots, graph)
    order = topo_sort(reach, graph)

    # Filter to static libs, reverse for link order (leaves first)
    libs = []
    for nid in reversed(order):
        if nid not in nodes:
            continue
        label, shape = nodes[nid]
        if shape == static_shape:
            libs.append(label)
    return libs


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    root_labels = sys.argv[1].split(",")
    dot_path = sys.argv[2]
    libs = get_static_libs(dot_path, root_labels)
    for lib in libs:
        print(lib)


if __name__ == "__main__":
    main()
