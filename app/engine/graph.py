"""DAG construction and validation using NetworkX (F1)."""
from __future__ import annotations

import networkx as nx

from app.models.workflow import WorkflowDef


class WorkflowGraphError(ValueError):
    """Raised when a workflow definition does not form a valid DAG."""


def build_graph(wf: WorkflowDef) -> nx.DiGraph:
    """Build and validate a directed acyclic graph from a workflow definition."""
    g = nx.DiGraph(name=wf.name)
    for node in wf.nodes:
        g.add_node(node.id, definition=node)
    for idx, edge in enumerate(wf.edges):
        # Parallel edges between the same pair are disallowed in a simple DiGraph;
        # store the edge index so the executor can map back to the EdgeDef.
        g.add_edge(edge.source, edge.target, definition=edge, index=idx)

    if not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        raise WorkflowGraphError(f"Workflow contains a cycle: {cycle}")

    return g


def generations(g: nx.DiGraph) -> list[list[str]]:
    """Topological generations — each inner list is safe to run in parallel (F3)."""
    return [sorted(gen) for gen in nx.topological_generations(g)]
