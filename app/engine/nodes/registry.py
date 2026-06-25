"""Node type registry — maps a NodeType string to a Node implementation."""
from __future__ import annotations

from app.engine.nodes.base import Node
from app.models.workflow import NodeDef

_REGISTRY: dict[str, type[Node]] = {}


def register(node_cls: type[Node]) -> type[Node]:
    """Class decorator that registers a node implementation by its ``type``."""
    _REGISTRY[node_cls.type] = node_cls
    return node_cls


def get_node(definition: NodeDef) -> Node:
    node_type = definition.type.value
    if node_type not in _REGISTRY:
        raise ValueError(f"No implementation registered for node type {node_type!r}")
    return _REGISTRY[node_type](definition)
