"""
Tree utilities and node traversal helpers.

We wrap Grammarinator's Rule objects but provide a cleaner interface
for our mutation operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


def get_root(node) -> any:
    """Walk up to the root of the tree."""
    while node.parent:
        node = node.parent
    return node


def left_sibling(node) -> any | None:
    """Get the left sibling of a node, or None if it's the first child."""
    if not node.parent:
        return None
    idx = node.parent.children.index(node)
    return node.parent.children[idx - 1] if idx > 0 else None


def right_sibling(node) -> any | None:
    """Get the right sibling of a node, or None if it's the last child."""
    if not node.parent:
        return None
    idx = node.parent.children.index(node)
    return node.parent.children[idx + 1] if idx < len(node.parent.children) - 1 else None


def ancestors(node, max_depth: int | None = None) -> Iterator:
    """Yield ancestors from parent up to root (or up to max_depth levels)."""
    current = node.parent
    depth = 0
    while current:
        if max_depth is not None and depth >= max_depth:
            break
        yield current
        current = current.parent
        depth += 1


def walk_tree(node, order: str = "pre") -> Iterator:
    """
    Walk tree nodes in specified order.

    Args:
        node: Root node to start from
        order: "pre" for pre-order, "post" for post-order, "bfs" for breadth-first
    """
    if order == "pre":
        yield node
        if node.children:
            for child in node.children:
                yield from walk_tree(child, order)
    elif order == "post":
        if node.children:
            for child in node.children:
                yield from walk_tree(child, order)
        yield node
    elif order == "bfs":
        queue = [node]
        while queue:
            current = queue.pop(0)
            yield current
            if current.children:
                queue.extend(current.children)
    else:
        raise ValueError(f"Unknown order: {order}")


def index_by_name(node, exclude_subtree=None) -> dict[str, list]:
    """
    Build an index of nodes by their rule name.

    Args:
        node: Root node to index from
        exclude_subtree: Subtree to exclude from indexing (optional)

    Returns:
        Dict mapping rule names to lists of nodes
    """
    index = {}

    for n in walk_tree(node):
        if exclude_subtree is not None and n is exclude_subtree:
            continue
        if n.name not in index:
            index[n.name] = []
        index[n.name].append(n)

    return index


def find_common_names(tree1_index: dict[str, list], tree2_index: dict[str, list]) -> set[str]:
    """Find rule names that appear in both trees."""
    return set(tree1_index.keys()) & set(tree2_index.keys())


@dataclass
class NodePath:
    """Represents a path from a node to an ancestor."""
    names: tuple[str, ...]  # Rule names from node upward

    @classmethod
    def from_node(cls, node, depth: int) -> "NodePath":
        names = []
        current = node
        for _ in range(depth):
            if current is None:
                break
            names.append(current.name)
            current = current.parent
        return cls(tuple(names))

    def matches(self, other: "NodePath", limit_by_shorter: bool = True) -> bool:
        """Check if two paths match up to the shorter length."""
        if limit_by_shorter:
            length = min(len(self.names), len(other.names))
        else:
            length = max(len(self.names), len(other.names))
            if len(self.names) < length or len(other.names) < length:
                return False
        return self.names[:length] == other.names[:length]