"""
Context matching for mutation placement.

Implements the LOCATE step: finding valid insertion points by matching
k-ancestors and l/r-siblings between donor and recipient.
"""

from __future__ import annotations

from dataclasses import dataclass

from germinator.core.tree import left_sibling, right_sibling, walk_tree


@dataclass
class ContextRequirements:
    """Describes the context a mutation expects."""
    k_ancestor_names: tuple[str, ...]  # Parent, grandparent, etc.
    l_sibling_names: tuple[str, ...]   # Left siblings (nearest first)
    r_sibling_names: tuple[str, ...]   # Right siblings (nearest first)

    @classmethod
    def from_node(cls, node, k: int, l: int, r: int) -> "ContextRequirements":
        """Extract context requirements from a donor node."""
        # k ancestors
        k_names = []
        current = node.parent
        for _ in range(k):
            if current is None:
                break
            k_names.append(current.name)
            current = current.parent

        # l left siblings
        l_names = []
        current = left_sibling(node)
        for _ in range(l):
            if current is None:
                break
            l_names.append(current.name)
            current = left_sibling(current)

        # r right siblings
        r_names = []
        current = right_sibling(node)
        for _ in range(r):
            if current is None:
                break
            r_names.append(current.name)
            current = right_sibling(current)

        return cls(
            k_ancestor_names=tuple(k_names),
            l_sibling_names=tuple(l_names),
            r_sibling_names=tuple(r_names),
        )


class ContextMatcher:
    """
    Matches mutation contexts between donor and recipient trees.

    This implements the LOCATE step from SynthFuzz: finding valid
    insertion points where the surrounding context matches.
    """

    def __init__(
        self,
        k_ancestors: int = 4,
        l_siblings: int = 4,
        r_siblings: int = 4,
        limit_by_donor: bool = True,
    ):
        """
        Args:
            k_ancestors: Number of ancestor levels to match
            l_siblings: Number of left siblings to match
            r_siblings: Number of right siblings to match
            limit_by_donor: If True, only match up to donor's available context
        """
        self.k_ancestors = k_ancestors
        self.l_siblings = l_siblings
        self.r_siblings = r_siblings
        self.limit_by_donor = limit_by_donor

    def matches(self, recipient_node, donor_node) -> bool:
        """Check if recipient location matches donor context."""
        return (
            self._match_ancestors(recipient_node, donor_node)
            and self._match_left_siblings(recipient_node, donor_node)
            and self._match_right_siblings(recipient_node, donor_node)
        )

    def _match_ancestors(self, recipient, donor) -> bool:
        r_node = recipient.parent
        d_node = donor.parent

        for _ in range(self.k_ancestors):
            if self.limit_by_donor and d_node is None:
                break
            if r_node is None or d_node is None:
                return False
            if r_node.name != d_node.name:
                return False
            r_node = r_node.parent
            d_node = d_node.parent

        return True

    def _match_left_siblings(self, recipient, donor) -> bool:
        r_node = left_sibling(recipient)
        d_node = left_sibling(donor)

        for _ in range(self.l_siblings):
            if self.limit_by_donor and d_node is None:
                break
            if r_node is None or d_node is None:
                return False
            if r_node.name != d_node.name:
                return False
            r_node = left_sibling(r_node)
            d_node = left_sibling(d_node)

        return True

    def _match_right_siblings(self, recipient, donor) -> bool:
        r_node = right_sibling(recipient)
        d_node = right_sibling(donor)

        for _ in range(self.r_siblings):
            if self.limit_by_donor and d_node is None:
                break
            if r_node is None or d_node is None:
                return False
            if r_node.name != d_node.name:
                return False
            r_node = right_sibling(r_node)
            d_node = right_sibling(d_node)

        return True

    def find_matching_locations(self, recipient_tree, donor_node) -> list:
        """
        Find all locations in recipient_tree where donor_node's context matches.

        Args:
            recipient_tree: Tree (or nodes_by_name index) to search
            donor_node: Node whose context we're trying to match

        Returns:
            List of recipient nodes that are valid insertion points
        """
        matching = []
        for node in walk_tree(recipient_tree, order="bfs"):
            if node.name == donor_node.name and self.matches(node, donor_node):
                matching.append(node)

        return matching