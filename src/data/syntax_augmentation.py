"""
syntax_augmentation.py — Syntax-aware contrastive augmentations (Axe A).

Instead of SimCSE's "same sentence, different dropout" paradigm, these
augmentations produce structurally different but semantically equivalent
sentence views using the dependency parse tree.

Inspired by SynCSE (Zhang et al., 2023): syntactic structure guides the
construction of positive pairs, forcing BERT to encode structure rather
than just surface patterns.

Three strategies:
  1. subtree_crop   — extract a random subtree as a positive view
  2. dep_reorder    — reorder tokens by dependency depth
  3. leaf_deletion  — remove random leaf (modifier) nodes

Each function takes a parse_result dict (from StanzaSyntaxParser) with keys:
  tokens, edges_src, edges_dst, num_tokens
and returns a string (the augmented sentence).
"""

from __future__ import annotations

import random
from typing import Any


def _build_children_map(parse_result: dict[str, Any]) -> dict[int, list[int]]:
    """Build a parent→children adjacency list from directed dependency edges.

    The parse stores edges as parent→child (edges_src[i] → edges_dst[i]).
    If reverse edges are present (child→parent), we skip them to avoid
    double-counting.

    Returns:
        Dict mapping each node index to its list of child indices.
    """
    num_tokens = parse_result.get("num_tokens", len(parse_result.get("tokens", [])))
    children: dict[int, list[int]] = {i: [] for i in range(num_tokens)}

    edges_src = parse_result.get("edges_src", [])
    edges_dst = parse_result.get("edges_dst", [])

    for src, dst in zip(edges_src, edges_dst):
        # In our parse format, directed edge is parent→child.
        # If add_reverse_edges=True, there are also child→parent edges.
        # We only want parent→child edges for the tree structure.
        # Heuristic: in a dependency tree, a parent appears as src for its children.
        # To avoid double-counting, we track which (src, dst) pairs we've seen.
        if 0 <= src < num_tokens and 0 <= dst < num_tokens and src != dst:
            # Only add if dst is not already a parent of src in our map
            if dst not in children.get(src, []):
                children.setdefault(src, []).append(dst)

    return children


def _find_root(parse_result: dict[str, Any]) -> int:
    """Find the root node of the dependency tree.

    The root is the node that never appears as a destination (child) in
    parent→child edges. Falls back to node 0 if detection fails.
    """
    num_tokens = parse_result.get("num_tokens", len(parse_result.get("tokens", [])))
    if num_tokens == 0:
        return 0

    edges_src = parse_result.get("edges_src", [])
    edges_dst = parse_result.get("edges_dst", [])

    # Collect all nodes that appear as children
    child_set = set()
    for src, dst in zip(edges_src, edges_dst):
        if 0 <= src < num_tokens and 0 <= dst < num_tokens:
            child_set.add(dst)

    # Root = node(s) that never appears as child
    for i in range(num_tokens):
        if i not in child_set:
            return i

    return 0


def _get_subtree_nodes(children: dict[int, list[int]], root: int) -> list[int]:
    """Collect all node indices in the subtree rooted at `root` (DFS)."""
    result = []
    stack = [root]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for child in children.get(node, []):
            if child not in visited:
                stack.append(child)
    return sorted(result)


def _compute_depths(parse_result: dict[str, Any]) -> list[int]:
    """Compute the depth of each node in the dependency tree via BFS from root."""
    num_tokens = parse_result.get("num_tokens", len(parse_result.get("tokens", [])))
    if num_tokens == 0:
        return []

    children = _build_children_map(parse_result)
    root = _find_root(parse_result)

    depths = [-1] * num_tokens
    depths[root] = 0
    queue = [root]

    while queue:
        node = queue.pop(0)
        for child in children.get(node, []):
            if depths[child] == -1:
                depths[child] = depths[node] + 1
                queue.append(child)

    # Nodes unreachable from root (parse errors) get depth = max_depth + 1
    max_depth = max(d for d in depths if d >= 0) if any(d >= 0 for d in depths) else 0
    for i in range(num_tokens):
        if depths[i] == -1:
            depths[i] = max_depth + 1

    return depths


# =========================================================================
# Strategy 1: Subtree Crop
# =========================================================================


def subtree_crop(parse_result: dict[str, Any], min_tokens: int = 3) -> str:
    """Extract a random subtree from the dependency parse as a positive view.

    Picks a random internal node whose subtree has between ``min_tokens``
    and ``num_tokens - 1`` words, then returns those words in their original
    sentence order.

    Falls back to the full sentence if no valid subtree exists (e.g., very
    short sentences or flat parses).

    Args:
        parse_result: Parse dict with keys: tokens, edges_src, edges_dst, num_tokens.
        min_tokens: Minimum subtree size to consider.

    Returns:
        Augmented sentence string.
    """
    tokens = parse_result.get("tokens", [])
    num_tokens = len(tokens)

    if num_tokens <= min_tokens:
        return " ".join(tokens)

    children = _build_children_map(parse_result)

    # Collect valid subtrees (big enough, but not the whole tree)
    candidates: list[list[int]] = []
    for node in range(num_tokens):
        subtree = _get_subtree_nodes(children, node)
        if min_tokens <= len(subtree) < num_tokens:
            candidates.append(subtree)

    if not candidates:
        return " ".join(tokens)

    chosen = random.choice(candidates)
    return " ".join(tokens[i] for i in chosen)


# =========================================================================
# Strategy 2: Dependency Reorder
# =========================================================================


def dep_reorder(parse_result: dict[str, Any]) -> str:
    """Reorder tokens by dependency depth (SynCSE-inspired).

    Tokens closer to the root come first. Within the same depth level,
    original sentence order is preserved (stable sort).

    This produces a semantics-preserving but surface-different view that
    emphasizes hierarchical structure over linear order.

    Args:
        parse_result: Parse dict with keys: tokens, edges_src, edges_dst, num_tokens.

    Returns:
        Reordered sentence string.
    """
    tokens = parse_result.get("tokens", [])
    if len(tokens) <= 1:
        return " ".join(tokens)

    depths = _compute_depths(parse_result)

    # Stable sort by depth — preserves original order within same depth
    indexed = sorted(enumerate(tokens), key=lambda x: (depths[x[0]], x[0]))
    return " ".join(w for _, w in indexed)


# =========================================================================
# Strategy 3: Leaf Deletion
# =========================================================================


def leaf_deletion(parse_result: dict[str, Any], drop_ratio: float = 0.3) -> str:
    """Remove random leaf nodes (modifiers) from the sentence.

    Leaf nodes in a dependency tree are typically modifiers (adjectives,
    adverbs, determiners) rather than core arguments. Dropping them
    produces a shorter but still grammatically reasonable sentence.

    Falls back to the full sentence if too few words would remain.

    Args:
        parse_result: Parse dict with keys: tokens, edges_src, edges_dst, num_tokens.
        drop_ratio: Fraction of leaf nodes to randomly drop.

    Returns:
        Augmented sentence string with some leaves removed.
    """
    tokens = parse_result.get("tokens", [])
    num_tokens = len(tokens)

    if num_tokens <= 2:
        return " ".join(tokens)

    children = _build_children_map(parse_result)

    # Identify leaf nodes (no children)
    leaves = [i for i in range(num_tokens) if not children.get(i, [])]

    if not leaves:
        return " ".join(tokens)

    # Don't drop so many that the sentence becomes trivial
    num_to_drop = max(1, int(len(leaves) * drop_ratio))
    # Ensure at least 2 tokens remain
    max_drop = max(0, num_tokens - 2)
    num_to_drop = min(num_to_drop, max_drop)

    if num_to_drop == 0:
        return " ".join(tokens)

    dropped = set(random.sample(leaves, num_to_drop))
    remaining = [tokens[i] for i in range(num_tokens) if i not in dropped]

    return " ".join(remaining) if remaining else " ".join(tokens)


# =========================================================================
# Dispatcher
# =========================================================================

STRATEGIES = {
    "subtree_crop": subtree_crop,
    "dep_reorder": dep_reorder,
    "leaf_deletion": leaf_deletion,
}


def augment_sentence(
    parse_result: dict[str, Any],
    strategy: str = "subtree_crop",
    **kwargs,
) -> str:
    """Apply a syntax-aware augmentation to a parsed sentence.

    Args:
        parse_result: Parse result dict from StanzaSyntaxParser.
        strategy: One of "subtree_crop", "dep_reorder", "leaf_deletion".
        **kwargs: Strategy-specific arguments (e.g., min_tokens, drop_ratio).

    Returns:
        Augmented sentence string.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown augmentation strategy '{strategy}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[strategy](parse_result, **kwargs)
