from __future__ import annotations

from typing import Sequence


def bounded_subset_sum(
    values_paise: Sequence[int],
    target_paise: int,
    max_items: int,
    min_items: int = 1,
    tolerance_paise: int = 100,
    max_nodes: int = 30000,
) -> tuple[int, ...] | None:
    """Find a bounded sum within tolerance, returning source positions."""
    if max_items < 1 or min_items < 1 or min_items > max_items or max_nodes < 1:
        raise ValueError("subset bounds and max_nodes must be positive and consistent")
    values = [int(value) for value in values_paise]
    if not values or target_paise < 0:
        return None
    # Keep original positions while searching larger payments first.
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    nodes = 0
    best: tuple[int, ...] | None = None
    best_gap = tolerance_paise + 1

    def visit(start: int, total: int, picked: tuple[int, ...]) -> None:
        nonlocal nodes, best, best_gap
        if nodes >= max_nodes:
            return
        nodes += 1
        gap = abs(target_paise - total)
        if len(picked) >= min_items and gap <= tolerance_paise and gap < best_gap:
            best, best_gap = picked, gap
            if gap == 0:
                return
        if len(picked) >= max_items or total > target_paise + tolerance_paise:
            return
        for position in range(start, len(ordered)):
            original, value = ordered[position]
            if value <= 0:
                continue
            if total + value > target_paise + tolerance_paise and total < target_paise - tolerance_paise:
                continue
            visit(position + 1, total + value, picked + (original,))
            if nodes >= max_nodes or best_gap == 0:
                break

    visit(0, 0, ())
    return best


def find_bounded_subset(
    values_paise: Sequence[int],
    target_paise: int,
    min_items: int,
    max_items: int,
    tolerance_paise: int = 100,
    max_nodes: int = 30000,
) -> tuple[int, ...] | None:
    """Find a subset that has between min_items and max_items members."""
    if min_items < 1 or min_items > max_items:
        raise ValueError("min_items must be positive and no larger than max_items")
    return bounded_subset_sum(
        values_paise,
        target_paise,
        max_items=max_items,
        min_items=min_items,
        tolerance_paise=tolerance_paise,
        max_nodes=max_nodes,
    )
