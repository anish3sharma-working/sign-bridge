"""Group-aware splits to keep frames from one recording/participant in one partition."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_by_group(labels, groups, test_size: float = 0.2, validation_size: float = 0.2, random_state: int = 42):
    y = np.asarray(labels, dtype=str)
    g = np.asarray(groups, dtype=str)
    if len(y) != len(g) or len(y) < 3:
        raise ValueError("At least three labeled samples with matching recording groups are required.")
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two gesture classes.")
    group_classes = {label: set(g[y == label]) for label in np.unique(y)}
    insufficient = [label for label, values in group_classes.items() if len(values) < 3]
    if insufficient:
        raise ValueError("Each class needs samples from at least three distinct recording/participant groups to create leakage-safe train, validation, and test splits: " + ", ".join(insufficient))
    classes = set(np.unique(y))
    best = None
    # Search reproducible candidates, choosing one where every class occurs in every split.
    for seed in range(random_state, random_state + 30):
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val, test = next(outer.split(np.zeros(len(y)), y, g))
        if set(y[test]) != classes:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed + 101)
        tr_rel, val_rel = next(inner.split(np.zeros(len(train_val)), y[train_val], g[train_val]))
        train, val = train_val[tr_rel], train_val[val_rel]
        if set(y[train]) == classes and set(y[val]) == classes:
            best = (train, val, test)
            break
    if best is None:
        raise ValueError("Could not form class-complete group-independent train, validation, and test sets. Collect more recordings for each class and use distinct participant groups where possible.")
    train, val, test = best
    group_sets = [set(g[idx]) for idx in best]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Internal split error: a recording group appears in multiple partitions.")
    return {"train": train, "validation": val, "test": test}
