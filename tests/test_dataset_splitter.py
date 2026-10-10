import numpy as np
import pytest
from training.dataset_splitter import split_by_group


def test_group_disjoint_splits_have_each_class():
    labels, groups = [], []
    for label in ("A", "B", "C"):
        for i in range(12):
            labels.append(label); groups.append(f"{label}-rec-{i}")
    result = split_by_group(labels, groups)
    group_sets = [set(np.asarray(groups)[result[key]]) for key in ("train", "validation", "test")]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    for indices in result.values():
        assert set(np.asarray(labels)[indices]) == {"A", "B", "C"}


def test_requires_at_least_three_groups_per_class():
    with pytest.raises(ValueError, match="at least three distinct"):
        split_by_group(["A", "A", "B", "B", "A", "B"], ["a1", "a2", "b1", "b2", "a1", "b1"])
