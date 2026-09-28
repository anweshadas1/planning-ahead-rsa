"""Precompute the legal utterance sequences up to horizon H."""

import os

from .paths import BFS_PRECOMPUTED_SEQ_DIR_PATH, UTTERANCE_MAP_4D6P
from .simulation import SequencePrecomputer


def tree_path(horizon):
    return os.path.join(BFS_PRECOMPUTED_SEQ_DIR_PATH, f"precomputed_H_{horizon}.csv")


def build_trees(max_horizon=7, utterance_map=None, overwrite=False):
    """Generate precomputed_H_1..max_horizon.csv if they are not already there."""
    os.makedirs(BFS_PRECOMPUTED_SEQ_DIR_PATH, exist_ok=True)
    target = tree_path(max_horizon)

    if os.path.exists(target) and not overwrite:
        print(f"Sequence tree for H={max_horizon} already present at {target}")
        return target

    precomputer = SequencePrecomputer(utterance_map or UTTERANCE_MAP_4D6P,
                                      max_horizon=max_horizon)
    precomputer.save_precomputed(file_pattern=tree_path("{}"))

    print(f"Sequence trees written to {BFS_PRECOMPUTED_SEQ_DIR_PATH}")
    return target


def require_tree(horizon=7):
    """Ensure the tree for `horizon` exists, building it if necessary."""
    path = tree_path(horizon)
    if not os.path.exists(path):
        build_trees(max_horizon=horizon)
    return path
