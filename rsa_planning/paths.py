"""Paths and shared settings. Set RSA_PLANNING_OUTPUT to change the output folder."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)

DATA_DIR = os.path.join(REPO_ROOT, "data")
LEXICON_DIR_PATH = os.path.join(DATA_DIR, "lexicon_tables")
UTT_MAP_DIR_PATH = os.path.join(DATA_DIR, "utterance_mapping")
SCENARIO_DIR_PATH = os.path.join(DATA_DIR, "scenarios")
BFS_PRECOMPUTED_SEQ_DIR_PATH = os.path.join(DATA_DIR, "precomputed_4D6P")

OUTPUT_DIR = os.environ.get("RSA_PLANNING_OUTPUT") or os.path.join(REPO_ROOT, "output")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
TRAJECTORY_DIR = os.path.join(OUTPUT_DIR, "trajectories")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

LEXICON_4D6P = os.path.join(LEXICON_DIR_PATH, "lexicon_table_4D6P.csv")
UTTERANCE_MAP_4D6P = os.path.join(UTT_MAP_DIR_PATH, "utter_map_4D6P.csv")
PROPERTY_SPEC_CSV = os.path.join(UTT_MAP_DIR_PATH, "prop_spec.csv")

DEFAULT_SCENARIOS_CSV = os.path.join(SCENARIO_DIR_PATH, "dynamic_scenarios_v5_0725.csv")
DEFAULT_TEST_CASES_CSV = os.path.join(SCENARIO_DIR_PATH, "dynamic_test_cases_v5_0725.csv")

# model key -> name in the paper
MODEL_ALIASES = {
    "full": "d-RSA + Priors + Planning",
    "uniform": "d-RSA + Planning",
    "myopic": "d-RSA + Priors",
    "baseline": "d-RSA",
}

# planning horizon (7 in the paper)
DEFAULT_HORIZON = 7

PLANNING_MODELS = ("full", "uniform")
GREEDY_MODELS = ("myopic", "baseline")
UNIFORM_PRIOR_MODELS = ("uniform", "baseline")


def ensure_output_dirs():
    for path in (OUTPUT_DIR, RUNS_DIR, TRAJECTORY_DIR, SUMMARY_DIR, COMPARISON_DIR):
        os.makedirs(path, exist_ok=True)


def run_dir(model, horizon=None):
    """Runs at a non-default horizon get their own folder."""
    if horizon is not None and horizon != DEFAULT_HORIZON:
        return os.path.join(RUNS_DIR, f"{model}_H{horizon}")
    return os.path.join(RUNS_DIR, model)


def trajectory_dir(model):
    return os.path.join(TRAJECTORY_DIR, model)


def summary_csv(model):
    return os.path.join(SUMMARY_DIR, f"summary_{model}.csv")
