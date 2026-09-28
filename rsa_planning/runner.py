"""Run a model variant over a set of test cases."""

import ast
import os
from functools import partial
from multiprocessing import Pool

import pandas as pd

from .paths import (DEFAULT_HORIZON, DEFAULT_TEST_CASES_CSV, GREEDY_MODELS,
                    LEXICON_4D6P, MODEL_ALIASES, PLANNING_MODELS,
                    UNIFORM_PRIOR_MODELS, UTTERANCE_MAP_4D6P,
                    ensure_output_dirs, run_dir)
from .simulation import (GreedySimulation, PlanningSimulation,
                         generate_uniform_belief)


def build_simulation(model, tc_id, all_world_states, initial_human_belief, output_file,
                     lexicon=None, utterance_map=None, time_horizon=DEFAULT_HORIZON):
    """Instantiate the right simulation class for `model`, priors already set."""
    lexicon = lexicon or LEXICON_4D6P
    utterance_map = utterance_map or UTTERANCE_MAP_4D6P

    if model in PLANNING_MODELS:
        simulation = PlanningSimulation(tc_id, lexicon, utterance_map, all_world_states,
                                        initial_human_belief, output_file, time_horizon)
    elif model in GREEDY_MODELS:
        simulation = GreedySimulation(lexicon, utterance_map, all_world_states,
                                      initial_human_belief, output_file, time_horizon)
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from {sorted(MODEL_ALIASES)}.")

    simulation.mode.uniform_priors_mode = model in UNIFORM_PRIOR_MODELS
    return simulation


def run_one(scenario_data, model="full", lexicon=None, utterance_map=None,
            time_horizon=DEFAULT_HORIZON, overwrite=False):
    """Run a single scenario under a single model variant."""
    tc_id = scenario_data["tc_id"]
    output_dir = run_dir(model, time_horizon)
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"result_{tc_id}_{model}.xlsx")

    if os.path.exists(result_path) and not overwrite:
        return {"tc_id": tc_id, "status": "skipped", "result_path": result_path}

    try:
        all_world_states = ast.literal_eval(scenario_data["all_scaled_world_states"])
        belief = ast.literal_eval(scenario_data["user_belief_distribution"])
        # uniform / baseline use a flat belief
        if model in UNIFORM_PRIOR_MODELS:
            belief = generate_uniform_belief(belief)
    except (ValueError, SyntaxError) as exc:
        return {"tc_id": tc_id, "status": "failed", "error": f"Parsing error: {exc}"}

    try:
        simulation = build_simulation(model, tc_id, all_world_states, belief,
                                      result_path, lexicon, utterance_map, time_horizon)
        simulation.search()
        simulation.prepare_op_file()
        return {"tc_id": tc_id, "status": "completed", "result_path": result_path}
    except Exception as exc:
        return {"tc_id": tc_id, "status": "failed", "error": str(exc)}


def load_scenarios(test_cases_csv=None, tc_ids=None, limit=None):
    """Load test cases, optionally restricted to given ids or a head slice."""
    df = pd.read_csv(test_cases_csv or DEFAULT_TEST_CASES_CSV)
    if tc_ids:
        df = df[df["tc_id"].isin(tc_ids)]
    if limit:
        df = df.head(limit)
    return df


def run_model(model, test_cases_csv=None, tc_ids=None, limit=None,
              processes=4, time_horizon=DEFAULT_HORIZON, overwrite=False):
    """Run one model variant over a set of scenarios, in parallel."""
    ensure_output_dirs()
    scenarios = load_scenarios(test_cases_csv, tc_ids, limit)
    if scenarios.empty:
        print("No scenarios selected.")
        return []

    print(f"Running '{model}' ({MODEL_ALIASES[model]}) over {len(scenarios)} "
          f"scenarios at H={time_horizon} on {processes} processes")

    rows = [row for _, row in scenarios.iterrows()]
    worker = partial(run_one, model=model, time_horizon=time_horizon,
                     overwrite=overwrite)

    if processes <= 1:
        results = [worker(row) for row in rows]
    else:
        with Pool(processes) as pool:
            results = pool.map(worker, rows)

    done = sum(r["status"] == "completed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    failed = [r for r in results if r["status"] == "failed"]
    print(f"{done} completed, {skipped} skipped, {len(failed)} failed")
    for r in failed[:10]:
        print(f"  FAILED {r['tc_id']}: {r['error']}")
    return results
