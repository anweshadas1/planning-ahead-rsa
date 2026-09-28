"""Rebuild the rank-1 trajectory of each run."""

import ast
import os
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from .paths import (DEFAULT_TEST_CASES_CSV, ensure_output_dirs, run_dir,
                    trajectory_dir)


def safe_literal_eval(value):
    """Safely convert string representations to Python objects"""
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except:
        return {}

def state_to_bin(value):

    if isinstance(value, float):
        bin_center = round(value * 5) / 5.0
        return f"{bin_center:.1f}"
    else:
        return str(value)

def get_bin_prob(belief_dict, true_bin):
    """
    Safely get probability for a bin from belief dictionary
    Handles different bin key types (str, int, float)
    """
    if not belief_dict:
        return 0

    # Try direct match first
    if true_bin in belief_dict:
        return belief_dict[true_bin]

    # Try alternative representations
    if isinstance(true_bin, float):
        # Try string representation
        str_bin = f"{true_bin:.1f}"
        if str_bin in belief_dict:
            return belief_dict[str_bin]

        # Try rounded integer if applicable
        if true_bin.is_integer():
            int_bin = int(true_bin)
            if int_bin in belief_dict:
                return belief_dict[int_bin]

    elif isinstance(true_bin, int):
        # Try string representation
        str_bin = str(true_bin)
        if str_bin in belief_dict:
            return belief_dict[str_bin]

    elif isinstance(true_bin, str):
        # Try float conversion for numeric strings
        if '.' in true_bin:
            try:
                float_bin = float(true_bin)
                if float_bin in belief_dict:
                    return belief_dict[float_bin]
            except ValueError:
                pass
        # Try integer conversion
        elif true_bin.isdigit():
            int_bin = int(true_bin)
            if int_bin in belief_dict:
                return belief_dict[int_bin]

    # Default to 0 if no match found
    return 0

def calculate_tvd(dist1, dist2):
    bins = set(dist1.keys()) | set(dist2.keys())

    tvd = 0.0
    for b in bins:
        p1 = dist1.get(b, 0)
        p2 = dist2.get(b, 0)
        tvd += abs(p1 - p2)

    return 0.5 * tvd

def extract_rank1_entry(df):
    rank1 = df[df['rank'] == 1]
    return rank1.sort_values('timestep', ascending=False).iloc[0] if not rank1.empty else None

def preprocess_trajectory_df(df):
    """Convert all string columns to proper objects"""
    for col in ['world_state', 'beliefs', 'critical_properties', 'low_aware_crit_prop']:
        if col in df:
            df[col] = df[col].apply(safe_literal_eval)
    return df

def calculate_prioritization_metrics(results, critical_props):

    low_delays = list(results["low_aware_delays"].values())
    other_delays = [d for prop, d in results["awareness_delays"].items()
                    if critical_props[prop]["awareness"] != "low"]


    median_low = np.median(low_delays) if low_delays else np.nan
    median_other = np.median(other_delays) if other_delays else np.nan


    if not np.isnan(median_low) and not np.isnan(median_other) and (median_low + median_other) > 0:
        prioritization_index = (median_other - median_low) / (median_other + median_low)
    else:
        prioritization_index = np.nan

    critical_count = len(critical_props)
    low_aware_count = sum(1 for info in critical_props.values() if info["awareness"] == "low")

    return {
        "med_delay_low": median_low if not np.isnan(median_low) else -1,
        "med_delay_other": median_other if not np.isnan(median_other) else -1,
        "prioritization_index": prioritization_index if not np.isnan(prioritization_index) else 0,
        "prop_missed_critical": len(results["missed_critical"]) / critical_count if critical_count else 0,
        "prop_missed_low_critical": len(results["missed_low_critical"]) / low_aware_count if low_aware_count else 0
    }

def calculate_temporal_metrics(trajectory_df, tc_info, awareness_threshold=0.7):

    trajectory_df = preprocess_trajectory_df(trajectory_df.copy())
    critical_info = safe_literal_eval(tc_info["list_cr_prop_aware"])

    results = {
        "first_awareness_times": {},
        "awareness_delays": {},
        "low_aware_delays": {},
        "missed_critical": [],
        "missed_low_critical": []
    }

    critical_props = {}
    for (prop, crit_time), awareness in critical_info.items():
        if crit_time <= 7:
            critical_props[prop] = {"crit_time": crit_time, "awareness": awareness}


    for timestep in range(1, 8):
        current_row = trajectory_df[trajectory_df['timestep'] == timestep]
        if current_row.empty:
            continue

        if current_row['current_msg'].iloc[0] == '(XXX)':
            continue


        world_state = safe_literal_eval(current_row['world_state'].iloc[0])
        beliefs = safe_literal_eval(current_row['beliefs'].iloc[0])


        if not isinstance(world_state, dict) or not isinstance(beliefs, dict):
            continue


        for prop, info in critical_props.items():
            crit_time = info["crit_time"]
            awareness = info["awareness"]

            if timestep < crit_time:
                continue

            if prop not in world_state:
                continue


            true_value = world_state[prop]
            true_bin = state_to_bin(true_value)

            prop_belief = beliefs.get(prop, {})
            current_prob = get_bin_prob(prop_belief, true_bin)


            if current_prob >= awareness_threshold:

                if prop not in results["first_awareness_times"]:
                    results["first_awareness_times"][prop] = timestep
                    delay = timestep - crit_time
                    results["awareness_delays"][prop] = max(0.01, delay)
                    if awareness == "low":
                        results["low_aware_delays"][prop] = max(0.01, delay)


    for prop, info in critical_props.items():
        if prop not in results["first_awareness_times"]:
            results["missed_critical"].append(prop)
            results["awareness_delays"][prop] = max(0.01, 7)
            if info["awareness"] == "low":
                results["missed_low_critical"].append(prop)
                results["low_aware_delays"][prop] = max(0.01, 7)
        if prop in results["first_awareness_times"] and results["first_awareness_times"][prop] == 7:
            results["missed_critical"].append(prop)
            if info["awareness"] == "low":
                results['missed_low_critical'].append(prop)


    results.update(calculate_prioritization_metrics(results, critical_props))

    return results

def reconstruct_rank1_trajectory(run_df, rank1_sequence, tc_info):
    critical_map = {}
    low_aware_map = {}
    critical_props = set()
    low_aware_props = set()

    critical_info = ast.literal_eval(tc_info["list_cr_prop_aware"])
    for (prop, crit_time), awareness in critical_info.items():
        if crit_time <= 7:
            critical_props.add(prop)
            if crit_time not in critical_map:
                critical_map[crit_time] = []
            critical_map[crit_time].append(prop)

            if awareness == 'low':
                low_aware_props.add(prop)
                if crit_time not in low_aware_map:
                    low_aware_map[crit_time] = []
                low_aware_map[crit_time].append(prop)

    seq_list = list(rank1_sequence)
    all_world_states = ast.literal_eval(tc_info["all_scaled_world_states"])

    last_beliefs = {}
    last_reward = 0
    trajectory = []

    for timestep in range(7, 0, -1):
        current_seq = tuple(seq_list)
        current_msg = seq_list[-1] if seq_list else None
        crit_props = critical_map.get(timestep, [])
        low_aware_crit = low_aware_map.get(timestep, [])

        try:
            row = run_df[(run_df["timestep"] == timestep) &
                         (run_df["utterance_sequence"] == current_seq)].iloc[0]

            last_beliefs = ast.literal_eval(row["beliefs"])
            last_reward = float(row["total_reward"])
            legality = 1
            world_state = all_world_states.get(timestep)

        except IndexError:
            legality = 0
            world_state = all_world_states.get(timestep)

        trajectory.append({
            "timestep": timestep,
            "sequence": current_seq,
            "legality": legality,
            "current_msg": current_msg,
            "total_reward": last_reward,
            "world_state": world_state,
            "beliefs": last_beliefs,
            "critical_properties": crit_props,
            "low_aware_crit_prop": low_aware_crit,
        })

        if timestep > 1:
            seq_list.pop()

    trajectory.append({
        "timestep": 0,
        "sequence": "",
        "legality": 0,
        "current_msg": "",
        "total_reward": 0,
        "world_state": all_world_states.get(1),
        "beliefs": ast.literal_eval(tc_info["user_belief_distribution"]),
        "critical_properties": [],
        "low_aware_crit_prop": [],
    })

    df = pd.DataFrame(list(trajectory))

    df['tc_id'] = tc_info["tc_id"]
    df['final_sequence'] = str(rank1_sequence)

    column_order = [
        'tc_id', 'timestep', 'sequence', 'current_msg', 'legality',
        'total_reward', 'critical_properties', 'low_aware_crit_prop',
        'world_state', 'beliefs', 'final_sequence',
    ]

    return df[column_order]

def process_single_file(args):

    filename, full_run_dir, trajectory_dir, model_name, tc_df, world_mode, version_num = args

    print(f"Processing: {filename}...")

    match = re.match(r"result_(.*?)_" + re.escape(model_name) + r"\.xlsx", filename)
    if match:
        tc_id = match.group(1)
        run_fp = os.path.join(full_run_dir, filename)

        try:
            run_df = pd.read_excel(run_fp, engine='openpyxl')
        except Exception as e:
            print(f"Error reading {run_fp}: {e}")
            return # Skip this file

        # Ensure 'utterance_sequence' column exists and convert if necessary
        if 'utterance_sequence' in run_df.columns:
            run_df['utterance_sequence'] = run_df['utterance_sequence'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else tuple()
            )
        else:
            print(f"Warning: 'utterance_sequence' column not found in {run_fp}")
            run_df['utterance_sequence'] = [()] * len(run_df) # Add an empty sequence column


        tc_info_row = tc_df[tc_df['tc_id'] == tc_id]
        if tc_info_row.empty:
            print(f"Warning: No test case information found for tc_id: {tc_id} in {filename}")
            return
        tc_info = tc_info_row.iloc[0]

        try:
            rank1_entry = extract_rank1_entry(run_df)
            rank1_sequence = rank1_entry['utterance_sequence']
        except Exception as e:
            print(f"Error extracting rank1_entry from {filename}: {e}")
            return

        try:
            trajectory = reconstruct_rank1_trajectory(run_df, rank1_sequence, tc_info)
        except Exception as e:
            print(f"Error reconstructing trajectory for {filename}: {e}")
            return

        trajectory_filename = f"result_{tc_id}_{model_name}.csv"
        trajectory_fp = os.path.join(trajectory_dir, trajectory_filename)
        try:
            trajectory.to_csv(trajectory_fp, index=False)
            print(f"Finished: {filename}")
        except Exception as e:
            print(f"Error saving trajectory for {filename} to {trajectory_fp}: {e}")

def build_trajectories(model, test_cases_csv=None, processes=None):
    """Convert every run file for `model` into a rank-1 trajectory CSV."""
    ensure_output_dirs()
    source_dir = run_dir(model)
    target_dir = trajectory_dir(model)
    os.makedirs(target_dir, exist_ok=True)

    test_cases_csv = test_cases_csv or DEFAULT_TEST_CASES_CSV
    tc_df = pd.read_csv(test_cases_csv)

    tasks = [(fn, source_dir, target_dir, model, tc_df, "dynamic", "")
             for fn in sorted(os.listdir(source_dir))
             if fn.startswith("result_") and fn.endswith(f"_{model}.xlsx")]

    if not tasks:
        print(f"No run files for model '{model}' in {source_dir}")
        return target_dir

    n = processes or min(cpu_count(), len(tasks))
    print(f"Reconstructing {len(tasks)} trajectories for '{model}' on {n} processes")
    with Pool(n) as pool:
        pool.map(process_single_file, tasks)
    return target_dir
