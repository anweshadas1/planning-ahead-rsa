"""Per-run metrics (reward, entropy, delays, ...) for each test case."""

import ast
import math
import multiprocessing
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .paths import (DEFAULT_TEST_CASES_CSV, UTTERANCE_MAP_4D6P,
                    ensure_output_dirs, run_dir, summary_csv, trajectory_dir)
from .trajectories import calculate_temporal_metrics


def _load_master_test_cases(master_csv_path: str):
    global _master_test_cases_df
    if _master_test_cases_df is None:
        try:
            _master_test_cases_df = pd.read_csv(master_csv_path)
            _master_test_cases_df['tc_id'] = _master_test_cases_df['tc_id'].astype(str)
            print(f"Loaded master test cases from: {master_csv_path}")
        except FileNotFoundError:
            print(f"Error: Master test cases file not found at {master_csv_path}")
            _master_test_cases_df = pd.DataFrame()  # Set to empty DataFrame to prevent re-attempts
    return _master_test_cases_df

def _load_utterance_map(utt_map_path: str):
    global _utterance_map_df
    if _utterance_map_df is None:
        try:
            _utterance_map_df = pd.read_csv(utt_map_path, index_col='Messages')
            _utterance_map_df['Time'] = pd.to_numeric(_utterance_map_df['Time'], errors='coerce')
            _utterance_map_df.dropna(subset=['Time'], inplace=True)
            print(f"Loaded utterance map from: {utt_map_path}")
        except FileNotFoundError:
            print(f"Error: Utterance map file not found at {utt_map_path}")
            _utterance_map_df = pd.DataFrame()  # Set to empty DataFrame
    return _utterance_map_df

def parse_filename(filename: str) -> dict:
    match = re.match(r'result_([a-zA-Z]+)_(\d+)_(\d+)_(\d+)_([a-zA-Z]+)\.xlsx', filename)
    if match:
        sc_type = match.group(1)
        counter = int(match.group(2))
        num_critical_prop = int(match.group(3))
        crit_prop_aware_type = int(match.group(4))
        model_type = match.group(5)

        # Reconstruct tc_id as {sc_type}_{counter}_{num_critical_prop}_{aware_type}
        tc_id = f"{sc_type}_{counter}_{num_critical_prop}_{crit_prop_aware_type}"

        return {
            'tc_id': tc_id,
            'sc_type': sc_type,
            'counter': counter,
            'num_critical_prop': num_critical_prop,
            'user_critical_aware_type': crit_prop_aware_type,
            'model_type': model_type
        }
    else:
        print(f"Warning: Filename '{filename}' does not match expected format. Skipping.")
        return None

def read_result_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and validate result Excel file"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')

        # Validate required columns
        required_cols = ['rank', 'timestep', 'utterance_sequence', 'total_reward', 'cost']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing columns {missing} in {os.path.basename(file_path)}")
            return None

        # Clean numeric columns
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
        return df.dropna(subset=['rank', 'timestep'])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_rank1_entry(df: pd.DataFrame) -> Optional[pd.Series]:
    """Get best rank 1 entry (highest timestep)"""
    rank1 = df[df['rank'] == 1]
    return rank1.sort_values('timestep', ascending=False).iloc[0] if not rank1.empty else None

def count_message_types(sequence: Union[str, List, Tuple], utterance_map_df: pd.DataFrame) -> Dict[str, int]:
    """Count occurrences of different message types with exact original behavior"""
    counts = {
        'beep': 0,
        'silent': 0,
        'time2': 0,
        'time3': 0
    }

    # Handle null/empty sequence
    if not sequence:
        return counts

    # Handle string representation of sequences
    if isinstance(sequence, str):
        try:
            sequence = ast.literal_eval(sequence)
        except (ValueError, SyntaxError) as e:
            print(f"Could not parse sequence string: {e}")
            return counts

    # Only proceed if we have a list/tuple
    if not isinstance(sequence, (list, tuple)):
        return counts

    # Count each message type
    for msg in sequence:
        if msg == "Beep":
            counts['beep'] += 1
        elif msg == "(...)":
            counts['silent'] += 1
        else:
            # Handle both tuple and non-tuple messages
            message_name = msg[0] if isinstance(msg, tuple) else msg

            # Check utterance map only if available and message exists
            if (utterance_map_df is not None and
                    not utterance_map_df.empty and
                    message_name in utterance_map_df.index):

                msg_time = utterance_map_df.loc[message_name, 'Time']
                if msg_time == 3:
                    counts['time3'] += 1
                elif msg_time == 2:
                    counts['time2'] += 1
            else:
                # If no utterance map or message not found, count as nothing
                pass

    return counts

def extract_critical_properties(list_cr_prop_aware: dict, sc_type: str) -> List[str]:
    """Get flat list of critical properties (removing timestamps for dynamic)"""
    if not list_cr_prop_aware:
        return []

    if sc_type == 'stat':
        return list(list_cr_prop_aware.keys())

    # For dynamic: keys are (property, timestamp) tuples
    return [prop for prop, _ in list_cr_prop_aware.keys()]

def compute_shared_metrics(critical_props: List[str]) -> Dict[str, float]:
    """Calculate sharing metrics for properties"""
    if not critical_props:
        return {'shared_attr': 0, 'shared_drones': 0, 'sharing_metric': 0}

    # Count attribute sharing (last part after '_')
    attr_counts = {}
    for prop in critical_props:
        attr = prop.split('_')[-1]
        attr_counts[attr] = attr_counts.get(attr, 0) + 1

    # Count drone sharing (first part before '_')
    drone_counts = {}
    for prop in critical_props:
        drone = prop.split('_')[0]
        drone_counts[drone] = drone_counts.get(drone, 0) + 1

    # Calculate sharing metrics
    shared_attr = sum(c for c in attr_counts.values() if c > 1)
    shared_drones = sum(c for c in drone_counts.values() if c > 1)
    total_props = len(critical_props)

    return {
        'shared_attr': shared_attr,
        'shared_drones': shared_drones,
        'sharing_metric': (shared_attr + shared_drones) / (2 * total_props) if total_props else 0
    }

def extract_awareness_levels(list_cr_prop_aware: dict, sc_type: str) -> Dict[str, List[str]]:
    """Categorize properties by awareness level"""
    low_aware = []
    high_aware = []

    if not list_cr_prop_aware:
        return {'low': low_aware, 'high': high_aware}

    # Static: {property: awareness}
    if sc_type == 'stat':
        for prop, awareness in list_cr_prop_aware.items():
            if awareness == 'low':
                low_aware.append(prop)
            elif awareness == 'high':
                high_aware.append(prop)
    # Dynamic: {(property, time): awareness}
    else:
        for (prop, _), awareness in list_cr_prop_aware.items():
            if awareness == 'low':
                low_aware.append(prop)
            elif awareness == 'high':
                high_aware.append(prop)

    return {'low': low_aware, 'high': high_aware}

def compute_HOW_metrics(counts, cost):

    total_steps = 7 # or len(sequence) or max_timesteps; fixed
    total_msgs = cost

    metrics = {
        'pct_beep': 0.0,
        'pct_time3': 0.0,
        'silent_step_ratio': 0.0,
        'verbosity_ratio': 0.0,
        'avg_dur_per_spoken_msg': 0.0,
        'entropy_spoken_msgs': 0.0,
    }

    num_silent = counts['silent']
    num_time2 = counts['time2']
    num_time3 = counts['time3']
    num_beep = counts['beep']

    if total_msgs > 0:

        metrics['pct_beep'] = (num_beep / total_msgs) * 100
        metrics['pct_time3'] = (num_time3 / total_msgs) * 100
        metrics['silent_step_ratio'] = num_silent / total_steps

        if (num_time2 + num_beep) > 0:
            metrics['verbosity_ratio'] = num_time3 / (num_time2 + num_beep)
        else:
            metrics['verbosity_ratio'] = num_time3

        total_dur = 3 * num_time3 + 2 * num_time2 + num_beep
        metrics['avg_dur_per_spoken_msg'] = total_dur / total_msgs

        p_beep = num_beep / total_msgs
        p_time2 = num_time2 / total_msgs
        p_time3 = num_time3 / total_msgs
        entropy = 0.0

        if p_beep > 0:
            entropy -= p_beep * math.log2(p_beep)
        if p_time2 > 0:
            entropy -= p_time2 * math.log2(p_time2)
        if p_time3 > 0:
            entropy -= p_time3 * math.log2(p_time3)
        metrics['entropy_spoken_msgs'] = entropy

    return metrics

def process_rank10_entries(df: pd.DataFrame) -> Dict:
    """Process top 10 ranked sequences"""
    rank10 = df[df['rank'] <= 10].sort_values('rank')
    if rank10.empty:
        return {
            'sequences': [],
            'rewards': [],
            'mean_reward': None
        }

    sequences = []
    rewards = []

    for _, row in rank10.iterrows():
        try:
            seq = ast.literal_eval(row['utterance_sequence']) if isinstance(
                row['utterance_sequence'], str) else row['utterance_sequence']
            sequences.append(seq)
            rewards.append(row['total_reward'])
        except (ValueError, SyntaxError):
            continue

    return {
        'sequences': sequences,
        'rewards': rewards,
        'mean_reward': sum(rewards) / len(rewards) if rewards else None
    }

def process_single_result_file(file_path, utterance_map_df, master_test_cases_df, trajectory_dir):


    # Step 1: Parse filename and load data
    filename = os.path.basename(file_path)
    file_info = parse_filename(filename)
    if not file_info:
        return None

    print(f"Processing: {filename}")

    df = read_result_data(file_path)
    if df is None:
        return None

    rank1_entry = extract_rank1_entry(df)
    if rank1_entry is None:
        print(f"No rank 1 entry found in {filename}")
        return None

    sequence = rank1_entry['utterance_sequence']
    message_counts = count_message_types(sequence, utterance_map_df)

    tc_id = file_info['tc_id']
    tc_row = master_test_cases_df[master_test_cases_df['tc_id'] == tc_id].iloc[
        0] if not master_test_cases_df.empty else None
    if tc_row is None:
        print(f"TC_ID {tc_id} not found in master data")
        return None

    HOW_metrics = compute_HOW_metrics(
        message_counts,
        rank1_entry['cost']
    )

    trajectory_filename = f"result_{tc_id}_{file_info['model_type']}.csv"
    trajectory_fp = os.path.join(trajectory_dir, trajectory_filename)
    trajectory_df = pd.read_csv(trajectory_fp)
    print(trajectory_fp)
    temporal_metrics = calculate_temporal_metrics(trajectory_df, tc_row)


    return {
        'tc_id': tc_id,
        'scenario_id': tc_row['scenario_id'],
        'sc_type': tc_row['sc_type'],
        'user_critical_aware': tc_row['user_critical_aware'],
        'critical_properties': tc_row['critical_properties'],
        'num_critical_prop': tc_row['num_critical_prop'],
        'list_critical_prop': tc_row['list_critical_prop'],
        'list_cr_prop_aware': tc_row['list_cr_prop_aware'],
        'density': tc_row['density'],
        'user_general_aware': tc_row['user_general_aware'],
        'low_aware_props': tc_row['low_aware_props'],
        'high_aware_props': tc_row['high_aware_props'],
        'num_low_aware': tc_row['num_low_aware'],
        'num_high_aware': tc_row['num_high_aware'],
        'shared_attr': tc_row['shared_attr'],
        'shared_drones': tc_row['shared_drones'],
        'sharing_metric': tc_row['sharing_metric'],
        'scaled_world_state': tc_row['scaled_world_state'],
        'user_belief_distribution': tc_row['user_belief_distribution'],
        'sequence_rank1': sequence,
        'total_reward': rank1_entry['total_reward'],
        **HOW_metrics,
        **temporal_metrics,
        'cost': rank1_entry['cost'],
        'num_beep': message_counts['beep'],
        'num_silent': message_counts['silent'],
        'num_time2': message_counts['time2'],
        'num_time3': message_counts['time3'],
    }

def create_summary_dataset(result_dir, utterance_map_path, master_test_cases_path, trajectory_dir):

    utterance_map_df = pd.read_csv(utterance_map_path).set_index('Messages') if utterance_map_path else None
    master_df = pd.read_csv(master_test_cases_path) if master_test_cases_path else None

    files_to_process = []
    for filename in os.listdir(result_dir):
        if filename.endswith((".csv", ".xlsx")):
            file_path = os.path.join(result_dir, filename)
            files_to_process.append((file_path, utterance_map_df, master_df, trajectory_dir))

    results = []
    with multiprocessing.Pool() as pool:
        processed_data = pool.starmap(process_single_result_file, files_to_process)


    for file_data in processed_data:
        if file_data:
            results.append(file_data)

    return pd.DataFrame(results)

def build_summary(model, test_cases_csv=None, utterance_map=None):
    """Write the per-scenario summary CSV for one model variant."""
    ensure_output_dirs()
    out = summary_csv(model)
    df = create_summary_dataset(
        result_dir=run_dir(model),
        utterance_map_path=utterance_map or UTTERANCE_MAP_4D6P,
        master_test_cases_path=test_cases_csv or DEFAULT_TEST_CASES_CSV,
        trajectory_dir=trajectory_dir(model),
    )
    df.to_csv(out, index=False)
    print(f"{len(df)} summary rows -> {out}")
    return out
