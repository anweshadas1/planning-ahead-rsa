"""Combine the per-model summaries into wide and long comparison tables."""

import ast
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from .paths import COMPARISON_DIR, MODEL_ALIASES, ensure_output_dirs, summary_csv


common_scenario_cols = [
    'scenario_id', 'sc_type', 'user_critical_aware', 'critical_properties',
    'num_critical_prop', 'list_critical_prop', 'list_cr_prop_aware',
    'density', 'user_general_aware', 'low_aware_props', 'high_aware_props',
    'num_low_aware', 'num_high_aware', 'shared_attr', 'shared_drones',
    'sharing_metric', 'scaled_world_state', 'user_belief_distribution'
]


def compare_model_performance(full_model_summary_df, myopic_model_summary_df,
                              uniform_model_summary_df, baseline_model_summary_df, output_csv_path):


    cols_base = ['tc_id', 'sequence_rank1'] + common_scenario_cols


    full_cols = cols_base + ['total_reward']
    myopic_cols = ['tc_id', 'total_reward', 'sequence_rank1']
    uniform_cols = ['tc_id', 'total_reward', 'sequence_rank1']
    baseline_cols = ['tc_id', 'total_reward', 'sequence_rank1']

    for df_name, df, cols in [
        ("Full model", full_model_summary_df, full_cols),
        ("Myopic model", myopic_model_summary_df, myopic_cols),
        ("Uniform model", uniform_model_summary_df, uniform_cols),
        ("Baseline model", baseline_model_summary_df, baseline_cols)
    ]:
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} DataFrame is missing required columns: {missing_cols}")

    merged_df = pd.merge(
        full_model_summary_df[full_cols],
        myopic_model_summary_df[myopic_cols],
        on='tc_id',
        suffixes=('_full', '_myopic')
    )


    uniform_df_for_merge = uniform_model_summary_df[uniform_cols].copy()
    uniform_df_for_merge.rename(columns={'total_reward': 'total_reward_uniform',
                                         'sequence_rank1': 'sequence_rank1_uniform'}, inplace=True)
    merged_df = pd.merge(
        merged_df,
        uniform_df_for_merge[['tc_id', 'total_reward_uniform', 'sequence_rank1_uniform']],
        on='tc_id'
    )


    baseline_df_for_merge = baseline_model_summary_df[baseline_cols].copy()
    baseline_df_for_merge.rename(columns={'total_reward': 'total_reward_baseline',
                                          'sequence_rank1': 'sequence_rank1_baseline'}, inplace=True)
    merged_df = pd.merge(
        merged_df,
        baseline_df_for_merge[['tc_id', 'total_reward_baseline', 'sequence_rank1_baseline']],
        on='tc_id'
    )

    if merged_df.empty:
        print("Merged DataFrame is empty. No common test cases found across all models.")
        return pd.DataFrame()


    for col_suffix in ['_full', '_myopic', '_uniform', '_baseline']:
        col_name = f'sequence_rank1{col_suffix}'
        parsed_col_name = f'{col_name}_parsed'
        if col_name in merged_df.columns:
            try:
                merged_df[parsed_col_name] = merged_df[col_name].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except (ValueError, SyntaxError):
                merged_df[parsed_col_name] = merged_df[col_name].astype(str)
        else:
            merged_df[parsed_col_name] = None
            print(f"Warning: Column {col_name} not found in merged_df. Setting {parsed_col_name} to None.")


    merged_df['diff_full_myopic'] = merged_df['total_reward_full'] - merged_df['total_reward_myopic']
    merged_df['diff_full_uniform'] = merged_df['total_reward_full'] - merged_df['total_reward_uniform']
    merged_df['diff_myopic_uniform'] = merged_df['total_reward_myopic'] - merged_df['total_reward_uniform']


    merged_df['diff_full_baseline'] = merged_df['total_reward_full'] - merged_df['total_reward_baseline']
    merged_df['diff_myopic_baseline'] = merged_df['total_reward_myopic'] - merged_df['total_reward_baseline']
    merged_df['diff_uniform_baseline'] = merged_df['total_reward_uniform'] - merged_df['total_reward_baseline']


    merged_df['sequences_differ'] = (
            (merged_df['sequence_rank1_full_parsed'] != merged_df['sequence_rank1_myopic_parsed']) |
            (merged_df['sequence_rank1_full_parsed'] != merged_df['sequence_rank1_uniform_parsed']) |
            (merged_df['sequence_rank1_full_parsed'] != merged_df['sequence_rank1_baseline_parsed']) |
            (merged_df['sequence_rank1_myopic_parsed'] != merged_df['sequence_rank1_uniform_parsed']) |
            (merged_df['sequence_rank1_myopic_parsed'] != merged_df['sequence_rank1_baseline_parsed']) |
            (merged_df['sequence_rank1_uniform_parsed'] != merged_df['sequence_rank1_baseline_parsed'])
    )


    def get_best_model(row):
        reward_full = row['total_reward_full']
        reward_myopic = row['total_reward_myopic']
        reward_uniform = row['total_reward_uniform']
        reward_baseline = row['total_reward_baseline']  # New

        max_reward = max(reward_full, reward_myopic, reward_uniform, reward_baseline)

        is_full_best = np.isclose(reward_full, max_reward, atol=1e-9)
        is_myopic_best = np.isclose(reward_myopic, max_reward, atol=1e-9)
        is_uniform_best = np.isclose(reward_uniform, max_reward, atol=1e-9)
        is_baseline_best = np.isclose(reward_baseline, max_reward, atol=1e-9)

        winners = []
        if is_full_best:
            winners.append('full')
        if is_myopic_best:
            winners.append('myopic')
        if is_uniform_best:
            winners.append('uniform')
        if is_baseline_best:
            winners.append('baseline')

        if len(winners) > 1:
            return 'tie'
        else:
            return winners[0] if winners else 'none'

    merged_df['best_model'] = merged_df.apply(get_best_model, axis=1)

    final_column_order = (
            ['tc_id'] +
            common_scenario_cols +
            ['total_reward_full', 'sequence_rank1_full'] +
            ['total_reward_myopic', 'sequence_rank1_myopic'] +
            ['total_reward_uniform', 'sequence_rank1_uniform'] +
            ['total_reward_baseline', 'sequence_rank1_baseline'] +
            ['diff_full_myopic', 'diff_full_uniform', 'diff_myopic_uniform',
             'diff_full_baseline', 'diff_myopic_baseline', 'diff_uniform_baseline',
             'sequences_differ', 'best_model']
    )

    df_to_save = merged_df.drop(columns=[col for col in merged_df.columns if '_parsed' in col], errors='ignore')
    existing_columns_in_order = [col for col in final_column_order if col in df_to_save.columns]
    df_to_save = df_to_save[existing_columns_in_order]

    if output_csv_path:
        try:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df_to_save.to_csv(output_csv_path, index=False)
            print(f"Comparison results saved to: {output_csv_path}")
        except Exception as e:
            print(f"Error saving comparison results to CSV: {e}")
            pass

    return df_to_save

def create_long_format_df(
        full_model_summary_df: pd.DataFrame,
        myopic_model_summary_df: pd.DataFrame,
        uniform_model_summary_df: pd.DataFrame,
        baseline_model_summary_df: pd.DataFrame,
        output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    all_dfs_to_concat = []

    model_dfs_map = {
        "full": full_model_summary_df,
        "myopic": myopic_model_summary_df,
        "uniform": uniform_model_summary_df,
        "baseline": baseline_model_summary_df,
    }

    original_summary_cols = []
    for df in model_dfs_map.values():
        if not df.empty:
            original_summary_cols = df.columns.tolist()
            break

    if not original_summary_cols:

        return pd.DataFrame()


    for model_name, df in model_dfs_map.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['model_name'] = model_name
            all_dfs_to_concat.append(df_copy)
        else:
            pass

    if not all_dfs_to_concat:
        return pd.DataFrame()

    long_df = pd.concat(all_dfs_to_concat, ignore_index=True)


    final_column_order = ['tc_id', 'model_name'] + [col for col in original_summary_cols if col != 'tc_id']
    existing_final_column_order = [col for col in final_column_order if col in long_df.columns]
    long_df = long_df[existing_final_column_order]

    def natural_sort_key(tc_id_string):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', tc_id_string)]

    if 'tc_id' in long_df.columns:
        long_df = long_df.sort_values(by='tc_id', key=lambda x: x.apply(natural_sort_key), ascending=True)

    if output_csv_path:
        try:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            long_df.to_csv(output_csv_path, index=False)
        except Exception as e:
            pass

    return long_df

def load_model_summaries():
    """Load the four per-model summary CSVs, tolerating missing ones."""
    frames = {}
    for model in MODEL_ALIASES:
        path = summary_csv(model)
        try:
            frames[model] = pd.read_csv(path)
            print(f"loaded {model}: {len(frames[model])} rows")
        except FileNotFoundError:
            print(f"missing summary for '{model}' at {path}")
            frames[model] = pd.DataFrame()
    return frames


def build_comparison():
    """Write the wide and long comparison tables across all four models."""
    ensure_output_dirs()
    frames = load_model_summaries()

    missing = [m for m, df in frames.items() if df.empty]
    if missing:
        named = ", ".join(f"{m} ({MODEL_ALIASES[m]})" for m in missing)
        steps = "\n".join(
            f"  python run_pipeline.py simulate  --model {m}\n"
            f"  python run_pipeline.py summarize --model {m}" for m in missing)
        raise SystemExit(
            f"Cannot build the comparison: no summary yet for {named}.\n"
            f"Run these first:\n{steps}")

    wide_path = os.path.join(COMPARISON_DIR, "comparison_wide.csv")
    long_path = os.path.join(COMPARISON_DIR, "comparison_long.csv")

    wide = compare_model_performance(
        full_model_summary_df=frames["full"],
        myopic_model_summary_df=frames["myopic"],
        uniform_model_summary_df=frames["uniform"],
        baseline_model_summary_df=frames["baseline"],
        output_csv_path=wide_path,
    )
    if wide.empty:
        print("Comparison is empty; skipping the long table.")
        return wide_path, None

    create_long_format_df(
        full_model_summary_df=frames["full"],
        myopic_model_summary_df=frames["myopic"],
        uniform_model_summary_df=frames["uniform"],
        baseline_model_summary_df=frames["baseline"],
        output_csv_path=long_path,
    )
    print(f"comparison tables -> {wide_path}, {long_path}")
    return wide_path, long_path
