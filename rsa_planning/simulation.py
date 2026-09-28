"""Dynamic RSA simulation.

PlanningSimulation: full, uniform. GreedySimulation: myopic, baseline.
"""

import ast
import datetime
import gc
import json
import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.special import softmax

from .paths import (BFS_PRECOMPUTED_SEQ_DIR_PATH, DEFAULT_HORIZON,
                    OUTPUT_DIR, run_dir)
from .properties import PROPERTY_SPECS_DICT
from .rsa import colnorm, column_vector, safelog

class ModeManager:
    def __init__(self):

        self.baseline_mode = False

        self.silent_mode = True
        self.silent_mode_only_block = False

        self.multistep_mode = True

        # Is Dynamic -- True
        self.dynamic_states_mode = True

        # Binary States: set False; when Non-Binary set True
        self.non_binary_states_mode = True

        # UNiFORM priors -- baseline model comparison
        self.uniform_priors_mode = False

        if self.baseline_mode:
            self.multistep_mode = False

        # deprecated:
        self.costs_mode = False
        self.belief_delta_mode = False  # will not go ahead
        self.weighted_rewards_mode = False  # will not go ahead

class DynamicRSASimulation:
    """Pragmatic core shared by all four model variants."""

    def _load_data(self, lexicon_file, utterance_map_file):

        lexicon_data = pd.read_csv(lexicon_file, index_col=0)
        self.lexicon = lexicon_data.values
        self.utterances = lexicon_data.index.tolist()
        self.properties = lexicon_data.columns.tolist()
        self.prior = [0.001] * len(self.properties)
        self.costs = [0] * len(self.utterances)
        self.utterance_map = pd.read_csv(utterance_map_file, index_col='Messages')

    def _display(self, matrix, shape="Listener", name="Matrix"):
        if shape == "Listener":
            df = pd.DataFrame(matrix, index=self.properties, columns=self.utterances)
            print(f"\n{name}:")
            print(df, "\n")
        elif shape == "Speaker":
            df = pd.DataFrame(matrix, index=self.utterances, columns=self.properties)
            print(f"\n{name}:")
            print(df, "\n")
        elif shape == "FullBelief":
            print("\nThe full estimated Beliefs 'matrix':")
            for utterance, belief in matrix.items():
                print(f"utt: {utterance}")
                for prop, values in belief.items():
                    values_formatted = {state_value: f"{state_prob:.3f}" for state_value, state_prob in values.items()}
                    print(f"  {prop}: {values_formatted}")
        elif shape == "RewardTable":
            print(f"\n{name}:")
            print(self.reward_table_R, "\n")

    def _set_current_world_state(self):
        if self.mode.dynamic_states_mode:
            self.world_state_s = self.all_world_states[self.timestep_t]
        else:
            self.world_state_s = self.all_world_states

    def _set_criticality(self, use_scaled=True):

        if self.mode.non_binary_states_mode:
            criticality = {}
            for key, value in self.world_state_s.items():
                property_type = key.split("_")[1]
                prop_spec = self.property_specs_dict.get(property_type)

                # cluster error minimization
                if prop_spec is None:
                    criticality[key] = 0
                    continue

                critical_threshold = prop_spec.critical_start_scaled if use_scaled else prop_spec.critical_start_raw

                if prop_spec.prop_type == "binary":
                    criticality[key] = 1 if value == 0 else 0
                else:
                    if prop_spec.relation_sign == -1:  # Battery, Distance
                        criticality[key] = 1 if value <= critical_threshold else 0
                    elif prop_spec.relation_sign == +1:  # WindSpeed, Altitude
                        criticality[key] = 1 if value >= critical_threshold else 0
                    else:
                        criticality[key] = 0

            self.criticality_cr = criticality

        # when in binary state space
        # if not self.mode.non_binary_states_mode:
        else:
            self.criticality_cr = {
                prop: 1 if state_val == 0 else 0
                for prop, state_val in self.world_state_s.items()
            }

    def _set_priors(self):

        if self.mode.non_binary_states_mode:
            priors = []

            for entity_property, belief_distribution in self.human_belief_b.items():
                base_property = entity_property.split("_")[1]
                spec = self.property_specs_dict.get(base_property)

                if spec is None:
                    prior = 0.0001
                elif spec.prop_type == "continuous":
                    critical_threshold = spec.critical_start_scaled
                    if spec.relation_sign == -1:
                        prior = sum(prob for value, prob in belief_distribution.items() if value <= critical_threshold)
                    elif spec.relation_sign == 1:
                        prior = sum(prob for value, prob in belief_distribution.items() if value >= critical_threshold)
                    else:
                        prior = 0.0001

                elif spec.prop_type == "binary":
                    prior = belief_distribution.get(spec.critical_start_scaled, 0.0001)

                priors.append([prior])

            self.prior = column_vector(priors)

        else:
            priors = []
            for prop in self.properties:
                if prop in self.human_belief_b:
                    prior = self.human_belief_b[prop].get(0)
                else:
                    prior = 0.0001
                priors.append(prior)

            self.prior = column_vector(priors)

    def _set_costs(self):

        if self.mode.costs_mode:
            costs = [5,2,1,3,4,1,2,3,4]
            self.costs = costs

        self.costs = column_vector(self.costs)

    def _remove_silent_utterances(self):

        if self.mode.silent_mode_only_block:
            silent_actions = ["(XXX)"]
        else:
            silent_actions = ["(XXX)", "(...)"]
        self.utterances = list(filter(lambda ut: ut not in silent_actions, self.utterances))

    def _literal_speaker(self):
        alpha = 1
        utilities = alpha * (safelog(self.lexicon) - self.costs)
        literal_speaker_matrix = np.exp(utilities)
        self.S0 = colnorm(literal_speaker_matrix)

    def _pragmatic_listener(self):
        lit_spk = self.S0.T
        pragmatic_listener_matrix = lit_spk * self.prior
        self.L1 = colnorm(pragmatic_listener_matrix)
        return self.L1

    def run_rsa_S0_L1(self):

        self._set_current_world_state()
        self._set_criticality()
        # print(self.criticality_cr)
        if self.mode.silent_mode:
            self._remove_silent_utterances()
        self._set_priors()
        if self.mode.uniform_priors_mode:
            priors = [0.5] * len(self.properties)
            self.prior = column_vector(priors)
        self._set_costs()

        if self.timestep_t == 1:
            self._literal_speaker()
            # self._display(self.S0, shape="Speaker", name="S0 Distribution")

        self.L1 = self._pragmatic_listener()
        # self._display(self.L1, shape="Listener", name="L1 Distribution")

        return self.L1

    def _add_silent_utterances(self):

        if self.mode.silent_mode_only_block:
            self.utterances.extend(["(XXX)"])
        else:
            self.utterances.extend(["(XXX)", "(...)"])

        n_p = len(self.properties)
        block_actions = np.zeros((n_p, 1))
        silence_actions = np.full((n_p, 1), self.att_on_screen / n_p)

        if self.mode.silent_mode_only_block:
            silent_actions = block_actions
        else:
            silent_actions = np.hstack((block_actions, silence_actions))

        if self.attention_at is None:
            self.attention_at = silent_actions
        else:
            self.attention_at = np.hstack((self.attention_at, silent_actions))

    def AT(self, at_type=2):

        L1_matrix_copy = np.nan_to_num(self.L1.copy(), nan=0.0)

        zero_cols = np.all(L1_matrix_copy == 0.0, axis=0)
        if np.any(zero_cols):
            for col_idx in range(L1_matrix_copy.shape[1]):
                if np.all(L1_matrix_copy[:, col_idx] == 0.0):
                    max_prop_index = np.where(self.S0[col_idx, :] == self.S0[col_idx, :].max())[0]
                    L1_matrix_copy[max_prop_index, col_idx] = 1
        L1_matrix_copy = colnorm(L1_matrix_copy)

        if at_type == 1:
            # given I hear an utterance, which is the property that I will think of?
            # so, for each utterance, which property is it that has the max prob. mass
            attention_matrix = np.zeros_like(L1_matrix_copy)
            for col_idx in range(L1_matrix_copy.shape[1]):
                max_indices = np.where(L1_matrix_copy[:, col_idx] == L1_matrix_copy[:, col_idx].max())[0]
                attention_matrix[max_indices, col_idx] = 1
            self.attention_at = attention_matrix

        elif at_type == 2:
            self.attention_at = L1_matrix_copy

        elif at_type == 3:
            self.attention_at = np.apply_along_axis(softmax, axis=0, arr=L1_matrix_copy)

        if self.mode.silent_mode:
            self._add_silent_utterances()

        return self.attention_at

    def _set_closest_bin(self, value, bins):
        bins_array = np.array(list(bins))
        closest_idx = np.argmin(np.abs(bins_array - value))
        return bins_array[closest_idx]

    def _get_closest_bin(self):
        closest_bins = {}
        for prop in self.properties:
            beliefs = self.human_belief_b.get(prop, {})
            state_val = self.world_state_s.get(prop)
            if beliefs and state_val is not None:
                closest_bins[prop] = self._set_closest_bin(state_val, beliefs.keys())

        return closest_bins

    def B(self):

        new_beliefs_hat = {
            utterance: {
                property: {val: 0.0 for val in beliefs.keys()}
                for property, beliefs in self.human_belief_b.items()
            }
            for utterance in self.utterances
        }

        if self.mode.non_binary_states_mode:
            closest_bins = self._get_closest_bin()
            for utt_i, utterance in enumerate(self.utterances):
                for prop_j, property in enumerate(self.properties):

                    beliefs = self.human_belief_b.get(property, {})
                    closest_bin = closest_bins.get(property)
                    attention_p = self.attention_at[prop_j, utt_i]

                    if closest_bin is None:
                        continue

                    # print("utterance, property, value, closest_bin, prior_belief, prob_of_state, updated_prob")
                    for value, prior_belief in beliefs.items():
                        prob_of_state = 1.0 if closest_bin == value else 0.0
                        updated_prob = round(((attention_p * prob_of_state) + (1 - attention_p) * prior_belief), 5)
                        # print(utterance, property, value, closest_bin, prior_belief, prob_of_state, updated_prob)
                        # print(f'({attention_p} * {prob_of_state}) + ({1 - attention_p}) * {prior_belief})')
                        new_beliefs_hat[utterance][property][value] = updated_prob

            self.full_belief_matrix_B = new_beliefs_hat
            return new_beliefs_hat


        else:

            for utt_i, utterance in enumerate(self.utterances):
                for prop_j, property in enumerate(self.properties):

                    beliefs = self.human_belief_b.get(property, {})
                    state_value_p = self.world_state_s.get(property)
                    attention_p = self.attention_at[prop_j, utt_i]

                    for value, prior_belief in beliefs.items():
                        prob_of_state = 1.0 if state_value_p == value else 0.0
                        updated_prob = round(((attention_p * prob_of_state) + (1 - attention_p) * prior_belief),
                                             5)
                        new_beliefs_hat[utterance][property][value] = updated_prob

            self.full_belief_matrix_B = new_beliefs_hat
            return new_beliefs_hat

    def _calculate_reward(self, belief_value, cr_p, weight):

        critical_reward = belief_value * (cr_p + weight)
        # non_critical_reward = belief_value * (1 - cr_p)
        # for each property so if its a cr - sum of cr; else sum of non-cr
        # return critical_reward + non_critical_reward
        return critical_reward

    def _generate_sequence_tree(self):

        self.tree = {t: [] for t in range(self.time_horizon + 1)}
        queue = deque([(0, 0, [])])

        while queue:
            current_pos, req_blocks, current_seq = queue.popleft()
            self.tree[current_pos].append(current_seq.copy())

            if current_pos == self.time_horizon:
                continue

            if req_blocks > 0:
                new_seq = current_seq.copy()
                new_seq.append("(XXX)")
                queue.append((current_pos + 1, req_blocks - 1, new_seq))
            else:
                for msg in self.utterance_map.index:
                    if msg == "(XXX)":
                        continue
                    time = self.utterance_map.loc[msg, "Time"]
                    if current_pos + time > self.time_horizon:
                        continue
                    new_seq = current_seq.copy()
                    new_seq.append(msg)
                    queue.append((current_pos + 1, time - 1, new_seq))

    def _run_model_for_root(self):
        parent_seq = tuple()
        if parent_seq not in self.parent_belief_cache:
            self.human_belief_b = self.human_belief_b
            self.run_rsa_S0_L1()
            self.AT(at_type=2)
            estimated_beliefs = self.B()
            rewards = self.reward_table()
            rewards = rewards['summed_rewards_t'].to_dict()
            self.parent_belief_cache[parent_seq] = estimated_beliefs
            self.parent_reward_cache[parent_seq] = rewards

    def _run_model_for_parent(self, parent_seq):

        if parent_seq not in self.parent_belief_cache:
            self.human_belief_b = self.parent_belief_cache.get(parent_seq, self.human_belief_b)
            self.run_rsa_S0_L1()
            self.AT(at_type=2)
            estimated_beliefs = self.B()
            rewards = self.reward_table()
            rewards = rewards['summed_rewards_t'].to_dict()
            self.parent_belief_cache[parent_seq] = estimated_beliefs
            self.parent_reward_cache[parent_seq] = rewards

    def _get_current_belief(self, parent_seq, current_msg):
        return self.parent_belief_cache[parent_seq][current_msg]

    def _get_current_reward(self, parent_seq, current_msg):
        # return self.parent_reward_cache[parent_seq].loc[current_msg, 'summed_rewards_t']
        return self.parent_reward_cache[parent_seq].get(current_msg, 0)

    def search(self):
        raise NotImplementedError


class PlanningSimulation(DynamicRSASimulation):
    """Finite-horizon planning over the pruned sequence tree (full, uniform)."""

    def __init__(self, tc_id, lexicon_file, utterance_map_file, all_world_states,
                 initial_human_belief, output_file, time_horizon=DEFAULT_HORIZON):

        self.time_horizon = time_horizon
        self._load_data(lexicon_file, utterance_map_file)
        self._initialize_simulation(tc_id, all_world_states, initial_human_belief, output_file)
        self._initialize_timestep()
        self.mode = ModeManager()

    def _initialize_simulation(self, tc_id, all_world_states, initial_human_belief, output_file):

        self.num_drones = 4
        self.num_properties = 6
        self.property_specs_dict = PROPERTY_SPECS_DICT

        self.tc_id = tc_id
        self.all_world_states = all_world_states
        self.human_belief_b = initial_human_belief
        self.output_file = output_file
        self.time_horizon = getattr(self, "time_horizon", DEFAULT_HORIZON)
        self.att_on_screen = 0.5
        self.rewards_over_time_df = pd.DataFrame()

        # self.tree = {t: [] for t in range(self.time_horizon + 1)}
        self.utt_seq_tree_data = defaultdict(list)
        self.parent_belief_cache = {}
        self.parent_reward_cache = {}
        self.belief_cache = {}
        self.reward_cache = {}

        self.results = []
        self.results_df = pd.DataFrame()

        self.intermediate_sequence_output_dir = self._setup_intermediate_output_dir()
        intermediate_filename_base = os.path.basename(output_file).replace('.xlsx', '')
        self.intermediate_results_jsonl_path = os.path.join(
            self.intermediate_sequence_output_dir, f"{intermediate_filename_base}_result_tree.jsonl")

        if os.path.exists(self.intermediate_results_jsonl_path):
            os.remove(self.intermediate_results_jsonl_path)

    def _initialize_timestep(self):

        self.S0 = None
        self.L1 = None
        self.S2 = None
        self.timestep_t = 1
        self.world_state_s = None
        self.attention_at = None
        self.full_belief_matrix_B = None
        self.criticality_cr = None
        self.reward_table_R = None
        self.selected_utterance = ""
        self.utterance_history = []
        self.visited_sequences = set()
        self.previous_up_pair_seq = []

    def _setup_intermediate_output_dir(self):

        job_output_base = os.getenv('JOB_OUTPUT_DIR')

        if job_output_base:
            output_dir = os.path.join(job_output_base, "result_tree_dump")
        else:
            local_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(OUTPUT_DIR, "job_output", local_run_timestamp, "result_tree_dump")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def reward_table(self):

        reward_matrix = np.zeros_like(self.attention_at.T, dtype=float)
        if self.mode.non_binary_states_mode:
            closest_bins = self._get_closest_bin()

        for prop_i, property in enumerate(self.properties):

            if self.mode.non_binary_states_mode:
                closest_bin = closest_bins.get(property)
                # print(f"Property: {property}, Closest Bin: {closest_bin}")
            else:
                state_value_p = self.world_state_s.get(property)

            cr_p = self.criticality_cr.get(property)

            for utt_j, utterance in enumerate(self.utterances):
                if self.mode.non_binary_states_mode:
                    current_belief_prob = self.full_belief_matrix_B[utterance][property][closest_bin]
                    prior_belief_prob = self.human_belief_b[property][closest_bin]
                else:
                    current_belief_prob = self.full_belief_matrix_B[utterance][property][state_value_p]
                    prior_belief_prob = self.human_belief_b[property][state_value_p]

                if self.mode.belief_delta_mode:
                    delta_belief = current_belief_prob - prior_belief_prob
                    belief_value = delta_belief
                else:
                    belief_value = current_belief_prob
                    # belief_value = 1 - current_belief_prob

                if self.mode.weighted_rewards_mode:
                    reward = self._calculate_reward(belief_value, cr_p, weight = 2)
                else:
                    reward = belief_value * cr_p

                reward_matrix[utt_j, prop_i] = reward

        sum_rewards_utt = np.sum(reward_matrix, axis=1)
        rewards_df = pd.DataFrame(reward_matrix, index=self.utterances, columns=self.properties)
        rewards_df["summed_rewards_t"] = sum_rewards_utt
        self.reward_table_R = rewards_df
        # print(rewards_df)
        return self.reward_table_R

    def _force_clean_grandparents(self, current_t):

        # which generation to remove after processing current?
        # - 1 parent after processing, -2 grandparents

        remove_gen = current_t - 1

        # only remove if we're at least at timestep <=1
        if remove_gen < 0:
            return

        print(f"TC: {self.tc_id}; cleaning parent: {remove_gen}")


        for cache in [self.belief_cache, self.reward_cache,
                      self.parent_belief_cache, self.parent_reward_cache]:

            keys_to_remove = [k for k in cache.keys() if len(k) <= remove_gen]

            for key in keys_to_remove:
                # print(key)
                del cache[key]

        gc.collect()

    def _force_parent_process(self, parent_seq):
        if not parent_seq:
            if parent_seq not in self.parent_belief_cache:
                self.parent_belief_cache[parent_seq] = {}
                self.parent_reward_cache[parent_seq] = {}
            return

        if parent_seq not in self.parent_belief_cache:
            # when (xxx) (XXX) we check farther back?
            grandparent_seq = parent_seq[:-1]
            self._force_parent_process(grandparent_seq)

    def _process_sequence(self, parent_seq, new_seq, current_msg, cost, legality):

        seq_key = tuple(new_seq)

        # STOP FOR long runs
        # print(f'processing {seq_key}')

        # current_belief = {}
        # current_reward = 0

        if len(new_seq) == 1:
            parent_reward = 0
        else:
            parent_reward = self.reward_cache[parent_seq]

        if parent_seq not in self.parent_belief_cache:
            self.human_belief_b = self.belief_cache[parent_seq]
            self._run_model_for_parent(parent_seq)


        current_belief = self._get_current_belief(parent_seq, current_msg)
        current_reward = self._get_current_reward(parent_seq, current_msg)
        self.belief_cache[seq_key] = current_belief
        self.reward_cache[seq_key] = current_reward + parent_reward

        if legality:
            result_data = {
                'timestep': len(new_seq),
                'utterance_sequence': seq_key,
                'total_reward': parent_reward + current_reward,
                'cost': cost,
                'world_state': self.world_state_s,
                'legality': legality,
                'beliefs': current_belief,
                'current_reward': current_reward,
            }

            with open(self.intermediate_results_jsonl_path, 'a') as f:
                f.write(json.dumps(result_data) + '\n')

    def normed_ranked_results(self):
        # results_df = pd.DataFrame(self.results)
        print("Entering Normed Results")
        results_list = []
        if os.path.exists(self.intermediate_results_jsonl_path):
            try:
                with open(self.intermediate_results_jsonl_path, 'r') as f:
                    for line in f:
                        results_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error reading JSONL file {self.intermediate_results_jsonl_path}: {e}")
                return pd.DataFrame()
        else:
            print(f"Temporary results file not found: {self.intermediate_results_jsonl_path}")

        results_df = pd.DataFrame(results_list)

        if results_df.empty:
            return results_df

        results_df['utterance_sequence'] = results_df['utterance_sequence'].apply(tuple)

        # norm rewards (global)
        min_reward = results_df['total_reward'].min()
        max_reward = results_df['total_reward'].max()
        denominator = max_reward - min_reward if max_reward != min_reward else 1
        results_df['norm_rewards'] = (results_df['total_reward'] - min_reward) / denominator

        # temp key for ranks
        results_df['_composite_key'] = (
                results_df['norm_rewards'].round(10).astype(str) +
                '_' +
                results_df['cost'].astype(str)
        )

        sorted_df = results_df.sort_values(
            ['norm_rewards', 'cost'],
            ascending=[False, True]
        )

        sorted_df['rank'] = sorted_df.groupby('_composite_key', sort=False).ngroup() + 1

        ranked_df = results_df.merge(
            sorted_df[['utterance_sequence', 'rank']],
            on='utterance_sequence',
            how='left'
        )

        return ranked_df.sort_values('rank').drop(columns=['_composite_key'])

    def prepare_op_file(self):

        results_df = self.normed_ranked_results()
        print("Before uniform process: \n",results_df.head())

        if self.mode.uniform_priors_mode:
            results_df = self.uniform_model_file_process(results_df)
            print("After uniform process: \n",results_df.head())
        try:
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            writer = pd.ExcelWriter(self.output_file, engine='openpyxl')
            results_df.to_excel(writer, index=False)
            writer.close()
            return True
        except Exception:
            try:
                # self.rewards_over_time_df.to_csv(self.output_file + ".csv", index=False)
                results_df.to_csv(self.output_file + ".csv", index=False)
                return True
            except Exception:
                return False

    def uniform_model_file_process(self, uniform_model_df):

        uniform_model_df = uniform_model_df.copy()
        # print(f"DEBUG: self.output_file = {self.output_file}")
        tc_id_match =  re.search(r'result_(dyn_\d+_\d+_\d+)_uniform\.xlsx$', self.output_file)
        tc_id = tc_id_match.group(1)
        # print(f"Extracted TC ID: {tc_id}")

        full_model_path = os.path.join(
            run_dir("full"),
            os.path.basename(self.output_file).replace("_uniform.xlsx", "_full.xlsx"))
        if os.path.exists(full_model_path):
            # print(f"full_model_path = {full_model_path}")
            try:
                full_model_df = pd.read_excel(full_model_path, engine='openpyxl')
                full_model_df['utterance_sequence'] = full_model_df['utterance_sequence'].apply(ast.literal_eval)
                full_model_reward_map = full_model_df.set_index('utterance_sequence')['total_reward'].to_dict()
                # print(full_model_reward_map)
                uniform_model_df['full_model_rewards'] = \
                    (uniform_model_df['utterance_sequence'].apply(lambda x: full_model_reward_map.get(x, 0.0)))
                # print(uniform_model_df.head())
            except Exception as e:
                uniform_model_df['full_model_rewards'] = 0.0
        else:
            print(f"Full model file not run/found: {full_model_path}")
            uniform_model_df['full_model_rewards'] = 0.0

        if 'full_model_rewards' in uniform_model_df.columns:
            uniform_model_df.rename(columns={
                'total_reward': 'uniform_priors_reward',
                'full_model_rewards': 'total_reward'}, inplace=True)

        result_df_final = uniform_model_df
        return result_df_final

    def search(self):
        self._run_model_for_root()

        precomputed_path = f"{BFS_PRECOMPUTED_SEQ_DIR_PATH}/precomputed_H_{self.time_horizon}.csv"
        precomputed_df = pd.read_csv(precomputed_path)
        self.utt_seq_tree_data.clear()

        # group sequences by timestep
        sequences_by_timestep = {}
        for i, row in precomputed_df.iterrows():
            t = int(row['timestep'])
            utt_seq = ast.literal_eval(row['sequence'])
            cost = float(row['cost'])
            legality = int(row['legality'])
            sequences_by_timestep.setdefault(t, []).append((utt_seq, cost, legality))

        # process by timestep in order (bfs)
        for t in sorted(sequences_by_timestep.keys()):
            print(f"TC: {self.tc_id} starting to process: time {t}")
            self.timestep_t = t
            for seq, cost, legality in sequences_by_timestep[t]:
                parent_seq = tuple(seq[:-1])
                current_msg = seq[-1]
                self._process_sequence(parent_seq, seq, current_msg, cost, legality)
                # print(f"TC: {self.tc_id} end of process: time {t}")
            self._force_clean_grandparents(t)

class GreedySimulation(DynamicRSASimulation):
    """Greedy single-timestep selection (myopic, baseline)."""

    def __init__(self, lexicon_file, utterance_map_file, all_world_states,
                 initial_human_belief, output_file, time_horizon=DEFAULT_HORIZON):

        self.time_horizon = time_horizon
        self._load_data(lexicon_file, utterance_map_file)
        self._initialize_simulation(all_world_states, initial_human_belief, output_file)
        self._initialize_timestep()
        self.mode = ModeManager()

    def _initialize_simulation(self, all_world_states, initial_human_belief, output_file):

        self.num_drones = 4
        self.num_properties = 6
        self.property_specs_dict = PROPERTY_SPECS_DICT

        self.all_world_states = all_world_states
        self.human_belief_b = initial_human_belief
        self.output_file = output_file

        self.time_horizon = getattr(self, "time_horizon", DEFAULT_HORIZON)
        self.att_on_screen = 0.5
        self.rewards_over_time_df = pd.DataFrame()

        # self.tree = {t: [] for t in range(self.time_horizon + 1)}
        self.utt_seq_tree_data = defaultdict(list)
        self.parent_belief_cache = {}
        self.parent_reward_cache = {}
        self.belief_cache = {}
        self.reward_cache = {}
        self.greedy_sequence = tuple()

        self.results = []
        self.results_df = pd.DataFrame()

    def _initialize_timestep(self):

        self.S0 = None
        self.L1 = None
        self.S2 = None
        self.timestep_t = 1
        self.world_state_s = None
        self.attention_at = None
        self.full_belief_matrix_B = None
        self.criticality_cr = None
        self.reward_table_R = None
        self.selected_utterance = ""
        self.utterance_history = []
        self.visited_sequences = set()
        self.processed_sequences = set()
        self.previous_up_pair_seq = []

    def reward_table(self):

        reward_matrix = np.zeros_like(self.attention_at.T, dtype=float)
        if self.mode.non_binary_states_mode:
            closest_bins = self._get_closest_bin()

        for prop_i, property in enumerate(self.properties):

            if self.mode.non_binary_states_mode:
                closest_bin = closest_bins.get(property)
            else:
                state_value_p = self.world_state_s.get(property)

            cr_p = self.criticality_cr.get(property)

            for utt_j, utterance in enumerate(self.utterances):

                if self.mode.non_binary_states_mode:
                    current_belief_prob = self.full_belief_matrix_B[utterance][property][closest_bin]
                    prior_belief_prob = self.human_belief_b[property][closest_bin]
                else:
                    current_belief_prob = self.full_belief_matrix_B[utterance][property][state_value_p]
                    prior_belief_prob = self.human_belief_b[property][state_value_p]

                if self.mode.belief_delta_mode:
                    delta_belief = current_belief_prob - prior_belief_prob
                    belief_value = delta_belief
                else:
                    belief_value = current_belief_prob
                    # belief_value = 1 - current_belief_prob

                if self.mode.weighted_rewards_mode:
                    reward = self._calculate_reward(belief_value, cr_p, weight = 2)
                else:
                    reward = belief_value * cr_p

                reward_matrix[utt_j, prop_i] = reward

        sum_rewards_utt = np.sum(reward_matrix, axis=1)
        rewards_df = pd.DataFrame(reward_matrix, index=self.utterances, columns=self.properties)
        rewards_df["summed_rewards_t"] = sum_rewards_utt
        self.reward_table_R = rewards_df
        # print(rewards_df)
        return self.reward_table_R

    def _force_parent_process(self, parent_seq):
        if not parent_seq:
            if parent_seq not in self.parent_belief_cache:
                self.parent_belief_cache[parent_seq] = {}
                self.parent_reward_cache[parent_seq] = {}
            return

        if parent_seq not in self.parent_belief_cache:
            # when (xxx) (XXX) we check farther back?
            grandparent_seq = parent_seq[:-1]
            self._force_parent_process(grandparent_seq)

    def _process_sequence(self, parent_seq, new_seq, current_msg, cost, legality):

        seq_key = tuple(new_seq)
        # print(f'processing {seq_key}')
        # current_belief = {}
        # current_reward = 0

        if len(new_seq) == 1:
            parent_reward = 0
        else:
            parent_reward = self.reward_cache[parent_seq]

        if parent_seq not in self.parent_belief_cache:
            self.human_belief_b = self.belief_cache[parent_seq]
            self._run_model_for_parent(parent_seq)

            # if current_msg == "(XXX)" and parent_seq in self.belief_cache:
            #     # there can't be any other msg that follows the parent_seq because it needs a block -- so we don't need the full beliefs
            #     self.parent_belief_cache[parent_seq], current_belief = self.belief_cache[parent_seq]
            #     self.parent_reward_cache[parent_seq], current_reward = self.rewards_cache[parent_seq]


        current_belief = self._get_current_belief(parent_seq, current_msg)
        current_reward = self._get_current_reward(parent_seq, current_msg)
        self.belief_cache[seq_key] = current_belief
        self.reward_cache[seq_key] = current_reward + parent_reward

        # if legality:
        if seq_key not in self.processed_sequences:
            # print(f"Processing {seq_key}...")
            self.results.append({
                'timestep': len(new_seq),
                'utterance_sequence': seq_key,
                'total_reward': parent_reward + current_reward,
                'cost': cost,
                'world_state': self.world_state_s,
                'legality': legality,
                'beliefs': current_belief,
                'current_reward': current_reward,
            })
            self.processed_sequences.add(seq_key)

    def normed_ranked_results(self):
        results_df = pd.DataFrame(self.results)

        if results_df.empty:
            return results_df

        results_df['utterance_sequence'] = results_df['utterance_sequence'].apply(tuple)

        # norm rewards (global)
        min_reward = results_df['total_reward'].min()
        max_reward = results_df['total_reward'].max()
        denominator = max_reward - min_reward if max_reward != min_reward else 1
        results_df['norm_rewards'] = (results_df['total_reward'] - min_reward) / denominator

        # temp key for ranks
        results_df['_composite_key'] = (
                results_df['norm_rewards'].round(10).astype(str) +
                '_' +
                results_df['cost'].astype(str)
        )

        sorted_df = results_df.sort_values(
            ['norm_rewards', 'cost'],
            ascending=[False, True]
        )

        sorted_df['rank'] = sorted_df.groupby('_composite_key', sort=False).ngroup() + 1

        ranked_df = results_df.merge(
            sorted_df[['utterance_sequence', 'rank']],
            on='utterance_sequence',
            how='left'
        )

        return ranked_df.sort_values('rank').drop(columns=['_composite_key'])

    def prepare_op_file(self):
        results_df = self.normed_ranked_results()
        if self.mode.uniform_priors_mode:
            results_df = self.uniform_model_file_process(results_df)
        try:
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            writer = pd.ExcelWriter(self.output_file, engine='openpyxl')
            results_df.to_excel(writer, index=False)
            writer.close()
            return True
        except Exception:
            try:
                # self.rewards_over_time_df.to_csv(self.output_file + ".csv", index=False)
                results_df.to_csv(self.output_file + ".csv", index=False)
                return True
            except Exception:
                return False

    def uniform_model_file_process(self, baseline_df):

        baseline_df = baseline_df.copy()
        # print(f"DEBUG: self.output_file = {self.output_file}")
        tc_id_match =  re.search(r'result_(dyn_\d+_\d+_\d+)_baseline\.xlsx$', self.output_file)
        tc_id = tc_id_match.group(1)
        # print(f"Extracted TC ID: {tc_id}")

        full_model_path = os.path.join(
            run_dir("full"),
            os.path.basename(self.output_file).replace("_baseline.xlsx", "_full.xlsx"))
        if os.path.exists(full_model_path):
            # print(f"full_model_path = {full_model_path}")
            try:
                full_model_df = pd.read_excel(full_model_path, engine='openpyxl')
                full_model_df['utterance_sequence'] = full_model_df['utterance_sequence'].apply(ast.literal_eval)
                full_model_reward_map = full_model_df.set_index('utterance_sequence')['total_reward'].to_dict()
                # print(full_model_reward_map)
                baseline_df['full_model_rewards'] = \
                    (baseline_df['utterance_sequence'].apply(lambda x: full_model_reward_map.get(x, 0.0)))
                # print(uniform_model_df.head())
            except Exception as e:
                baseline_df['full_model_rewards'] = 0.0
        else:
            print(f"Full model file not run/found: {full_model_path}")
            baseline_df['full_model_rewards'] = 0.0

        if 'full_model_rewards' in baseline_df.columns:
            baseline_df.rename(columns={
                'total_reward': 'uniform_priors_reward',
                'full_model_rewards': 'total_reward'}, inplace=True)

        result_df_final = baseline_df
        return result_df_final

    def search(self):

        self._run_model_for_root()

        precomputed_path = f"{BFS_PRECOMPUTED_SEQ_DIR_PATH}/precomputed_H_{self.time_horizon}.csv"
        precomputed_df = pd.read_csv(precomputed_path)
        self.utt_seq_tree_data.clear()

        for i, row in precomputed_df.iterrows():
            t = int(row['timestep'])
            utt_seq = ast.literal_eval(row['sequence'])
            if isinstance(utt_seq, list):
                utt_seq = tuple(utt_seq)
            cost = float(row['cost'])
            legality = int(row['legality'])
            self.utt_seq_tree_data[t].append((utt_seq, cost, legality))

        # greedy path start
        self.greedy_sequence = tuple()

        for t in range(1, self.time_horizon + 1):

            self.timestep_t = t
            current_timestep_candidates = []

            sequences_at_current_t = [
                (seq, cost, legality) for seq, cost, legality in self.utt_seq_tree_data[t]
                if len(seq) == t and tuple(seq[:-1]) == self.greedy_sequence
            ]

            if not sequences_at_current_t and t > 1:
                # print(f"Time {t}: No more paths to explore for the greedy sequence '{self.greedy_sequence}'. Stopping.")
                break

            for seq_tuple, cost, legality in sequences_at_current_t:
                parent_seq = tuple(seq_tuple[:-1])
                current_msg = seq_tuple[-1]

                self._process_sequence(parent_seq, seq_tuple, current_msg, cost, legality)
                total_reward_for_candidate = self.reward_cache.get(seq_tuple, -float('inf'))

                # if total_reward_for_candidate != -float('inf') and legality:
                # reminder: won't need to check for legality here because the pre-computed files have only the final seqs that are legal
                if total_reward_for_candidate != -float('inf'):
                    current_timestep_candidates.append({
                        'utterance_sequence': seq_tuple,
                        'total_reward': total_reward_for_candidate,
                        'cost': cost
                    })

            if not current_timestep_candidates:
                # print(f"Time {t}: No legal or valid candidate sequences found. Stopping.")
                break


            candidates_df = pd.DataFrame(current_timestep_candidates)

            if candidates_df.empty:
                # print(f"Time {t}: Candidates DataFrame is empty after filtering. Stopping.")
                break

            # min_reward = candidates_df['total_reward'].min()
            # max_reward = candidates_df['total_reward'].max()
            # denominator = max_reward - min_reward if max_reward != min_reward else 1
            # candidates_df['norm_rewards'] = (candidates_df['total_reward'] - min_reward) / denominator

            ranked_candidates = candidates_df.sort_values(
                by=['total_reward', 'cost'],
                ascending=[False, True]
            )

            ranked_candidates['rank'] = ranked_candidates.reset_index(drop=True).index + 1

            best_candidate_row = ranked_candidates.iloc[0]
            best_candidate_sequence = best_candidate_row['utterance_sequence']

            if best_candidate_sequence is not None:
                self.greedy_sequence = best_candidate_sequence
            else:
                # no optimal message - stop greedy path
                break

class SequencePrecomputer:
    def __init__(self, utterance_map, max_horizon=6):
        self.utterance_map = pd.read_csv(utterance_map, index_col='Messages')
        self.utterance_map['Time'] = pd.to_numeric(self.utterance_map['Time'])
        self.utterance_map['Cost'] = pd.to_numeric(self.utterance_map['Cost'])
        self.max_horizon = max_horizon
        self.tree = []

    def _generate_sequences(self, current_horizon = 1):

        self.tree = []
        queue = deque()
        queue.append((0, 0, []))  # (current_pos, required_blocks, current_seq)

        while queue:
            current_pos, req_blocks, current_seq = queue.popleft()

            if current_seq:
                self.tree.append(current_seq.copy())

            if current_pos == current_horizon:
                continue

            if req_blocks > 0:
                new_seq = current_seq.copy()
                new_seq.append("(XXX)")
                queue.append((current_pos + 1, req_blocks - 1, new_seq))
            else:
                for msg in self.utterance_map.index:
                    if msg == "(XXX)":
                        continue
                    time = self.utterance_map.loc[msg, "Time"]
                    if current_pos + time > current_horizon:
                        continue
                    new_seq = current_seq.copy()
                    new_seq.append(msg)
                    queue.append((current_pos + 1, time - 1, new_seq))

        return self.tree

    def _calculate_properties(self, seq):

        cost = sum(self.utterance_map.loc[msg, "Cost"]
                   for msg in seq if msg != "(XXX)")


        required_blocks = 0
        legal = True
        for idx, msg in enumerate(seq):
            if required_blocks > 0:
                if msg != "(XXX)":
                    legal = False
                    break
                required_blocks -= 1
            else:
                if msg == "(XXX)":
                    legal = False
                    break
                duration = self.utterance_map.loc[msg, "Time"]
                if idx + duration > len(seq):
                    legal = False
                    break
                required_blocks = duration - 1
        legal &= (required_blocks == 0)

        return cost, int(legal)

    def save_precomputed(self, file_pattern="precomputed_H_{}.csv"):

        for H in range(1, self.max_horizon + 1):

            print(f"precomputing for H={H}...")

            final_sequences = self._generate_sequences(H)
            # print(final_sequences)
            data_for_horizon = []

            for seq in final_sequences:
                cost, legal = self._calculate_properties(seq)
                data_for_horizon.append({
                    "timestep": len(seq),
                    "sequence": json.dumps(seq) if format == "csv" else seq,
                    "cost": cost,
                    "legality": legal
                })

            df = pd.DataFrame(data_for_horizon)
            df.to_csv(file_pattern.format(H), index=False)


def generate_uniform_belief(original_belief_dist):
    uniform_belief = {}
    for prop, values_dist in original_belief_dist.items():
        num_values = len(values_dist)
        if num_values == 0:
            uniform_belief[prop] = {}
            continue

        uniform_prob = 1.0 / num_values
        uniform_belief[prop] = {val: uniform_prob for val in values_dist.keys()}

    return uniform_belief

def create_subset_test_scenarios(master_csv_path, counter_range, num_critical_prop_values, user_critical_aware_values, world_mode):

    master_df = pd.read_csv(master_csv_path)

    master_df['num_critical_prop'] = master_df['num_critical_prop'].astype(str)
    master_df['user_critical_aware'] = master_df['user_critical_aware'].astype(str).str.replace('%', '')

    tc_type = ''
    if world_mode == "static":
        tc_type = 'stat'
    elif world_mode == "dynamic":
        tc_type = 'dyn'

    target_tc_ids = []
    for counter in counter_range:
        for num_cp in num_critical_prop_values:
            for aware_type in user_critical_aware_values:
                tc_id = f"{tc_type}_{counter}_{num_cp}_{aware_type}"
                target_tc_ids.append(tc_id)

    print(f"Total Num TCs: {len(target_tc_ids)}")
    subset_df = master_df[master_df['tc_id'].isin(target_tc_ids)].copy()

    selected_columns_df = subset_df[['tc_id', 'user_general_aware', 'all_scaled_world_states', 'user_belief_distribution']].copy()
    return selected_columns_df

def get_scenarios_by_tc_ids(master_csv_path, list_of_tc_ids):

    master_df = pd.read_csv(master_csv_path)
    filtered_df = master_df[master_df['tc_id'].isin(list_of_tc_ids)].copy()
    selected_columns_df = filtered_df[['tc_id', 'user_general_aware', 'all_scaled_world_states', 'user_belief_distribution']].copy()

    return selected_columns_df
