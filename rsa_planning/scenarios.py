"""Scenario and test case generation."""

import ast
import itertools
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from .paths import SCENARIO_DIR_PATH
from .worldgen import scale_world_state


def seed_scenario_generation(seed=None):
    """Seed the RNGs that drive scenario and belief generation."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


class PropertySpec:
    def __init__(self, spec_data):
        self.property_type = spec_data['property_type']
        self.min_range = spec_data['min_range']
        self.max_range = spec_data['max_range']
        self.prop_type = spec_data['type']
        self.critical_start_raw = spec_data['critical_start_raw']
        self.critical_start_scaled = spec_data['critical_start_scaled']
        self.relation_sign = spec_data['relation_sign']
        self.center_high_aware = spec_data.get('center_high_aware')
        self.center_low_aware = spec_data.get('center_low_aware')

    def __repr__(self):
        return f"PropertySpec(Type: {self.property_type}, PropType: {self.prop_type})"

PROPERTY_SPECS = [
    {"property_type": "Battery", "min_range": 0.1, "max_range": 100, "type": "continuous",
     "critical_start_raw": 10, "critical_start_scaled": 0.1, "relation_sign": -1,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)},

    {"property_type": "WindSpeed", "min_range": 0.1, "max_range": 22, "type": "continuous",
     "critical_start_raw": 18, "critical_start_scaled": 0.818, "relation_sign": +1,
     "center_high_aware": random.uniform(0.75, 1), "center_low_aware": random.uniform(0.01, 0.2)},

    {"property_type": "Rotor", "min_range": 0, "max_range": 1, "type": "binary",
     "critical_start_raw": 0, "critical_start_scaled": 0, "relation_sign": 0,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)},

    {"property_type": "Altitude", "min_range": 50, "max_range": 110, "type": "continuous",
     "critical_start_raw": 100, "critical_start_scaled": 0.833, "relation_sign": +1,
     "center_high_aware": random.uniform(0.75, 1), "center_low_aware": random.uniform(0.01, 0.2)},

    {"property_type": "NoFlyZone", "min_range": 0, "max_range": 1, "type": "binary",
     "critical_start_raw": 0, "critical_start_scaled": 0, "relation_sign": 0,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)},

    {"property_type": "Distance", "min_range": 0.1, "max_range": 100, "type": "continuous",
     "critical_start_raw": 10, "critical_start_scaled": 0.1, "relation_sign": -1,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)}
]

PROPERTY_SPECS_DICT = {spec["property_type"]: PropertySpec(spec) for spec in PROPERTY_SPECS}

class DroneScenarioGenerator:
    def __init__(self, num_drones, max_timesteps=10, num_critical_prop = 0):

        self.num_drones = num_drones
        property_specs_df = pd.DataFrame(PROPERTY_SPECS)
        self.property_specs_df = property_specs_df.set_index('property_type')
        self.all_properties = [f"Drone{d}_{p}"
                               for d in range(1, num_drones + 1)
                               for p in self.property_specs_df.index]
        self.max_timesteps = max_timesteps
        self.num_critical_prop = num_critical_prop
        self.safety_margin = 0.5

        self._setup_property_updaters()

    # --- Setting up + Update Functions for DRONE WORLD PROPERTIES ---
    def _setup_property_updaters(self):

        self.property_updaters = {
            'Battery': self._update_battery,
            'WindSpeed': self._update_wind_speed,
            'Rotor': self._update_rotor,
            'Altitude': self._update_altitude,
            'NoFlyZone': self._update_no_fly_zone,
            'Distance': self._update_distance
        }

    def _is_critical(self, prop_type, value):
        spec = self.property_specs_df.loc[prop_type]
        if spec['type'] == 'binary':
            # return value == spec['critical_start_scaled']
            return value == spec['critical_start_raw']

        if spec['relation_sign'] == -1:  # low (Battery, Distance)
            # return value <= spec['critical_start_scaled']
            return value <= spec['critical_start_raw']
        else:  # high values critical (WindSpeed, Altitude)
            # return value >= spec['critical_start_scaled']
            return value >= spec['critical_start_raw']

    def _update_battery(self, current_value_raw: float, prop_type_key: str) -> float:
        spec = self.property_specs_df.loc[prop_type_key]
        delta_raw = np.random.uniform(0.01, 0.51)
        new_value_raw = current_value_raw - delta_raw
        if self._is_critical(prop_type_key, current_value_raw):
            return round(np.clip(new_value_raw, spec['min_range'], spec['critical_start_raw']), 2)
        return round(np.clip(new_value_raw, spec['min_range'], spec['max_range']), 2)

    def _update_wind_speed(self, current_value_raw: float, prop_type_key: str) -> float:
        spec = self.property_specs_df.loc[prop_type_key]
        delta_raw = np.random.normal(0, 1.5)
        new_value_raw = current_value_raw + delta_raw
        if self._is_critical(prop_type_key, current_value_raw):
            return round(np.clip(new_value_raw, spec['critical_start_raw'], spec['max_range']), 2)
        return round(np.clip(new_value_raw, spec['min_range'], spec['max_range']), 2)

    def _update_rotor(self, current_value_raw: int, prop_type_key: str) -> int:
        return 0 if (current_value_raw == 1 and np.random.rand() < 0.01) else current_value_raw

    def _update_altitude(self, current_value_raw: float, prop_type_key: str) -> float:
        spec = self.property_specs_df.loc[prop_type_key]
        delta_raw = np.random.normal(0, 4)
        new_value_raw = current_value_raw + delta_raw
        if self._is_critical(prop_type_key, current_value_raw):
            return round(np.clip(new_value_raw, spec['critical_start_raw'], spec['max_range']), 2)
        return round(np.clip(new_value_raw, spec['min_range'], spec['max_range']), 2)

    def _update_no_fly_zone(self, current_value_raw: int, prop_type_key: str) -> int:
        if current_value_raw == 0:
            return 0
        return 0 if np.random.rand() < 0.01 else 1

    def _update_distance(self, current_value_raw: float, prop_type_key: str) -> float:
        spec = self.property_specs_df.loc[prop_type_key]
        delta_raw = np.random.uniform(0.1, 1.1)
        new_value_raw = current_value_raw - delta_raw
        if self._is_critical(prop_type_key, current_value_raw):
            return round(np.clip(new_value_raw, spec['min_range'], spec['critical_start_raw']), 2)
        return round(np.clip(new_value_raw, spec['min_range'], spec['max_range']), 2)

    # --- Helpers: non-critical value sampling ---

    def _sample_critical_value(self, prop_spec: pd.Series) -> float:
        if prop_spec['type'] == 'continuous':
            if prop_spec['relation_sign'] == -1:
                return round(random.uniform(prop_spec['min_range'], prop_spec['critical_start_raw']), 2)
            else:
                return round(random.uniform(prop_spec['critical_start_raw'], prop_spec['max_range']), 2)
        else:  # binary
            return prop_spec['critical_start_raw']

    def _sample_noncritical_value(self, prop_spec):

        min_val = prop_spec['min_range']
        max_val = prop_spec['max_range']
        critical_start = prop_spec['critical_start_raw']
        relation_sign = prop_spec['relation_sign']
        margin = self.safety_margin

        if relation_sign == -1: # cr: when value is lower
            lower_bound_nc = min(critical_start + margin, max_val)
            if lower_bound_nc >= max_val:
                lower_bound_nc = max_val - 0.01
            return round(np.random.uniform(lower_bound_nc, max_val), 2)

        elif relation_sign == +1: # cr: when value higher
            upper_bound_nc = max(critical_start - margin, min_val)
            if upper_bound_nc <= min_val:
                upper_bound_nc = min_val + 0.01
            return round(np.random.uniform(min_val, upper_bound_nc), 2)

        else: # Binary
            return 1 if critical_start == 0 else 0

    def _generate_initial_world_state(self, noncritical_only: bool) -> Dict[str, float]:
        world_state = {}
        for drone in range(1, self.num_drones + 1):
            for prop_type in self.property_specs_df.index:
                prop_spec = self.property_specs_df.loc[prop_type]
                drone_prop_key = f"Drone{drone}_{prop_type}"
                if noncritical_only:
                    world_state[drone_prop_key] = self._sample_noncritical_value(prop_spec)
                else:
                    if prop_spec['type'] == "continuous":
                        world_state[drone_prop_key] = round(
                            np.random.uniform(prop_spec['min_range'], prop_spec['max_range']), 2)
                    else:
                        world_state[drone_prop_key] = np.random.choice([0, 1])
        return world_state


    def _generate_world_state_with_critical(self, critical_props):

        world_state = self._generate_initial_world_state(noncritical_only=True)

        for prop_key in critical_props:
            prop_type_name = prop_key.split('_')[1]
            prop_spec = self.property_specs_df.loc[prop_type_name]

            if prop_spec['type'] == 'continuous':
                if prop_spec['relation_sign'] == -1:
                    world_state[prop_key] = round(np.random.uniform(
                        prop_spec['min_range'], prop_spec['critical_start_raw']
                    ), 2)
                else:
                    world_state[prop_key] = round(np.random.uniform(
                        prop_spec['critical_start_raw'], prop_spec['max_range']
                    ), 2)
            else:
                world_state[prop_key] = prop_spec['critical_start_raw'] # binary prop

        return world_state

    def generate_static_scenarios(self, num_scenario_per_cr_type = 50):
        scenarios = []

        for num_critical in [2, 3, 4]:
            all_combinations = list(itertools.combinations(self.all_properties, num_critical))

            if len(all_combinations) <= num_scenario_per_cr_type:
                selected_combinations = all_combinations
            else:
                #unique combinations w/o replacement
                selected_combinations = random.sample(all_combinations, num_scenario_per_cr_type)

            for i, critical_props in enumerate(selected_combinations, start=1):
                scenario_id = f"stat_{i}_{num_critical}"
                world_state = self._generate_world_state_with_critical(list(critical_props))

                scenarios.append({
                    'scenario_id': scenario_id,
                    'sc_type': 'static',
                    'num_critical_prop': num_critical,
                    'list_critical_prop': list(critical_props),
                    'world_state': world_state,
                    'scaled_world_states': scale_world_state(world_state, self.property_specs_df)

                })

        return pd.DataFrame(scenarios)


    # --- DYANAMIC SCENARIO GENERATION ---
    def _apply_critical_at_ts(self, naturally_updated_raw_state: Dict[str, float], permanently_critical_props: set) -> \
    Dict[str, float]:
        final_raw_state = {}
        for prop_key, raw_value_after_update in naturally_updated_raw_state.items():
            prop_type_name = prop_key.split('_')[1]
            prop_spec = self.property_specs_df.loc[prop_type_name]

            if prop_key in permanently_critical_props:
                if not self._is_critical(prop_type_name, raw_value_after_update):
                    final_raw_state[prop_key] = self._sample_critical_value(prop_spec)
                else:
                    final_raw_state[prop_key] = raw_value_after_update
            else:
                if self._is_critical(prop_type_name, raw_value_after_update):
                    final_raw_state[prop_key] = self._sample_noncritical_value(prop_spec)
                else:
                    final_raw_state[prop_key] = raw_value_after_update
        return final_raw_state
    def _assign_critical_timesteps_by_density(self, critical_props: List[str], max_timesteps: int):

        list_critical_prop_time = []
        if not critical_props:
            return [], 0

        current_timestep = random.randint(1, 3)
        density = random.randint(0, 3)

        for prop in critical_props:
            assigned_timestep = current_timestep
            if assigned_timestep > max_timesteps:
                assigned_timestep = max_timesteps

            list_critical_prop_time.append((prop, assigned_timestep))
            current_timestep += density

        return list_critical_prop_time, density

    def _simulate_scenario_timesteps(self, list_critical_prop_time, max_timesteps: int):
        world_states_raw_sequence = {}
        world_states_scaled_sequence = {}
        current_raw_world_state = self._generate_initial_world_state(noncritical_only=True)
        permanently_critical_props = set()

        for timestep in range(1, max_timesteps + 1):
            props_forced_critical_at_this_ts = [
                prop_key for prop_key, ts in list_critical_prop_time if ts == timestep
            ]
            permanently_critical_props.update(props_forced_critical_at_this_ts)

            naturally_updated_raw_state = {}
            for prop_key, raw_value in current_raw_world_state.items():
                prop_type_name = prop_key.split('_')[1]
                updater_func = self.property_updaters.get(prop_type_name)
                if updater_func:
                    naturally_updated_raw_state[prop_key] = updater_func(raw_value, prop_type_name)
                else:
                    naturally_updated_raw_state[prop_key] = raw_value

            final_raw_state_for_ts = self._apply_critical_at_ts(
                naturally_updated_raw_state,
                permanently_critical_props
            )

            # This loop ensures any property that naturally became critical (or was forced by _apply_critical_at_ts)
            # is added to the permanent set so it stays critical.
            for prop_key, raw_value_final in final_raw_state_for_ts.items():
                prop_type_name = prop_key.split('_')[1]
                if self._is_critical(prop_type_name, raw_value_final):
                    permanently_critical_props.add(prop_key)

            current_raw_world_state = final_raw_state_for_ts.copy()
            world_states_raw_sequence[timestep] = current_raw_world_state
            world_states_scaled_sequence[timestep] = scale_world_state(current_raw_world_state, self.property_specs_df)

        return world_states_raw_sequence, world_states_scaled_sequence
    def generate_dynamic_scenarios(self, num_scenarios_per_type: int = 50,
                                   start_id: int = 51) -> pd.DataFrame:

        scenarios = []
        i = start_id
        for num_critical in [2, 3, 4]:

            selected_combinations = []
            all_combinations = list(itertools.combinations(self.all_properties, num_critical))
            if len(all_combinations) <= num_scenarios_per_type:
                selected_combinations = all_combinations
            else:
                selected_combinations = random.sample(all_combinations, num_scenarios_per_type)

            for i, critical_props_base in enumerate(selected_combinations, start=start_id):
                scenario_id = f"dyn_{i}_{num_critical}"

                # 1. Assign critical timesteps based on density
                list_critical_prop_time, density_value = self._assign_critical_timesteps_by_density(
                    list(critical_props_base), self.max_timesteps
                )

                # 2. Simulate the world states over time
                world_states_raw_sequence, world_states_scaled_sequence = self._simulate_scenario_timesteps(
                    list_critical_prop_time, self.max_timesteps
                )

                scenarios.append({
                    'scenario_id': scenario_id,
                    'type': 'dynamic',
                    'num_critical_prop': num_critical,
                    'density': density_value,
                    'list_critical_prop_time': list_critical_prop_time,
                    'world_states': world_states_raw_sequence,
                    'scaled_world_states': world_states_scaled_sequence
                })

        return pd.DataFrame(scenarios)

    def verify_criticality_over_time(self, dynamic_scenarios_df, output_csv_path):

        verification_data = []

        for index, row in dynamic_scenarios_df.iterrows():
            scenario_id = row['scenario_id']
            # Safely evaluate string representation of dict of dicts
            scaled_world_states_seq = row['scaled_world_states']

            # Ensure it's a dictionary if it was read from CSV as a string
            if isinstance(scaled_world_states_seq, str):
                scaled_world_states_seq = ast.literal_eval(scaled_world_states_seq)

            for timestep, scaled_state in scaled_world_states_seq.items():
                critical_props_at_ts = []
                for prop_key, scaled_value in scaled_state.items():
                    prop_type_name = prop_key.split('_')[1]

                    raw_world_states_seq = row['world_states']
                    if isinstance(raw_world_states_seq, str):
                        raw_world_states_seq = ast.literal_eval(raw_world_states_seq)

                    raw_value_at_ts = raw_world_states_seq[timestep].get(prop_key)

                    if raw_value_at_ts is not None and self._is_critical(prop_type_name, raw_value_at_ts):
                        critical_props_at_ts.append(prop_key)

                verification_data.append({
                    'scenario_id': scenario_id,
                    'timestep': timestep,
                    'critical_properties': str(critical_props_at_ts)
                })

        verification_df = pd.DataFrame(verification_data)
        verification_df.to_csv(output_csv_path, index=False)

class Scenario:
    def __init__(self, scenario_data):

        self.scenario_id = scenario_data['scenario_id']
        self.num_critical_prop = scenario_data['num_critical_prop']

        loaded_scaled_world_state = ast.literal_eval(scenario_data['scaled_world_states'])

        if isinstance(loaded_scaled_world_state, dict) and all(
                isinstance(v, dict) for v in loaded_scaled_world_state.values()):

            self.sc_type = 'dynamic'
            self.list_critical_prop = ast.literal_eval(scenario_data['list_critical_prop_time'])
            self.density = scenario_data['density']
            self.world_state = ast.literal_eval(scenario_data['world_states'])

            self.all_scaled_world_states = loaded_scaled_world_state
            self.scaled_world_state = self.all_scaled_world_states.get(1)


        elif isinstance(loaded_scaled_world_state, dict):

            self.sc_type = 'static'
            self.list_critical_prop = ast.literal_eval(scenario_data['list_critical_prop'])
            self.density = None
            self.world_state = ast.literal_eval(scenario_data['world_state'])

            self.all_scaled_world_states = loaded_scaled_world_state
            self.scaled_world_state = self.all_scaled_world_states


        else:
            raise ValueError(
                f"Unexpected format for scaled_world_state in scenario {self.scenario_id}. Expected a dict or dict of dicts.")

    def __repr__(self):
        return f"Scenario(ID: {self.scenario_id}, Critical Props: {self.num_critical_prop})"

class BeliefGenerator:

    def __init__(self, property_specs_dict):
        self.property_specs_dict = property_specs_dict

    def belief_prob_mass_continuous(self, bins, center=None, spread=0.2):

        if center is None:
            center = np.random.uniform(0.01, 0.99)

        distances = np.abs(bins - center)
        weights = np.exp(-distances / spread)
        probs = weights / weights.sum()
        sum_probs = sum(round(p, 2) for p in probs)
        probs = [(round(p, 2) / sum_probs) for p in probs]

        return {round(b, 2): round(p, 2) for b, p in zip(bins, probs)}

    def belief_prob_mass_binary(self, high_prob_for_one=None):

        if high_prob_for_one is True: # User believes non-critical (value = 1) with high probability
            p_one = random.uniform(0.75, 0.95)

        elif high_prob_for_one is False: # User believes critical (value = 0) with high probability
            p_one = random.uniform(0.05, 0.25)

        else:
            p_one = random.uniform(0.3, 0.7)

        p_zero = 1 - p_one
        return {0: round(p_zero, 2), 1: round(p_one, 2)}


    def generate_user_belief(self, property_name, actual_scaled_value, property_spec, awareness_tag=None, user_general_awareness=None, num_bins=6):

        spec = property_spec

        if not spec:
            # print(f"Warning: Property spec not found for {property_name}.")
            if 'continuous' in property_name.lower():
                return self.belief_prob_mass_continuous(np.linspace(0, 1, num_bins), center=np.random.uniform(0,1))
            else:
                return self.belief_prob_mass_binary(high_prob_for_one=None)

        if spec.prop_type == "continuous":
            bins = np.linspace(0, 1, num_bins)
            center = None
            spread = 0.15


            SMALL_OFFSET_RANGE = [0.05, 0.15]
            LARGE_OFFSET_RANGE = [0.3, 0.5]

            if awareness_tag == 'high':  # CR. Prop: High Aware
                center = spec.center_high_aware
                spread = 0.15
            elif awareness_tag == 'low':  # CR. Prop: Low Aware
                center = spec.center_low_aware
                spread = 0.15

            else:  # OTHER or Non-Critical Prop
                if user_general_awareness == 'aware' and actual_scaled_value is not None:

                    # Non-critical: Aware user small offset, tight Gaussian
                    sampled_offset = random.uniform(SMALL_OFFSET_RANGE[0], SMALL_OFFSET_RANGE[1])
                    center = random.uniform(max(0, actual_scaled_value - sampled_offset),
                                            min(1, actual_scaled_value + sampled_offset))
                    spread = 0.1

                elif user_general_awareness == 'unaware' and actual_scaled_value is not None:
                    # Non-critical: Unaware user center with large offset, wide Gaussian -- both wrong and imprecise
                    sampled_offset = random.uniform(LARGE_OFFSET_RANGE[0], LARGE_OFFSET_RANGE[1])
                    center = random.uniform(max(0, actual_scaled_value - sampled_offset),
                                            min(1, actual_scaled_value + sampled_offset))
                    spread = 0.45
                else:
                    center = np.random.uniform(0.01, 0.99)
                    spread = 0.3

            return self.belief_prob_mass_continuous(bins, center, spread)

        elif spec.prop_type == "binary":
            high_prob_for_one = None
            if awareness_tag == 'high':
                # High Awareness -- user believes prop is critical (value 0)
                high_prob_for_one = False
            elif awareness_tag == 'low':
                # Low Awareness -- user believes prop is NOT critical (value 1)
                high_prob_for_one = True
            else:  # OTHER or Non-Critical Prop (Original logic for binary non-critical)
                if user_general_awareness == 'unaware':
                    high_prob_for_one = None  # Default random 50/50
                elif user_general_awareness == 'aware' and actual_scaled_value is not None:
                    if actual_scaled_value == 1:
                        high_prob_for_one = True
                    elif actual_scaled_value == 0:
                        high_prob_for_one = False
                else:
                    high_prob_for_one = None

            return self.belief_prob_mass_binary(high_prob_for_one)

        return {}

class TestCaseGenerator:

    AWARENESS_LEVELS = {
        '10%': {'tag': 'low', 'threshold': 0.1},
        '25%': {'tag': 'low', 'threshold': 0.25},
        '50%': {'tag': 'okay', 'threshold': 0.5},
        '80%': {'tag': 'high', 'threshold': 0.8}
    }
    GENERAL_AWARENESS_OPTIONS = ['aware', 'unaware']

    def __init__(self, scenario_csv_path, property_specs_dict):

        self.scenario_csv_path = scenario_csv_path
        self.property_specs_dict = property_specs_dict
        self.belief_generator = BeliefGenerator(property_specs_dict)
        self.test_cases_df = pd.DataFrame(columns=[
            'tc_id',
            'scenario_id',
            'sc_type',

            # Critical property metrics
            'user_critical_aware',
            'critical_properties',
            'num_critical_prop',
            'list_critical_prop',
            'list_cr_prop_aware',

            # Scenario properties
            'density',
            'user_general_aware',

            # Awareness metrics
            'low_aware_props',
            'high_aware_props',
            'num_low_aware',
            'num_high_aware',

            #Sharing Metrics
            'shared_attr',
            'shared_drones',
            'sharing_metric',

            #World Values
            'world_state',
            'scaled_world_state',
            'all_scaled_world_states',
            'user_belief_distribution'
        ])

    def _load_scenarios(self):
        df = pd.read_csv(self.scenario_csv_path)
        print(len(df))
        return [Scenario(row) for index, row in df.iterrows()]

    def _extract_critical_properties(self, list_cr_prop_aware, sc_type):
        if not list_cr_prop_aware:
            return []
        if sc_type == 'static':
            return list(list_cr_prop_aware.keys())
        else:
            return [prop for prop, _ in list_cr_prop_aware.keys()]

    def _extract_crprop_with_awareness(self, list_cr_prop_aware, sc_type):

        if not list_cr_prop_aware:
            return {}

        if sc_type == 'static':
            return list_cr_prop_aware

        if sc_type == 'dynamic':
            temp_extract = {}
            for (prop, _), aware_tag in list_cr_prop_aware.items():
                temp_extract[prop] = aware_tag
            return temp_extract

    def _compute_shared_metrics(self, critical_props):

        if not critical_props:
            return {'shared_attr': 0, 'shared_drones': 0, 'sharing_metric': 0}


        attr_counts = {}
        for prop in critical_props:
            attr = prop.split('_')[-1]
            attr_counts[attr] = attr_counts.get(attr, 0) + 1


        drone_counts = {}
        for prop in critical_props:
            drone = prop.split('_')[0]
            drone_counts[drone] = drone_counts.get(drone, 0) + 1

        shared_attr = sum(c for c in attr_counts.values() if c > 1)
        shared_drones = sum(c for c in drone_counts.values() if c > 1)
        total_props = len(critical_props)

        return {
            'shared_attr': shared_attr,
            'shared_drones': shared_drones,
            'sharing_metric': (shared_attr + shared_drones) / (2 * total_props) if total_props else 0
        }

    def _extract_awareness_levels(self, list_cr_prop_aware, sc_type):

        low_aware = []
        high_aware = []

        if not list_cr_prop_aware:
            return {'low': low_aware, 'high': high_aware}

        # Static: {property: awareness}
        if sc_type == 'static':
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

    def _assign_awareness_tags(self, critical_properties, awareness_percentage):

        num_critical = len(critical_properties)
        num_high_awareness = int(awareness_percentage * num_critical)  # based on %, those many properties get 'high', rest we will determine whether high/low

        # when non-integer % pick random (eg., 10% of num_critical_prop = 5 is 0.5 -- because not integer, random int choice (0 or 1); 25% is 1.5 so pick b/w 1 or 2 for tag assignment
        if not num_high_awareness == awareness_percentage * num_critical:
            if random.random() < (awareness_percentage * num_critical) - num_high_awareness:
                num_high_awareness += 1

        if num_high_awareness > num_critical:
            num_high_awareness = num_critical

        crit_props_awareness = {prop: 'low' for prop in critical_properties}

        high_awareness_props = random.sample(critical_properties, num_high_awareness)
        for prop in high_awareness_props:
            crit_props_awareness[prop] = 'high'

        return crit_props_awareness

    def _generate_beliefs_for_scenario(self, scenario, list_cr_prop_aware_dict, user_general_aware_value):

        user_belief_distribution = {}
        all_properties = list(scenario.scaled_world_state.keys())

        for prop_name in all_properties:
            actual_scaled_value = scenario.scaled_world_state.get(prop_name)
            prop_type_key = prop_name.split('_')[1]
            prop_spec = self.property_specs_dict.get(prop_type_key)

            if prop_name in list_cr_prop_aware_dict:
                print(prop_name, list_cr_prop_aware_dict)
                awareness_tag_for_prop = list_cr_prop_aware_dict[prop_name]
                # Critical property: pass awareness_tag
                # print(f"{prop_name} with {awareness_tag_for_prop}")
                user_belief_distribution[prop_name] = self.belief_generator.generate_user_belief(
                    prop_name,
                    actual_scaled_value=actual_scaled_value,
                    property_spec=prop_spec,
                    awareness_tag=awareness_tag_for_prop,
                    user_general_awareness=None)

            else:
                # Non-critical property: general awareness logic
                user_belief_distribution[prop_name] = self.belief_generator.generate_user_belief(
                    prop_name,
                    actual_scaled_value=actual_scaled_value,
                    property_spec=prop_spec,
                    awareness_tag=None,
                    user_general_awareness=user_general_aware_value)

        return user_belief_distribution

    def _create_test_case_row(self, scenario, user_critical_aware_type_str, user_general_aware_value,
                              list_cr_prop_aware_dict, user_belief_distribution):

        tc_id = f"{scenario.scenario_id}_{user_critical_aware_type_str.replace('%', '')}"

        list_cr_prop_aware = list_cr_prop_aware_dict
        if isinstance(list_cr_prop_aware, str):
            try:
                list_cr_prop_aware = ast.literal_eval(list_cr_prop_aware)
            except (ValueError, SyntaxError):
                list_cr_prop_aware = None

        critical_props = self._extract_critical_properties(list_cr_prop_aware, scenario.sc_type)
        sharing_metrics = self._compute_shared_metrics(critical_props)
        awareness_levels = self._extract_awareness_levels(list_cr_prop_aware, scenario.sc_type)

        return {
                'tc_id': tc_id,
                'scenario_id': scenario.scenario_id,
                'sc_type': scenario.sc_type,

                'user_critical_aware': user_critical_aware_type_str,
                'critical_properties': critical_props,
                'num_critical_prop': scenario.num_critical_prop,
                'list_critical_prop': str(scenario.list_critical_prop),
                'list_cr_prop_aware': str(list_cr_prop_aware_dict),

                'density': getattr(scenario, 'density', None),
                'user_general_aware': user_general_aware_value,

                'low_aware_props': awareness_levels['low'],
                'high_aware_props': awareness_levels['high'],
                'num_low_aware': len(awareness_levels['low']),
                'num_high_aware': len(awareness_levels['high']),

                **sharing_metrics,

                'world_state': str(scenario.world_state),
                'scaled_world_state': str(scenario.scaled_world_state),
                'all_scaled_world_states': scenario.all_scaled_world_states,
                'user_belief_distribution': str(user_belief_distribution)
            }

    def generate_test_cases(self):

        scenarios = self._load_scenarios()

        for scenario in scenarios:

            for user_critical_aware_type, critical_aware_info in self.AWARENESS_LEVELS.items():

                crit_prop_aware_percentage = critical_aware_info['threshold']

                #1: Assign awareness tags to CRITICAL properties as per percentage (10%. 25%, 50%, 80%)

                list_cr_prop_aware_dict = self._assign_awareness_tags(
                    scenario.list_critical_prop, crit_prop_aware_percentage
                )

                #2: Randomly pick user type for general awareness (aware/unaware)
                user_general_aware_value = random.choice(self.GENERAL_AWARENESS_OPTIONS)

                #3: Generate user belief distribution for ALL properties
                critical_props_aware = self._extract_crprop_with_awareness(list_cr_prop_aware_dict, scenario.sc_type)
                user_belief_distribution = self._generate_beliefs_for_scenario(
                    scenario, critical_props_aware, user_general_aware_value
                )

                #4: Construct & Add TC
                new_TC_data = self._create_test_case_row(
                    scenario, user_critical_aware_type, user_general_aware_value,
                    list_cr_prop_aware_dict, user_belief_distribution
                )
                self.test_cases_df = pd.concat(
                    [self.test_cases_df, pd.DataFrame([new_TC_data])], ignore_index=True
                )

        return self.test_cases_df


    def save_test_cases(self, output_csv_path):
        self.test_cases_df.to_csv(output_csv_path, index=False)

def create_scenarios(mode="dynamic", version="custom", num_drones=4, max_timestep=7,
                     per_type=25, start_id=101, output_dir=None, overwrite=False):
    """Generate scenarios and their test cases (4 awareness levels each)."""
    output_dir = output_dir or SCENARIO_DIR_PATH
    os.makedirs(output_dir, exist_ok=True)

    scenario_fp = os.path.join(output_dir, f"{mode}_scenarios_v{version}.csv")
    test_case_fp = os.path.join(output_dir, f"{mode}_test_cases_v{version}.csv")
    for path in (scenario_fp, test_case_fp):
        if os.path.exists(path) and not overwrite:
            raise SystemExit(
                f"Refusing to overwrite {path}.\n"
                f"Pick a different --version, or pass --overwrite if you mean it.")

    generator = DroneScenarioGenerator(num_drones, max_timestep)
    if mode == "static":
        scenario_df = generator.generate_static_scenarios(num_scenario_per_cr_type=per_type)
    elif mode == "dynamic":
        scenario_df = generator.generate_dynamic_scenarios(
            num_scenarios_per_type=per_type, start_id=start_id)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'static' or 'dynamic'.")

    scenario_df.to_csv(scenario_fp, index=False)

    test_case_generator = TestCaseGenerator(scenario_fp, PROPERTY_SPECS_DICT)
    generated_df = test_case_generator.generate_test_cases()
    test_case_generator.save_test_cases(test_case_fp)

    print(f"{len(scenario_df)} scenarios -> {scenario_fp}")
    print(f"{len(generated_df)} test cases -> {test_case_fp}")
    print(f"Run them with:  --test-cases {test_case_fp}")
    return scenario_fp, test_case_fp
