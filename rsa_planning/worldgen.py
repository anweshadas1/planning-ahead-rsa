"""World states, criticality and initial beliefs."""

import random

import numpy as np
import pandas as pd


def sample_noncritical_value(min_val, max_val, critical_start, relation_sign, margin=5):
    '''
    sample non-critical values given property's specifications.
    relation = -1, lower values are critical eg., Battery, Dist
    relation = +1, higher values are critical eg., WS, Alt
    margin - init world state atleast that many units from/till critical threshold
    '''
    if relation_sign == -1:
        lower_bound = min(critical_start + margin, max_val)
        return round(np.random.uniform(lower_bound, max_val), 3)
    elif relation_sign == +1:
        upper_bound = max(critical_start - margin, min_val)
        return round(np.random.uniform(min_val, upper_bound), 3)
    else:
        return 1 if max_val > 0.5 else 0

def random_initial_world_state(num_drones, num_properties, property_specs_df, noncritical_only=False):
    '''
    init world state with random values for each of the drone properties.
    flag: noncritical_only, sample vlaues outside critical ranges.
    '''
    world_state = {}
    for drone in range(1, num_drones + 1):
        for prop in property_specs_df.iloc[:num_properties].itertuples():
            key = f"Drone{drone}_{prop.property_type}"

            if prop.type == "continuous":
                if noncritical_only:
                    world_state[key] = sample_noncritical_value(
                        prop.min_range, prop.max_range, prop.critical_start_raw, prop.relation_sign
                    )
                else:
                    world_state[key] = np.random.uniform(prop.min_range, prop.max_range)
            else:
                if noncritical_only:
                    world_state[key] = 1
                else:
                    world_state[key] = np.random.choice([0, 1])
    return world_state

def scale_world_state(world_state, property_specs_df: pd.DataFrame):
    """
    scale all different world state values to the domain of [0,1]
    """
    scaled_world_state = {}
    for key, value in world_state.items():
        property_name = key.split("_")[1]
        # Ensure property_name exists in property_specs_df index before .loc
        if property_name not in property_specs_df.index:
            scaled_world_state[key] = value # Keep original if spec not found
            continue

        # Use .loc with the property_name as index
        prop_row = property_specs_df.loc[property_name]

        if prop_row["type"] == "continuous":
            min_range = prop_row["min_range"]
            max_range = prop_row["max_range"]
            if max_range == min_range: # Avoid division by zero
                scaled_value = 0.0
            else:
                scaled_value = (value - min_range) / (max_range - min_range)
            scaled_world_state[key] = round(np.clip(scaled_value, 0.0, 1.0), 3) # Ensure clip to 0-1
        else:
            scaled_world_state[key] = value # Binary values are already 0 or 1
    return scaled_world_state

def set_criticality(world_state, property_specs_df, use_scaled=False):

    """
    check whether property value is critical or not
    flag: use_scaled, which threshold to use, raw (init) OR scaled (during sim/prior setting).
    """

    criticality = {}
    for key, value in world_state.items():
        property_name = key.split("_")[1]
        prop_row = property_specs_df[property_specs_df["property_type"] == property_name].iloc[0]
        critical_threshold = prop_row["critical_start_scaled"] if use_scaled else prop_row["critical_start_raw"]

        if prop_row["type"] == "binary":
            criticality[key] = 1 if value == 0 else 0
        else:
            if prop_row["relation_sign"] == -1:  # lower critical Batt, Dist
                criticality[key] = 1 if value <= critical_threshold else 0
            elif prop_row["relation_sign"] == +1:  # higher critical WS, Alt
                criticality[key] = 1 if value >= critical_threshold else 0
            else:
                criticality[key] = 0
    return criticality

def belief_prob_mass_continuous(bins, center=None, spread=0.2):
    """
    Generate a probability distribution over bins.
    - Centered around a likely value if given.
    - Uses exponential decay to distribute probability realistically.
    """
    distances = np.abs(bins - center)
    weights = np.exp(-distances / spread)
    probs = weights / weights.sum()

    return {round(b, 2): round(p, 2) for b, p in zip(bins, probs)}

def belief_prob_mass_binary():
    p_high = np.random.uniform(0.5, 0.9)
    p_low = 1 - p_high
    return {0: round(p_low, 2), 1: round(p_high, 2)} if np.random.rand() > 0.5 else {0: round(p_high, 2),
                                                                                     1: round(p_low, 2)}

def set_initial_human_belief(num_drones, property_specs, num_bins=5):

    human_belief = {}

    for drone in range(1, num_drones + 1):
        for _, spec in property_specs.iterrows():
            key = f"Drone{drone}_{spec['property_type']}"

            if spec["type"].lower() == "continuous":
                bins = np.linspace(0, 1, num_bins)
                center = np.random.uniform(0.2, 0.9)
                human_belief[key] = belief_prob_mass_continuous(bins, center)

            elif spec["type"].lower() == "binary":
                human_belief[key] = belief_prob_mass_binary()

    return human_belief

def set_priors(human_belief, property_specs_df):

    priors = []

    for entity_property, belief_distribution in human_belief.items():

        # type of property & specs, "Battery" from "D1_B"
        base_property = entity_property.split("_")[1]
        spec = property_specs_df[property_specs_df["property_type"] == base_property].iloc[0]


        if spec["type"] == "continuous":
            critical_threshold = spec["critical_start_scaled"]
            if spec["relation_sign"] == -1:
                prior = sum(prob for value, prob in belief_distribution.items() if value <= critical_threshold)
            elif spec["relation_sign"] == 1:
                prior = sum(prob for value, prob in belief_distribution.items() if value >= critical_threshold)
            else:
                prior = 0.0001

        elif spec["type"] == "binary":
            prior = belief_distribution.get(spec["critical_start_scaled"], 0.0001)

        priors.append([prior])

    return np.array(priors)

def generate_test_cases(num_drones, num_properties, is_dynamic=False, horizon=6):
    drones = [f"Drone{i + 1}" for i in range(num_drones)]
    properties = ["Battery", "WindSpeed", "Rotor", "Altitude", "NoFlyZone", "Distance"][:num_properties]

    if not is_dynamic:
        # static
        all_world_states = {f"{drone}_{prop}": random.randint(0, 1) for drone in drones for prop in properties}
    else:
        # dyn
        all_world_states = {}
        for t in range(1, horizon + 1):
            # all_world_states[t] = {f"{drone}_{prop}": random.randint(0, 1) for drone in drones for prop in properties}
            all_world_states[t] = {f"{drone}_{prop}": 1 for drone in drones for prop in properties}

    initial_human_belief = {}
    for key in (all_world_states[1].keys() if is_dynamic else all_world_states.keys()):

        # generic random
        prob_critical = round(random.uniform(0.1, 0.9), 1)
        prob_not_critical = round(1 - prob_critical, 1)

        # near (im/)perfect awareness
        # prob_critical = round(random.uniform(0.05, 0.15), 2)
        # prob_critical = round(random.uniform(0.85, 0.95), 2)
        # prob_not_critical = round(1 - prob_critical, 2)

        initial_human_belief[key] = {0: prob_critical, 1: prob_not_critical}

    return all_world_states, initial_human_belief

def load_test_case(tc_num, filename="test_cases.xlsx"):
    try:
        df = pd.read_excel(filename)
        df['tc_num'] = df['tc_num'].astype(str)
        row = df[df['tc_num'] == str(tc_num)]

        if not row.empty:
            all_world_states = eval(row['all_world_states'].iloc[0])
            initial_human_belief = eval(row['initial_human_belief'].iloc[0])
            return all_world_states, initial_human_belief
        else:
            return None, None
    except FileNotFoundError:
        return None, None
    except KeyError as e:
        print(f"KeyError: {e}. Ensure that the excel file has the columns 'tc_num', 'all_world_states', and 'initial_human_belief'")
        return None, None
