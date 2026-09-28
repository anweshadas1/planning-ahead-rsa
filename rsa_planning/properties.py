"""Drone property specs."""

import random

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
    {"property_type": "Battery", "min_range": 0, "max_range": 100, "type": "continuous",
     "critical_start_raw": 10, "critical_start_scaled": 0.1, "relation_sign": -1,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)},

    {"property_type": "WindSpeed", "min_range": 0, "max_range": 22, "type": "continuous",
     "critical_start_raw": 18, "critical_start_scaled": 0.818, "relation_sign": +1,
     "center_high_aware": random.uniform(0.75, 1), "center_low_aware": random.uniform(0.01, 0.2)},

    {"property_type": "Rotor", "min_range": 0, "max_range": 1, "type": "binary",
     "critical_start_raw": 0, "critical_start_scaled": 0, "relation_sign": 0,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)}, # Corrected logic

    {"property_type": "Altitude", "min_range": 50, "max_range": 110, "type": "continuous",
     "critical_start_raw": 100, "critical_start_scaled": 0.833, "relation_sign": +1,
     "center_high_aware": random.uniform(0.75, 1), "center_low_aware": random.uniform(0.01, 0.2)},

    {"property_type": "NoFlyZone", "min_range": 0, "max_range": 1, "type": "binary",
     "critical_start_raw": 0, "critical_start_scaled": 0, "relation_sign": 0,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)}, # Corrected logic

    {"property_type": "Distance", "min_range": 0, "max_range": 100, "type": "continuous",
     "critical_start_raw": 10, "critical_start_scaled": 0.1, "relation_sign": -1,
     "center_high_aware": random.uniform(0.01, 0.2), "center_low_aware": random.uniform(0.75, 1)}
]

PROPERTY_SPECS_DICT = {spec["property_type"]: PropertySpec(spec) for spec in PROPERTY_SPECS}
