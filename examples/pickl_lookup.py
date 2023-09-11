import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


with open(f"WeDoWind/data/penmanshiel/plots_pt_dict.pkl", "rb") as f:
    plots_yaw = pickle.load(f)


with open(f"WeDoWind/data/penmanshiel/result_pt_dict.pkl", "rb") as f:
    summary_df = pickle.load(f)

print(plots_yaw)
print(summary_df)

ws_bins = [5.0, 6.0, 7.0, 8.0, 9.0]
# pitch_threshold = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.1, 2.25]
pitch_threshold = [0.7, 1.5, 2.1]
ws_bin_width = 2.0
min_vane_bin_count = 50
UQ = True
iterate = True
use_power_coeff = False


num_sim = 10
max_power_filter = (0.92, 0.98)
power_bin_mad_thresh = (4.0, 10.0)
asset = "penmanshiel"
p = "Penmanshiel"
project = "project"

wake_free_zones = {
    1: {"turb": [f"{p} 01", f"{p} 04", f"{p} 08", f"{p} 12"], "wind_d_r": (278, 8)},
    2: {
        "turb": [f"{p} 02", f"{p} 13", f"{p} 14", f"{p} 15"],
        "wind_d_r": (348, 93),
    },
    3: {"turb": [f"{p} 07", f"{p} 11", f"{p} 15"], "wind_d_r": (87, 209)},
    4: {"turb": [f"{p} 01", f"{p} 02"], "wind_d_r": (184, 274)},
}


def generate_param_dicts(fixed_params, variable_params):
    param_dicts = []
    for variable_values in product(*variable_params.values()):
        param_dict = {**fixed_params}
        for key, value in zip(variable_params.keys(), variable_values):
            param_dict[key] = value
        param_dicts.append(param_dict)
    return param_dicts


# Define your fixed and variable parameters
fixed_params = {
    "project": project,
    "iterate": iterate,
    "use_power_coeff": use_power_coeff,
    "num_sim": num_sim,
    "power_bin_mad_thresh": power_bin_mad_thresh,
    "max_power_filter": max_power_filter,
    "UQ": UQ,
    "min_vane_bin_count": min_vane_bin_count,
    "wake_free_zones": wake_free_zones,
    "asset": asset,
}

variable_params = {
    "ws_bins": ws_bins,
    "ws_bin_width": [ws_bin_width],  # Fixed value
    "pitch_threshold": pitch_threshold,
}

# Generate the parameter dictionaries
param_dicts = generate_param_dicts(fixed_params, variable_params)

# Print or use the generated parameter dictionaries
for idx, params in enumerate(param_dicts, 1):
    print(f"Parameter Set {idx}:")
    print(params)
