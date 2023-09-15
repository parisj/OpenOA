import csv
import toml
from inspect import signature
from itertools import product
from typing import Any, Dict, Iterator, Optional
import os

class Params:
    """
    Class to manage simulation parameters.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize Params object with default values and update based on provided kwargs.
        """
        self.pitch_threshold = []
        self.num_sim = 100
        self.power_bin_mad_thresh = 7
        self.ws_bins = [5.0, 6.0, 7.0, 8.0]
        self.ws_bin_width = 1.0
        self.vane_bin_width = 1.0
        self.min_vane_bin_count = 100
        self.max_abs_vane_angle = 25.0
        self.max_power_filter = None
        self.use_power_coeff = False
        self.wake_free_zones = None
        self.iterate = False
        self.UQ = False
        self.asset = 'kelmarsh'
        
        # Update attributes based on provided kwargs
        self.__dict__.update(kwargs)

    @classmethod
    def load_from_toml(cls, filename: str) -> 'Params':
 
        data = toml.load(filename)
        fixed_params = data.get('fixed_params', {})
        variable_params = data.get('variable_params', {})
        
        if not fixed_params and not variable_params:
            raise ValueError("Both fixed_params and variable_params are empty.")
        
        return cls(**fixed_params), variable_params

    def yield_param_combinations(self, varying_params: Optional[Dict[str, list]] = None) -> Iterator[Dict[str, Any]]:

        if varying_params is None or len(varying_params) == 0:
            yield self.__dict__.copy()
            return

        keys, values = zip(*varying_params.items())
        for combination in product(*values):
            temp_dict = self.__dict__.copy()
            temp_dict.update(dict(zip(keys, combination)))
            yield temp_dict

    def __str__(self) -> str:
        return str(self.__dict__)

    def update_params(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def filter_params(func, params_dict):
        """Filter out parameters from params_dict that are relevant to func."""
        # Try to get the original, undecorated function if possible
        original_func = getattr(func, '__wrapped__', func)
        sig = signature(original_func)
        param_names = set(sig.parameters.keys())
        return {k: v for k, v in params_dict.items() if k in param_names}

    

if __name__ == "__main__":
    # Test loading from TOML
    os.chdir("/home/OST/anton.paris/WeDoWind")

    # Test loading from TOML
    params, variable_params = Params.load_from_toml("code/settings.toml")
    print("params: {params} \n")

    # Test yielding parameter combinations
    for count, param_combination in enumerate(params.yield_param_combinations(variable_params)):
        print(f"count: {count}, params: {param_combination} \n")
        