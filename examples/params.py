import csv
import toml
from inspect import signature
from itertools import product
from typing import Any, Dict, Iterator, Optional, List, Union
import os


class Params:
    """
    Class to manage simulation parameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize Params object with default values and update them based on provided kwargs.

        Keyword Arguments:
        **kwargs: Arbitrary keyword arguments.
        """
        # Initialize default parameters
        self.pitch_threshold: List = []
        self.num_sim: int = 100
        self.power_bin_mad_thresh: int = 7
        self.ws_bins: List[float] = [5.0, 6.0, 7.0, 8.0]
        self.ws_bin_width: float = 1.0
        self.vane_bin_width: float = 1.0
        self.min_vane_bin_count: int = 100
        self.max_abs_vane_angle: float = 25.0
        self.max_power_filter: Optional[float] = None
        self.use_power_coeff: bool = False
        self.wake_free_zones: Optional[Any] = None
        self.iterate: bool = False
        self.UQ: bool = False
        self.asset: str = "kelmarsh"

        # Update attributes based on provided kwargs
        self.__dict__.update(kwargs)

    @classmethod
    def load_from_toml(cls, filename: str) -> Union["Params", Dict[str, Any]]:
        """
        Load parameters from a TOML file.

        Arguments:
        filename (str): The path of the TOML file.

        Returns:
        Params, Dict: Params object and a dictionary containing variable parameters.
        """
        data = toml.load(filename)
        fixed_params = data.get("fixed_params", {})
        variable_params = data.get("variable_params", {})

        if not fixed_params and not variable_params:
            raise ValueError("Both fixed_params and variable_params are empty.")

        return cls(**fixed_params), variable_params

    def yield_param_combinations(
        self, varying_params: Optional[Dict[str, list]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield all combinations of parameters based on varying_params.

        Arguments:
        varying_params (Dict[str, list], optional): Dictionary containing varying parameters.

        Returns:
        Iterator[Dict[str, Any]]: An iterator yielding dictionaries containing parameter combinations.
        """
        if varying_params is None or len(varying_params) == 0:
            yield self.__dict__.copy()
            return

        keys, values = zip(*varying_params.items())
        for combination in product(*values):
            temp_dict = self.__dict__.copy()
            temp_dict.update(dict(zip(keys, combination)))
            yield temp_dict

    def __str__(self) -> str:
        """Return a string representation of the Params object."""
        return str(self.__dict__)

    def update_params(self, **kwargs: Any) -> None:
        """
        Update parameters with provided keyword arguments.

        Keyword Arguments:
        **kwargs: Arbitrary keyword arguments to update.
        """
        self.__dict__.update(kwargs)

    @staticmethod
    def filter_params(func, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out parameters from params_dict that are relevant to func.

        Arguments:
        func: The function to filter parameters for.
        params_dict (Dict[str, Any]): The dictionary containing parameters to be filtered.

        Returns:
        Dict[str, Any]: A dictionary containing filtered parameters.
        """
        # Try to get the original, undecorated function if possible
        original_func = getattr(func, "__wrapped__", func)
        sig = signature(original_func)
        param_names = set(sig.parameters.keys())
        return {k: v for k, v in params_dict.items() if k in param_names}


if __name__ == "__main__":
    # Test loading from TOML
    os.chdir("/home/OST/anton.paris/WeDoWind")

    # Test loading from TOML
    params, variable_params = Params.load_from_toml("code/settings.toml")
    print(f"params: {params} \n")

    # Test yielding parameter combinations
    for count, param_combination in enumerate(
        params.yield_param_combinations(variable_params)
    ):
        print(f"count: {count}, params: {param_combination} \n")
