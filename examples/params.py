import csv
from inspect import signature

class Params:
    def __init__(self, **kwargs):
        self.project = None  # Type should be PlantData but set to None here
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

    def save_to_csv(self, filename):
        """Save the attributes to a CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.__dict__.keys())
            writer.writerow(self.__dict__.values())

    @classmethod
    def load_from_csv(cls, filename):
        """Load attributes from a CSV file and return a new Params object."""
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            keys = next(reader)
            values = next(reader)
        kwargs = {key: value for key, value in zip(keys, values)}
        return cls(**kwargs)

    def __str__(self):
        """Return a string representation of the Params object."""
        return str(self.__dict__)

    def update_params(self, **kwargs):
        """Update attributes based on provided kwargs."""
        self.__dict__.update(kwargs)

    def params_func(self, func):
        """Extract parameters from the Params object that are relevant to the given function."""
        sig = signature(func)
        param_names = list(sig.parameters.keys())
        func_params = {k: v for k, v in self.__dict__.items() if k in param_names}
        return func_params
    
    
def test_loading_params(project:str, pitch_threshold:list, wake_free_zones:dict): 
    pass

if __name__ == "__main__":
    param_dict = {'project': "project", 'iterate': True}
    params = Params(**param_dict)
    print(params.__dict__)

    # Method 2
    params2 = Params(project="project", iterate=True)
    print(params2.__dict__)