# Import required packages
import sys
import os
from tqdm import tqdm
import cProfile
sys.path.append(r"./OpenOA/examples")
os.chdir("/home/OST/anton.paris/WeDoWind")
from typing import Tuple, Union
import numpy as np
import pandas as pd
from params import Params
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import matplotlib
from typing import List, Dict
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from bokeh.plotting import show
from openoa.analysis.yaw_misalignment import StaticYawMisalignment
from openoa import PlantData
from openoa.utils import plot, filters
import project_Cubico
import pickle
import math
import copy

def save_as_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a Pandas DataFrame as a Parquet file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The path where to save the Parquet file.
        
    Returns:
        None
    """
    df.to_parquet(path, index=False)
    
def read_and_convert_to_parquet(asset: str) -> None:
    """
    Read CSV files, convert and save them as Parquet files.
    
    Parameters:
        asset (str): The asset name used in the file path.
        
    Returns:
        None
    """
    # Read CSV files into Pandas DataFrames
    sc_df = pd.read_csv(f"data/{asset}/scada_df.csv")
    met_df = pd.read_csv(f"data/{asset}/meter_df.csv")
    cur_df = pd.read_csv(f"data/{asset}/curtail_df.csv")
    as_df = pd.read_csv(f"data/{asset}/asset_df.csv")
    
    # Save these DataFrames as Parquet files
    save_as_parquet(sc_df, f"data/{asset}/scada_df.parquet")
    save_as_parquet(met_df, f"data/{asset}/meter_df.parquet")
    save_as_parquet(cur_df, f"data/{asset}/curtail_df.parquet")
    save_as_parquet(as_df, f"data/{asset}/asset_df.parquet")



def load_data(asset: str = "kelmarsh") -> None:
    """
    Loads and prepares data for a given asset, then saves them to CSV and Pickle files.

    Parameters:
    asset (str): The name of the asset for which to load data. Default is "kelmarsh".

    Returns:
    None: The function saves the data to files and does not return anything.
    """
    sc_df, met_df, cur_df, as_df, re_dict = project_Cubico.prepare(
        asset=asset, return_value="dataframes"
    )
    sc_df.to_csv(f"data/{asset}/scada_df.csv", index=False)
    met_df.to_csv(f"data/{asset}/meter_df.csv", index=False)
    cur_df.to_csv(f"data/{asset}/curtail_df.csv", index=False)
    as_df.to_csv(f"data/{asset}/asset_df.csv", index=False)
    with open(f"data/{asset}/reanalysis_dict.pkl", "wb") as f:
        pickle.dump(re_dict, f)


def read_data(time_range: Union[Tuple[int, int], int], asset: str) -> PlantData:
    """
    Reads various datasets for a specific asset and time range, and encapsulates them in a PlantData object.
    
    Parameters:
    time_range (Union[Tuple[int, int], int]): Either a tuple specifying the start and end years or a single year.
    asset (str): The name of the asset for which to read data.

    Returns:
    PlantData: An object containing all the loaded data.
    """    

    sc_df = pd.read_parquet(f"data/{asset}/scada_df.parquet")
    met_df = pd.read_parquet(f"data/{asset}/meter_df.parquet")
    cur_df = pd.read_parquet(f"data/{asset}/curtail_df.parquet")
    as_df = pd.read_parquet(f"data/{asset}/asset_df.parquet")
    with open(f"data/{asset}/reanalysis_dict.pkl", "rb") as f:
        re_dict = pickle.load(f)
    
    sc_df["Timestamp"] = pd.to_datetime(sc_df["Timestamp"])
    met_df["Timestamp"] = pd.to_datetime(met_df["Timestamp"])
    cur_df["Timestamp"] = pd.to_datetime(cur_df["Timestamp"])

    if type(time_range) is tuple:
        sc_df = sc_df.loc[
            (sc_df["Timestamp"].dt.year >= time_range[0])
            & (sc_df["Timestamp"].dt.year <= time_range[1])
        ]
        met_df = met_df.loc[
            (met_df["Timestamp"].dt.year >= time_range[0])
            & (met_df["Timestamp"].dt.year <= time_range[1])
        ]
        cur_df = cur_df.loc[
            (cur_df["Timestamp"].dt.year >= time_range[0])
            & (cur_df["Timestamp"].dt.year <= time_range[1])
        ]

    else:
        sc_df = sc_df.loc[(sc_df["Timestamp"].dt.year == time_range)]
        met_df = met_df.loc[(met_df["Timestamp"].dt.year == time_range)]
        cur_df = cur_df.loc[(cur_df["Timestamp"].dt.year == time_range)]

    plantdata = PlantData(
        analysis_type="MonteCarloAEP",  # Choosing a random type that doesn't fail validation
        metadata=f"data/{asset}/plant_meta.yml",
        scada=sc_df,
        meter=met_df,
        curtail=cur_df,
        asset=as_df,
        reanalysis=re_dict,
    )
    return plantdata


def plot_farm_2d(
    project: PlantData,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    f_size_x: int,
    f_size_y: int,
    x_lim: Tuple[float, float] = None,
    y_lim: Tuple[float, float] = None,
    flag: dict = None,
    title_fig: str = None,
    path: str = None,
    **kwargs,
) -> None:
    """
    Plots 2D scatter plots for turbines in a wind farm.

    Parameters:
    project (PlantData): The project object containing turbine data.
    x (str): The column name in the data for the x-axis.
    y (str): The column name in the data for the y-axis.
    x_title (str): The title for the x-axis.
    y_title (str): The title for the y-axis.
    f_size_x (int): The figure size in the x-direction.
    f_size_y (int): The figure size in the y-direction.
    x_lim (Optional[Tuple[float, float]]): The limits for the x-axis.
    y_lim (Optional[Tuple[float, float]]): The limits for the y-axis.
    flag (Optional[Dict]): Dictionary of flags for highlighting specific data points.
    title_fig (Optional[str]): The title for the entire figure.
    path (Optional[str]): The path to save the figure.
    kwargs (Any): Additional keyword arguments for the scatter plot.

    Returns:
    None: This function plots the data but does not return any value.
    """
    num_turbines = len(project.turbine_ids)
    num_columns = min(2, num_turbines)
    num_rows = math.ceil(num_turbines / num_columns)
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        figsize=(f_size_x * num_columns, f_size_y * num_turbines // num_columns),
    )

    if num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    for i, t in enumerate(project.turbine_ids):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col]

        ax.scatter(
            project.scada.loc[(slice(None), t), x],
            project.scada.loc[(slice(None), t), y],
            **kwargs,
        )
        if flag is not None and t in flag and ~np.any(flag[t]):
            flag[t] = flag[t].reindex(
                project.scada.loc[(slice(None), t), x].index, fill_value=False
            )

            ax.scatter(
                project.scada.loc[(slice(None), t), x].loc[flag[t]],
                project.scada.loc[(slice(None), t), y].loc[flag[t]],
                color="r",
                **kwargs,
            )

        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid("on")
        ax.set_title(t)
    if (
        axs.shape[0] > 1
        and (num_columns - 1) < axs.shape[1]
        and num_turbines < num_rows * num_columns
    ):
        axs[-1, -1].axis("off")

    if title_fig is not None:
        fig.suptitle(title_fig)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path)


def plot_project_bld_ptch_ang(
    project: PlantData, title: str = None, path: str = None
) -> None:
    """
    Plots blade pitch angle against wind speed for all turbines in the given project.

    Parameters:
    project (PlantData): The project object containing turbine data.
    title (Optional[str]): The title for the entire figure.
    path (Optional[str]): The path to save the figure.

    Returns:
    None: This function plots the data but does not return any value.
    """
    
    plot_farm_2d(
        project=project,
        x="WMET_HorWdSpd",
        y="WROT_BlPthAngVal",
        x_title="Wind Speed (m/s)",
        y_title="Blade Pitch Angle (deg.)",
        f_size_x=6,
        f_size_y=4,
        x_lim=(0, 15),
        y_lim=(-1, 4),
        title_fig=title,
        path=path,
        alpha=0.3,
    )


def plot_project_pwr_crv(
    project: PlantData, flag: dict = None, title: str = None, path: str = None, **kwargs
) -> None:
    """
    Plots the power curve, i.e., power vs wind speed, for all turbines in the given project.

    Parameters:
    project (PlantData): The project object containing turbine data.
    flag (Optional[Dict]): Dictionary of flags for highlighting specific data points.
    title (Optional[str]): The title for the entire figure.
    path (Optional[str]): The path to save the figure.
    kwargs (Any): Additional keyword arguments for the scatter plot.

    Returns:
    None: This function plots the data but does not return any value.
    """
    plot_farm_2d(
        project=project,
        x="WMET_HorWdSpd",
        y="WTUR_W",
        x_title="Wind Speed (m/s)",
        y_title="Power (kW)",
        f_size_x=7,
        f_size_y=5,
        title_fig=title,
        alpha=0.3,
        flag=flag,
        path=path,
        **kwargs,
    )


def setup(
    time_range: Union[Tuple[int, int], int], asset: str, load: bool = True
) -> PlantData:
    """
    Initializes a PlantData project by either loading or reading data, and sets its analysis type.

    Parameters:
    time_range (Union[Tuple[int, int], int]): Either a tuple of start and end year or a single year to filter data.
    asset (str): The asset name corresponding to the data to be loaded or read.
    load (bool): Whether to load data. If set to True, the load_data function is called.

    Returns:
    PlantData: A PlantData object with the loaded or read data, set for 'StaticYawMisalignment' analysis.
    """

    if load:
        load_data(asset=asset)

    project = read_data(time_range=time_range, asset=asset)
    project.analysis_type.append("StaticYawMisalignment")
    project.validate()
    return project


def filter_wake_free_zones(project: PlantData, wake_free_zones: dict) -> PlantData:
    """
    Filters data from a PlantData project based on defined wake-free zones.
    
    Parameters:
    project (PlantData): The PlantData project containing all turbine data.
    wake_free_zones (Dict[str, Dict]): A dictionary containing the zones to filter.
        Expected keys for each zone dict are "turb" for turbines and "wind_d_r" for wind direction range.

    Returns:
    List[PlantData]: A list of PlantData projects, each corresponding to a filtered wake-free zone.
    """
    zones = []

    for zone, info in wake_free_zones.items():
        zone_project = copy.deepcopy(project)
        full_mask = pd.Series(False, index=zone_project.scada.index)
        for turbine in info["turb"]:
            wind_range = info["wind_d_r"]

            turbine_data = zone_project.scada.loc[(slice(None), turbine), :]

            if wind_range[0] <= wind_range[1]:
                mask = (turbine_data["WMET_HorWdDir"] >= wind_range[0]) & (
                    turbine_data["WMET_HorWdDir"] <= wind_range[1]
                )
            else:
                mask = (turbine_data["WMET_HorWdDir"] >= wind_range[0]) | (
                    turbine_data["WMET_HorWdDir"] <= wind_range[1]
                )

            full_mask.loc[(slice(None), turbine)] = mask

        zone_project.scada = zone_project.scada[full_mask]
        zone_project.scada = zone_project.scada.loc[(slice(None), info["turb"]), :]
        zone_project.asset = zone_project.asset[
            zone_project.asset.index.isin(info["turb"])
        ]
        zones.append(zone_project)

    return zones


def filter_data(
    project: PlantData, pitch_threshold: float = 1.5, power_bin_mad_thresh: float = 7.0
) -> list:
    """
    Filters turbine data based on multiple criteria.
    - Power curve window range flag
    - Power curve bin filter
    - Unresponsive flag wind speed measurement
    - Power curve bin filter

    Parameters:
    project (PlantData): The PlantData object containing all turbine data.
    pitch_threshold (float): The threshold for the blade pitch angle. Default is 1.5.
    power_bin_mad_thresh (float): The threshold for the power bin mean absolute deviation. Default is 7.0.

    Returns:
    Dict[str, pd.Series]: A dictionary containing the bin filter flags for each turbine.
    """
    
    flag_bins = {}

    for t in project.turbine_ids:
        df_sub = project.scada.loc[(slice(None), t), :]
        out_of_window = filters.window_range_flag(
            df_sub["WMET_HorWdSpd"], 5.0, 40, df_sub["WTUR_W"], 20.0, 2100.0
        )
        df_sub = df_sub[~out_of_window]
        max_bin = 0.90 * df_sub["WTUR_W"].max()
        bin_outliers = filters.bin_filter(
            df_sub["WTUR_W"],
            df_sub["WMET_HorWdSpd"],
            100,
            1.5,
            "median",
            20.0,
            max_bin,
            "scalar",
            "all",
        )
        df_sub = df_sub[~bin_outliers]

        #unresponsive flag
        frozen = filters.unresponsive_flag(df_sub["WMET_HorWdSpd"], 3)
        df_sub = df_sub[~frozen]
        df_sub = df_sub[df_sub["WROT_BlPthAngVal"] <= pitch_threshold]
        project.scada.loc[(slice(None), t), :] = df_sub


        turb_capac = project.asset.loc[t, "rated_power"]
        flag_bin = filters.bin_filter(
            bin_col=df_sub["WTUR_W"],
            value_col=df_sub["WMET_HorWdSpd"],
            bin_width=0.04 * 0.94 * turb_capac,
            threshold=power_bin_mad_thresh,
            center_type="median",
            bin_min=0.01 * turb_capac,
            bin_max=0.95 * turb_capac,
            threshold_type="mad",
            direction="all",
        )
        flag_bins[t] = flag_bin
        df_sub = df_sub[~flag_bin]
        project.scada.loc[(slice(None), t), :] = df_sub
    return flag_bins


def iterate_parameters(project: PlantData, params: Params, variables: dict) -> pd.DataFrame:
    """
    Iterates through different parameter combinations to process each zone.

    Parameters:
    project (PlantData): The PlantData object containing all turbine data.
    params (Params): Object containing methods to yield parameter combinations.
    variables (Dict): Dictionary of variables used for parameter combinations.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: DataFrames summarizing the analysis and plots for yaw angles.
    """

    parameter_iter = params.yield_param_combinations(variables)
    summary_df, plots_yaw = create_dataframes(variables)
    
    for parameter in tqdm(parameter_iter, desc="Processing combinations", dynamic_ncols= True):
        process_each_zone(
            project = project,
            params = parameter, 
            variables = variables,
            summary_df = summary_df,
            plots_yaw = plots_yaw 
        )


    return summary_df, plots_yaw

def filter_params_for_yaw_mis_run(params_dict: dict)-> dict:
    """
    Filters out invalid keys from the parameter dictionary for StaticYawMisalignment.run()

    Parameters:
    params_dict (Dict[str, any]): Original parameter dictionary.

    Returns:
    Dict[str, any]: new params_dict with only valid keys 
    """
    valid_keys = [
        'num_sim', 'ws_bins', 'ws_bin_width', 'vane_bin_width', 
        'min_vane_bin_count', 'max_abs_vane_angle', 'pitch_thresh', 
        'max_power_filter', 'power_bin_mad_thresh', 'use_power_coeff'
    ]
    return {k: v for k, v in params_dict.items() if k in valid_keys}

def create_dataframes(variable_params: Union[dict, List[str]])-> tuple:
    """
    Creates summary and plot DataFrames with dynamic index columns.
    Dynamic based on entries of variable params. 

    Parameters:
    variable_params (Union[Dict, List[str]]): A dictionary or list containing variable parameter names.

    Returns:
    tuple: A tuple containing two DataFrames: summary_df and plots_yaw.
    """

    # Always include these columns
    fixed = ['turbine', 'zone']
    base_columns_summary = ["avg_yaw", "avg_vane", "bin_yaw_ws",'stati_yaw' "conf_int"]
    base_columns_plots = ["figure", "axes"]
    
    if isinstance(variable_params, dict):
        variable_params_list = list(variable_params.keys())
    else:
        variable_params_list = variable_params
    
    # Create DataFrames with dynamic index columns
    summary_df = pd.DataFrame(columns=fixed + variable_params_list + base_columns_summary)
    summary_df = summary_df.set_index(fixed + variable_params_list)
    
    plots_yaw = pd.DataFrame(columns=fixed + variable_params_list + base_columns_plots)
    plots_yaw = plots_yaw.set_index(fixed + variable_params_list)
    
    return summary_df, plots_yaw

def process_each_zone(project: PlantData, params: dict, variables: dict, summary_df: pd.DataFrame, plots_yaw: pd.DataFrame) -> None:
    """
    Process each wake-free zone and update summary and plots DataFrames.
    
    Parameters:
        project (PlantData): The project object containing the plant data.
        params (dict): A dictionary of parameters related to wake-free zones, pitch thresholds, etc.
        variables (dict): A dictionary of variable parameters.
        summary_df (pd.DataFrame): DataFrame to store summary data.
        plots_yaw (pd.DataFrame): DataFrame to store plot data.
        
    Returns:
        None: This function modifies summary_df and plots_yaw in place.
    """
    
    zones = filter_wake_free_zones(project=project, wake_free_zones=params['wake_free_zones'])
    filter_data(
        project=project, pitch_threshold=params['pitch_threshold'], power_bin_mad_thresh=params['power_bin_mad_thresh']
    )
        
    for n_zones, project_wake_free in enumerate(zones):
        weibull_bin_weights = calculate_bin_weights(project_wake_free,"WMET_HorWdSpd",params['ws_bins'])
        yaw_mis = process_zone_plots(project_wake_free, params, n_zones, summary_df, plots_yaw)
        update_summary_and_plots(yaw_mis, params, n_zones, summary_df, plots_yaw, weibull_bin_weights)
        plt.cla()
        plt.close()
    


def update_summary_and_plots(yaw_mis: StaticYawMisalignment, params: dict, n_zones: int, summary_df: pd.DataFrame, plots_yaw: pd.DataFrame, weibull_bin_weights: dict )->None:
    """
    Update the summary and plots DataFrames with yaw misalignment analysis results.
    
    Parameters:
        yaw_mis (StaticYawMisalignment): An object containing yaw misalignment data.
        params (Dict): A dictionary containing parameters.
        n_zones (int): Zone number.
        summary_df (pd.DataFrame): DataFrame for storing summary data.
        plots_yaw (pd.DataFrame): DataFrame for storing plot data.
        weibull_bin_weights (Dict): Dictionary of Weibull bin weights.
    
    Returns:
        None: The function updates summary_df and plots_yaw in place.
    """
    dynamic_index = tuple(params[key] for key in sorted(variable_params.keys()))
    ws_bins = params["ws_bins"]
    UQ = params['UQ']
    
    for i, t in enumerate(yaw_mis.turbine_ids):
        #print(f"Overall yaw misalignment for Turbine {t}: {np.round(yaw_mis.yaw_misalignment[i],1)} degrees")

        percentile_results = []
        if UQ:
            for bin in range(yaw_mis.yaw_misalignment_ws.shape[2]):
                # Extract the nth measurements for the specific turbine and wind speed bin
                nth_measurements = yaw_mis.yaw_misalignment_ws[:, i, bin]
                # Calculate the 2.5th and 97.5th percentiles
                lower_percentile = np.percentile(nth_measurements, 2.5)
                upper_percentile = np.percentile(nth_measurements, 97.5)
                # Store the result in the list
                percentile_results.append([lower_percentile, upper_percentile])
            avg_vane = np.nanmean(yaw_mis.mean_vane_angle_ws[:, i, :], 0)
            bin_yaw_ws = np.nanmean(yaw_mis.yaw_misalignment_ws[:, i, :], 0)
            avg_yaw = np.nanmean(yaw_mis.yaw_misalignment[:, 0])
            
        else:
            avg_yaw = yaw_mis.yaw_misalignment[i]
            bin_yaw_ws = yaw_mis.yaw_misalignment_ws[i]
            avg_vane = yaw_mis.mean_vane_angle_ws[i]
            
        static_yaw = calculate_weighted_yaw_mean(bin_yaw_ws, weibull_bin_weights, ws_bins)
        location = (t, n_zones + 1) + dynamic_index
        summary_df.loc[location] = {
            "ws_bins": ws_bins,
            "avg_yaw": avg_yaw,
            "avg_vane": avg_vane,
            "bin_yaw_ws": bin_yaw_ws,
            "static_yaw": static_yaw,
            "conf_int": percentile_results
        }
        
    zone_plots = yaw_mis.plot_yaw_misalignment_by_turbine(return_fig=True)
    for key, value in zone_plots.items():
        location = (key, n_zones + 1) + dynamic_index
        plots_yaw.loc[location] = value
        
def process_zone_plots(project_wake_free: PlantData, params: dict, n_zones: int, summary_df: pd.DataFrame, plots_yaw: pd.DataFrame) -> StaticYawMisalignment:
    """
    Process plots and perform yaw misalignment analysis for each zone.

    Parameters:
        project_wake_free (PlantData): The data of the wake-free zone turbine group.
        params (Dict): A dictionary containing simulation parameters.
        n_zones (int): The zone number.
        summary_df (pd.DataFrame): DataFrame for storing summary data.
        plots_yaw (pd.DataFrame): DataFrame for storing plot data.

    Returns:
        StaticYawMisalignment: An object containing yaw misalignment analysis.
    """
    UQ = params['UQ']
    asset = params['asset']
    ws_bins = params['ws_bins']
    pitch_threshold = params['pitch_threshold']

    path_power_curve = f"plots/wake_free_zone/{asset}/Zone_{n_zones+1}/zone_power_curve_pt_{pitch_threshold}.png"
    path_pitch_angle = f"plots/wake_free_zone/{asset}/Zone_{n_zones+1}/zone_pitch_angle_pt_{pitch_threshold}.png"
    
    plot_project_pwr_crv(
        project=project_wake_free,
        title=f"Zone {n_zones+1} - Power Curve (filtered)",
        path=path_power_curve,
    )
    plot_project_bld_ptch_ang(
        project=project_wake_free,
        title=f"Zone {n_zones+1} - Blade Pitch Angle vs Wind Speed (filtered)",
        path=path_pitch_angle,
    )
    yaw_mis = StaticYawMisalignment(
        plant=project_wake_free, turbine_ids=None, UQ=UQ
    )
    param_dict = filter_params_for_yaw_mis_run(params)
    yaw_mis.run(**param_dict)
    return yaw_mis
    
def  weibull_pdf(x: Union[float, np.ndarray], c: float, scale: float) -> Union[float, np.ndarray]:
    """
    Calculate the probability density function (PDF) of a Weibull distribution.

    Parameters:
    - x (Union[float, np.ndarray]): The input value(s) for which the PDF is calculated.
    - c (float): The shape parameter of the Weibull distribution. Must be positive.
    - scale (float): The scale parameter of the Weibull distribution. Must be positive.

    Returns:
    - Union[float, np.ndarray]: The PDF value(s) for the given x.
    """
    return (c / scale) * (x / scale)**(c - 1) * np.exp(-((x / scale)**c))

def calculate_bin_weights(df: pd.DataFrame, ws_column: str, ws_bins: List[float]) -> Dict[float, float]:
    """
    Calculate Weibull weights for each wind speed bin.

    Parameters:
    - df (DataFrame): DataFrame containing wind speed data.
    - ws_column (str): Column name for wind speed in the DataFrame.
    - ws_bins (list): List of wind speed bins.

    Returns:
    - dict: Dictionary containing weights for each wind speed bin.
    """
    # Extract wind speed data
    wind_speed_data = df.scada[ws_column].dropna().values
    bins_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # Fit Weibull distribution to data
    counts, bins = np.histogram(wind_speed_data, bins= bins_size, density=True)
    params, _ = curve_fit(weibull_pdf, bins[0:-1], counts)
    c, scale = params

    # Calculate Weibull CDF at bin edges
    bin_edges = np.concatenate([[ws_bins[0] - 1], ws_bins])  # Adding a lower edge for the first bin
    cdf_values = weibull_min.cdf(bin_edges, c, scale=scale)

    # Calculate probability mass for each bin
    prob_mass = np.diff(cdf_values)

    # Normalize to get weights
    weights = prob_mass / np.sum(prob_mass)

    # Create a dictionary to store the weights
    weight_dict = {ws_bin: weight for ws_bin, weight in zip(ws_bins, weights)}

    return weight_dict
    
def calculate_weighted_yaw_mean(yaw_misalignment_ws: Dict[float, float], weibull_weights: Dict[float, float], ws_bins: list) -> float:
    """
    Calculate the weighted yaw misalignment mean using Weibull weights.

    Parameters:

        - weibull_weights (dict): Dictionary containing Weibull weights for each wind speed bin.

    Returns:
    - float: Weighted yaw misalignment mean.ddd
    """
    # Filter out the weights for which we have yaw misalignment data
    filtered_weights = {k: weibull_weights[k] for k in ws_bins}

    # Normalize these filtered weights
    total_weight = sum(filtered_weights.values())
    normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}

    # Calculate the weighted mean
    weighted_yaw_mean = sum(yaw_misalignment_ws[i] * normalized_weights[int(k)] for i, k in enumerate(normalized_weights))

    return weighted_yaw_mean

if __name__ == "__main__":
    """
    DEFAULT PARAMETERS IN PARAMS CLASS
    -------------------------------------
    project = None
    pitch_threshold = []
    num_sim = 100
    power_bin_mad_thresh = 7
    ws_bins = [5.0, 6.0, 7.0, 8.0]
    ws_bin_width = 1.0
    vane_bin_width = 1.0
    min_vane_bin_count = 100
    max_abs_vane_angle = 25.0
    pitch_thresh = 0.5
    max_power_filter = None
    use_power_coeff = False
    wake_free_zones = None
    iterate = False
    UQ = False
    ------------------------------------
    To change value pass them as kwargs to the class
    e.g. Params(project = project, pitch_threshold = 1.5, num_sim = 100, power_bin_mad_thresh = 7)
    """
    
    profiler = cProfile.Profile()
    profiler.enable()
  #  
    asset = "penmanshiel"

  #  # first time run should be with load = True to create the data files
  #  # after the first run, load = False will be much faster
    project = setup(time_range=(2019, 2021), asset=asset, load=False)

    # fix problem with duplicated index
    project = setup(time_range=(2019, 2021), asset=asset, load=False)
    project.scada = project.scada[~project.scada.index.duplicated(keep="first")]

    # plot_project_bld_ptch_ang(project=project)

    # Create Params object
    # to simulate different params, follow description in code/settings.toml
    params, variable_params = Params.load_from_toml("code/settings.toml")
    result, plots = iterate_parameters(project=project, params=params, variables=variable_params)

    with open(f"data/{asset}/result_pt_dict.pkl", "wb") as f:
        pickle.dump(result, f)

    with open(f"data/{asset}/plots_pt_dict.pkl", "wb") as g:
        pickle.dump(plots, g)
    os.sleep(1)
    profiler.disable()
    profiler.print_stats(sort='cumulative')