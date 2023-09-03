# Import required packages
import sys
sys.path.append(r"./OpenOA/examples")
from typing import Tuple, Union
import numpy as np
import pandas as pd
from params import Params
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from bokeh.plotting import show
from openoa.analysis.yaw_misalignment import StaticYawMisalignment
from openoa import PlantData
from openoa.utils import plot, filters
import project_Cubico
import pickle
import math 
import copy


 
def load_data(asset: str = "kelmarsh") -> None:

    sc_df, met_df, cur_df, as_df, re_dict = project_Cubico.prepare(asset=asset, return_value="dataframes")
    sc_df.to_csv(f'data/{asset}/scada_df.csv', index=False)
    met_df.to_csv(f'data/{asset}/meter_df.csv', index=False)
    cur_df.to_csv(f'data/{asset}/curtail_df.csv', index=False)
    as_df.to_csv(f'data/{asset}/asset_df.csv', index=False)
    with open(f'data/{asset}/reanalysis_dict.pkl', 'wb') as f:
        pickle.dump(re_dict, f)
    

def read_data(time_range: Union[Tuple[int, int], int],
              asset: str) -> PlantData:
     # import the data
    sc_df=pd.read_csv(f'data/{asset}/scada_df.csv')
    met_df=pd.read_csv(f'data/{asset}/meter_df.csv')
    cur_df=pd.read_csv(f'data/{asset}/curtail_df.csv')
    as_df=pd.read_csv(f'data/{asset}/asset_df.csv')
    with open(f'data/{asset}/reanalysis_dict.pkl', 'rb') as f:
        re_dict = pickle.load(f)
        
    sc_df["Timestamp"] = pd.to_datetime(sc_df["Timestamp"])
    met_df["Timestamp"] = pd.to_datetime(met_df["Timestamp"])
    cur_df["Timestamp"] = pd.to_datetime(cur_df["Timestamp"])
    
    if type(time_range) is tuple:
        sc_df = sc_df.loc[(sc_df["Timestamp"].dt.year >= time_range[0])
                          & (sc_df["Timestamp"].dt.year <= time_range[1])]
        met_df = met_df.loc[(met_df["Timestamp"].dt.year >= time_range[0])
                            & (met_df["Timestamp"].dt.year <= time_range[1])]
        cur_df = cur_df.loc[(cur_df["Timestamp"].dt.year >= time_range[0])
                            & (cur_df["Timestamp"].dt.year <= time_range[1])]

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

def plot_farm_2d(project: PlantData,
                 x: str, y: str,
                 x_title: str, y_title: str,
                 f_size_x: int, f_size_y: int ,
                 x_lim: Tuple[float, float] = None,
                 y_lim: Tuple[float, float] = None,
                 flag: dict = None,
                 title_fig: str = None,
                 path : str = None,
                 **kwargs) -> None:
    num_turbines = len(project.turbine_ids)
    num_columns = min(2, num_turbines)
    num_rows = math.ceil(num_turbines / num_columns)
    fig, axs = plt.subplots(nrows=num_rows,
                            ncols=num_columns, 
                            figsize=(f_size_x*num_columns, f_size_y*num_turbines//num_columns))

    if  num_rows == 1 or num_columns == 1:
        axs = np.reshape(axs, (num_rows, num_columns))

    for i, t in enumerate(project.turbine_ids):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col]

        ax.scatter(project.scada.loc[(slice(None), t), x],
                   project.scada.loc[(slice(None), t), y], **kwargs)
        if flag is not None and t in flag and ~np.any(flag[t]):
            flag[t] = flag[t].reindex(project.scada.loc[(slice(None), t), x].index, fill_value=False)

            ax.scatter(project.scada.loc[(slice(None), t), x].loc[flag[t]],
                       project.scada.loc[(slice(None), t), y].loc[flag[t]],
                       color='r', **kwargs)
        
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid("on")
        ax.set_title(t)
    if axs.shape[0] > 1 and (num_columns - 1) < axs.shape[1] and num_turbines < num_rows * num_columns:
        axs[-1, -1].axis('off')

    if title_fig is not None:
        fig.suptitle(title_fig)        
    plt.tight_layout()
    if path is not None:
        fig.savefig(path)
        
  

def plot_project_bld_ptch_ang(project: PlantData, title: str = None, path : str = None) -> None:
    # plot blade pitch angle vs wind speed of all turbines in the project
    plot_farm_2d(project=project,
                 x = "WMET_HorWdSpd",
                 y = "WROT_BlPthAngVal",
                 x_title="Wind Speed (m/s)",
                 y_title="Blade Pitch Angle (deg.)",
                 f_size_x=6, f_size_y=4,
                 x_lim=(0, 15),
                 y_lim=(-1, 4),
                 title_fig= title,
                 path = path,
                 alpha=0.3,
    )
def plot_project_pwr_crv(project: PlantData, flag:dict = None, title: str=None,path : str = None, **kwargs) -> None:
    plot_farm_2d(project=project,
                 x = "WMET_HorWdSpd",
                 y = "WTUR_W",
                 x_title="Wind Speed (m/s)",
                 y_title="Power (kW)",
                 f_size_x=7, f_size_y=5,
                 title_fig = title,
                 alpha=0.3,
                 flag=flag,
                 path = path,
                 **kwargs,
    )       

def setup(time_range: Union[Tuple[int, int], int],
          asset: str, load: bool = True) -> PlantData:
    
    if load:
        load_data(asset=asset)   
        
    project = read_data(time_range=time_range, asset=asset)
    project.analysis_type.append("StaticYawMisalignment")
    project.validate()
    return project



def filter_wake_free_zones(project: PlantData, wake_free_zone: dict) -> PlantData:
    zones = []
    
    for zone, info in wake_free_zones.items():
        
        zone_project = copy.deepcopy(project)
        full_mask = pd.Series(False, index=zone_project.scada.index)
        for turbine in info['turb']:
            wind_range = info['wind_d_r']
            
            turbine_data = zone_project.scada.loc[(slice(None), turbine), :]
            
            if wind_range[0] <= wind_range[1]:
                mask = (turbine_data['WMET_HorWdDir'] >= wind_range[0]) & \
                       (turbine_data['WMET_HorWdDir'] <= wind_range[1])
            else:
                mask = (turbine_data['WMET_HorWdDir'] >= wind_range[0]) | \
                       (turbine_data['WMET_HorWdDir'] <= wind_range[1])
            
            full_mask.loc[(slice(None), turbine)] = mask
            
        zone_project.scada = zone_project.scada[full_mask]
        zone_project.scada = zone_project.scada.loc[(slice(None), info['turb']), :]
        zone_project.asset = zone_project.asset[zone_project.asset.index.isin(info["turb"])]
        zones.append(zone_project)
    
    return zones

def filter_data(project: PlantData, pitch_threshold: float = 1.5,
                power_bin_mad_thresh: float = 7.0) -> list:

    flag_bins = {}

    for t in project.turbine_ids:

        df_sub = project.scada.loc[(slice(None), t),:]
        out_of_window = filters.window_range_flag(df_sub["WMET_HorWdSpd"], 5., 40,df_sub["WTUR_W"], 20., 2100.)
        df_sub = df_sub[~out_of_window]      
        max_bin = 0.90 * df_sub["WTUR_W"].max()
        bin_outliers = filters.bin_filter(df_sub["WTUR_W"], df_sub["WMET_HorWdSpd"], 100, 1.5, "median", 20., max_bin, "scalar", "all")
        df_sub = df_sub[~bin_outliers]
        frozen = filters.unresponsive_flag(df_sub["WMET_HorWdSpd"], 3)
        df_sub = df_sub[~frozen]
        df_sub = df_sub[df_sub["WROT_BlPthAngVal"] <= pitch_threshold]  
        project.scada.loc[(slice(None), t),:] = df_sub

        # Apply power bin filter
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
        project.scada.loc[(slice(None), t),:] = df_sub
    return flag_bins

def iterate_parameters(project: PlantData, params:Params)-> pd.DataFrame:
    summary_df = pd.DataFrame(columns=["turbine", "pitch_angle" ,'zone', 'ws_bins', 'avg_yaw', 'avg_vane', 'bin_yaw_ws', 'conf_int'])
    summary_df = summary_df.set_index(["turbine", "zone", "pitch_angle"])
    plots_yaw = pd.DataFrame(columns=["turbine", "zone", "pitch_angle", "figure", "axes"])
    plots_yaw = plots_yaw.set_index(["turbine", "zone", "pitch_angle"])
    if params.iterate:
        for pt in params.pitch_threshold: 
            params_new = copy.deepcopy(params)
            params_new.update_params(pitch_threshold = pt)
            summary_df, plots_yaw = process_wake_free_zones_with_parameters(project=project, params=params_new, summary_df=summary_df, plots_yaw=plots_yaw)
    else:
        summary_df, plots_yaw = process_wake_free_zones_with_parameters(project=project, params=params, summary_df=summary_df, plots_yaw=plots_yaw)
          
    return summary_df, plots_yaw

def filter_and_prepare_data(project: PlantData, params: Params):
    """
    Filter and prepare the data 

    Args:
        project (PlantData): Total project data unfiltered
        params (Params): Class params with all the params needed for the filtering
    """
    
    filter_data(project=project, pitch_threshold=params.pitch_threshold, power_bin_mad_thresh=7)
    zones = filter_wake_free_zones(project=project, wake_free_zone=params.wake_free_zones)
    return zones

def process_wake_free_zones_with_parameters(project : PlantData, params: Params, summary_df: pd.DataFrame, plots_yaw: pd.DataFrame):
    """
    A wrapper function to process wake free zones with specified parameters
    

    Args:
        project (PlantData): Total project data unfiltered       
        params (Params): params class with all the params needed for the filtering
    """
    zones = filter_and_prepare_data(project=project, params=params)
    summary_df, plots_yaw = process_each_zone(zones=zones, params=params, summary_df=summary_df, plots_yaw=plots_yaw)
    return summary_df, plots_yaw
    
def process_each_zone(zones: list, params: Params, summary_df: pd.DataFrame, plots_yaw: pd.DataFrame):
    UQ = params.UQ
    asset = params.asset
    ws_bins = params.ws_bins
    pitch_threshold = params.pitch_threshold
    
    for n_zones, project_wake_free in enumerate(zones):
        path_power_curve = f'plots/wake_free_zone/{asset}/Zone_{n_zones+1}/zone_power_curve_pt_{pitch_threshold}.png'
        path_pitch_angle = f'plots/wake_free_zone/{asset}/Zone_{n_zones+1}/zone_pitch_angle_pt_{pitch_threshold}.png'
        plot_project_pwr_crv(project=project_wake_free, title=f"Zone {n_zones+1} - Power Curve (filtered)", path = path_power_curve)
        plot_project_bld_ptch_ang(project=project_wake_free, title=f"Zone {n_zones+1} - Blade Pitch Angle vs Wind Speed (filtered)", path = path_pitch_angle)
        yaw_mis = StaticYawMisalignment(plant=project_wake_free, turbine_ids=None, UQ=UQ)    
        param_dict = {
            "num_sim": params.num_sim,
            "ws_bins": params.ws_bins,
            "ws_bin_width": params.ws_bin_width,
            "vane_bin_width": params.vane_bin_width,
            "min_vane_bin_count": params.min_vane_bin_count,
            "max_abs_vane_angle": params.max_abs_vane_angle,
            "pitch_thresh": params.pitch_threshold,
            "max_power_filter": params.max_power_filter,
            "power_bin_mad_thresh": params.power_bin_mad_thresh,
            "use_power_coeff": params.use_power_coeff,
            }
        print(param_dict)
        yaw_mis.run(**param_dict)
        
        for i, t in enumerate(yaw_mis.turbine_ids):
            file_path = f'data/{asset}/wake_free/{n_zones+1}_zone_test_pt_{pitch_threshold}.csv' 
            print(f"Overall yaw misalignment for Turbine {t}: {np.round(yaw_mis.yaw_misalignment[i],1)} degrees")
            
            if UQ: 
                
                percentile_results = []
                for bin in range(yaw_mis.yaw_misalignment_ws.shape[2]):
                    # Extract the nth measurements for the specific turbine and wind speed bin
                    nth_measurements = yaw_mis.yaw_misalignment_ws[:, i, bin]

                    # Calculate the 2.5th and 97.5th percentiles
                    lower_percentile = np.percentile(nth_measurements, 2.5)
                    upper_percentile = np.percentile(nth_measurements, 97.5)

                    # Store the result in the list
                    percentile_results.append([lower_percentile, upper_percentile])
                avg_vane = np.nanmean(yaw_mis.mean_vane_angle_ws[:, i, :], 0)
                bin_yaw_ws = np.nanmean(yaw_mis.yaw_misalignment_ws[:,i,:], 0)
                avg_yaw = np.nanmean(yaw_mis.yaw_misalignment[:,0])
                
                summary_df.loc[t, n_zones+1, pitch_threshold] = {
                'ws_bins': ws_bins,
                'avg_yaw': avg_yaw,
                'avg_vane': avg_vane,
                'bin_yaw_ws': bin_yaw_ws,
                'conf_int': percentile_results,
                }
                
                
            else: 
                avg_yaw = yaw_mis.yaw_misalignment[i]
                bin_yaw_ws = yaw_mis.yaw_misalignment_ws[i]
                avg_vane = yaw_mis.mean_vane_angle_ws[i]

                summary_df.loc[t, n_zones+1, pitch_threshold] = {
                'ws_bins': ws_bins,
                'avg_yaw': avg_yaw,
                'avg_vane': avg_vane,
                'bin_yaw_ws': bin_yaw_ws,
                }
        zone_plots = yaw_mis.plot_yaw_misalignment_by_turbine(return_fig = True)    
        for key, value in zone_plots.items():
            plots_yaw.loc[key, n_zones+1, pitch_threshold] = value
    return summary_df, plots_yaw  

def process_wake_free_zones(zones: list, params: Params)-> pd.DataFrame:
    UQ = params.UQ
    asset = params.asset
    ws_bins = params.ws_bins
    pitch_threshold = params.pitch_threshold
    
    
    for n_zone,  project_wake_free in enumerate(zones):
    
        path_power_curve = f'plots/wake_free_zone/{asset}/Zone_{n_zone+1}/zone_power_curve_pt_{pitch_threshold}.png'
        path_pitch_angle = f'plots/wake_free_zone/{asset}/Zone_{n_zone+1}/zone_pitch_angle_pt_{pitch_threshold}.png'
        plot_project_pwr_crv(project=project_wake_free, title=f"Zone {n_zone+1} - Power Curve (filtered)", path = path_power_curve)
        plot_project_bld_ptch_ang(project=project_wake_free, title=f"Zone {n_zone+1} - Blade Pitch Angle vs Wind Speed (filtered)", path = path_pitch_angle)

        #start the yaw misalignment analysis
        yaw_mis = StaticYawMisalignment(plant=project_wake_free, turbine_ids=None, UQ=UQ,)
        yaw_params = params.params_func(yaw_mis.run)
        yaw_mis.run(**yaw_params)

        if UQ:
            summary_df = pd.DataFrame(columns=['zone', 'ws_bins', 'avg_yaw', 'avg_vane', 'bin_yaw_ws', 'conf_int'])

        
        else:
            summary_df = pd.DataFrame(columns=['zone', 'ws_bins', 'avg_yaw', 'avg_vane', 'bin_yaw_ws'])

            
        for i, t in enumerate(yaw_mis.turbine_ids):
            file_path = f'data/{asset}/wake_free/{n_zone+1}_zone_test_pt_{pitch_threshold}.csv' 
            print(f"Overall yaw misalignment for Turbine {t}: {np.round(yaw_mis.yaw_misalignment[i],1)} degrees")
            
            if UQ: 
                
                percentile_results = []
                for bin in range(yaw_mis.yaw_misalignment_ws.shape[2]):
                    # Extract the nth measurements for the specific turbine and wind speed bin
                    nth_measurements = yaw_mis.yaw_misalignment_ws[:, i, bin]

                    # Calculate the 2.5th and 97.5th percentiles
                    lower_percentile = np.percentile(nth_measurements, 2.5)
                    upper_percentile = np.percentile(nth_measurements, 97.5)

                    # Store the result in the list
                    percentile_results.append([lower_percentile, upper_percentile])
                avg_vane = np.nanmean(yaw_mis.mean_vane_angle_ws[:, i, :], 0)
                bin_yaw_ws = np.nanmean(yaw_mis.yaw_misalignment_ws[:,i,:], 0)
                avg_yaw = np.nanmean(yaw_mis.yaw_misalignment[:,0])
                
                summary_df.loc[t] = {
                'zone': n_zone+1,
                'ws_bins': ws_bins,
                'avg_yaw': avg_yaw,
                'avg_vane': avg_vane,
                'bin_yaw_ws': bin_yaw_ws,
                'conf_int': percentile_results,
                }
                
                
            else: 
                avg_yaw = yaw_mis.yaw_misalignment[i]
                bin_yaw_ws = yaw_mis.yaw_misalignment_ws[i]
                avg_vane = yaw_mis.mean_vane_angle_ws[i]

                summary_df.loc[t] = {
                'zone': n_zone+1,
                'ws_bins': ws_bins,
                'avg_yaw': avg_yaw,
                'avg_vane': avg_vane,
                'bin_yaw_ws': bin_yaw_ws,
                }
            
            summary_df.to_csv(file_path, index=True)
        axes_dict = yaw_mis.plot_yaw_misalignment_by_turbine(return_fig = True)    


if __name__ == "__main__":
    '''
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
     '''
    asset = "penmanshiel"
    #first time run should be with load = True to create the data files
    #after the first run, load = False will be much faster
    
    project = setup(time_range=(2019,2021), asset=asset, load=False)
    p = "Penmanshiel"

    wake_free_zones = {
        1: {"turb": [f"{p} 01", f"{p} 04", f"{p} 08", f"{p} 12"], "wind_d_r": (278, 8)},
        2: {"turb": [f"{p} 02", f"{p} 13", f"{p} 14", f"{p} 15"], "wind_d_r": (348, 93)},
        3: {"turb": [f"{p} 07", f"{p} 11", f"{p} 15"], "wind_d_r": (87, 209)},
        4: {"turb": [f"{p} 01", f"{p} 02"], "wind_d_r": (184, 274)}
    }

    ws_bins = [5.0, 6.0, 7.0, 8.0]
    pitch_threshold = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25]
    ws_bin_width = 2.0
    min_vane_bin_count = 50
    UQ = True
    iterate = True
    use_power_coeff = False
    
    #fix problem with duplicated index
    project.scada = project.scada[~project.scada.index.duplicated(keep='first')]
    
    #plot_project_bld_ptch_ang(project=project)
        
    

    
    if UQ:
        num_sim = 100
        max_power_filter = (0.92,0.98)
        power_bin_mad_thresh = (4.0, 10.0)
    else:
        num_sim = 1
        max_power_filter = 0.95
        power_bin_mad_thresh = 7.0
        
    param_dict = {
        'project': project,
        'iterate': iterate,
        'use_power_coeff': use_power_coeff,
        'pitch_threshold': pitch_threshold,
        'num_sim': num_sim,
        'power_bin_mad_thresh': power_bin_mad_thresh,
        'max_power_filter': max_power_filter,
        'UQ': UQ,
        'ws_bins': ws_bins,
        'ws_bin_width': ws_bin_width,
        'min_vane_bin_count': min_vane_bin_count,
        'wake_free_zones': wake_free_zones,
        'asset': asset,
    }

    # Create Params object
    params = Params(**param_dict)

    result, plots = iterate_parameters(project = project, params= params)
    

   
    with open(f'data/{asset}/result_pt_dict.pkl', 'wb') as f:
        pickle.dump(result, f)
        
    with open(f'data/{asset}/plots_pt_dict.pkl', 'wb') as g:
        pickle.dump(plots, g)