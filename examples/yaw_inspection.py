# Import required packages
import sys
sys.path.append(r"./OpenOA/examples")
from typing import Tuple, Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from bokeh.plotting import show
from openoa.analysis.yaw_misalignment import StaticYawMisalignment
from openoa import PlantData
from openoa.utils import plot, filters
import project_Cubico
import pickle

 
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
                 **kwargs) -> None:
    
    num_turbines = len(project.turbine_ids)
    num_columns = num_turbines // 2  # Ceiling division to get the number of columns

    fig, axs = plt.subplots(nrows=2,
                            ncols=num_columns, 
                            figsize=(f_size_x*num_columns, f_size_y*num_turbines//num_columns))

    if num_columns == 1:
        axs = axs.reshape(2, 1)

    for i, t in enumerate(project.turbine_ids):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col]

        ax.scatter(project.scada.loc[(slice(None), t), x],
                   project.scada.loc[(slice(None), t), y], **kwargs)
        if flag is not None and ~np.any(flag[t]):
            ax.scatter(project.scada.loc[(slice(None), t), x].loc[flag[t]],
                       project.scada.loc[(slice(None), t), y].loc[flag[t]],
                       color='r', **kwargs)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid("on")
        ax.set_title(t)
    if num_turbines % 2:
        axs[1, num_columns-1].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_project_bld_ptch_ang(project: PlantData) -> None:
    # plot blade pitch angle vs wind speed of all turbines in the project
    plot_farm_2d(project=project,
                 x = "WMET_HorWdSpd",
                 y = "WROT_BlPthAngVal",
                 x_title="Wind Speed (m/s)",
                 y_title="Blade Pitch Angle (deg.)",
                 f_size_x=6, f_size_y=4,
                 x_lim=(0, 15),
                 y_lim=(-1, 4),
                 alpha=0.5,
    )
def plot_project_pwr_crv(project: PlantData, flag:dict = None, **kwargs) -> None:
    plot_farm_2d(project=project,
                 x = "WMET_HorWdSpd",
                 y = "WTUR_W",
                 x_title="Wind Speed (m/s)",
                 y_title="Power (kW)",
                 f_size_x=7, f_size_y=5,
                 alpha=0.5,
                 flag=flag,
                 **kwargs,
    )       

def setup(time_range: Union[Tuple[int, int], int],
          asset: str, load: bool = True) -> PlantData:
    
    if load:
        load_data(time_range=time_range, asset=asset)   
        
    project = read_data(time_range=time_range, asset=asset)
    project.analysis_type.append("StaticYawMisalignment")
    project.validate()
    return project


def filter_data(project: PlantData, pitch_threshold: float = 1.5,
                power_bin_mad_thresh: float = 7.0) -> PlantData:

    flag_bins = {}
    project.scada = project.scada[project.scada["WROT_BlPthAngVal"] <= pitch_threshold]

    for t in project.turbine_ids:

        df_sub = project.scada.loc[(slice(None), t),:]
        out_of_window = filters.window_range_flag(df_sub["WMET_HorWdSpd"], 5., 40,df_sub["WTUR_W"], 20., 2050.)
        df_sub = df_sub[~out_of_window]      
        max_bin = 0.90 * df_sub["WTUR_W"].max()
        bin_outliers = filters.bin_filter(df_sub["WTUR_W"], df_sub["WMET_HorWdSpd"], 100, 1.5, "median", 20., max_bin, "scalar", "all")
        df_sub = df_sub[~bin_outliers]
        frozen = filters.unresponsive_flag(df_sub["WMET_HorWdSpd"], 3)
        df_sub = df_sub[~frozen]
        df_sub = df_sub[df_sub["WROT_BlPthAngVal"] <= pitch_threshold]  
        
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

if __name__ == "__main__":
    #first time run should be with load = True to create the data files
    project = setup(time_range=(2019,2021), asset="kelmarsh", load=False)
    
    #fix problem with duplicated index
    project.scada = project.scada[~project.scada.index.duplicated(keep='first')]
    
    plot_project_bld_ptch_ang(project=project)
    flag_bins = filter_data(project=project)
    plot_project_pwr_crv(project=project, flag=flag_bins)
    
    #start the yaw misalignment analysis
    yaw_mis = StaticYawMisalignment(plant=project,
                                    turbine_ids=None,
                                    UQ=False,
                                    )
    power_bin_mad_thresh = 7.0
    pitch_threshold = 1.5
    yaw_mis.run(
    num_sim = 1,
    #num_sim = 100,
    ws_bins = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ws_bin_width = 2.0,
    vane_bin_width = 1.0,
    min_vane_bin_count = 50,
    max_abs_vane_angle = 25.0,
    pitch_thresh = pitch_threshold,
    #max_power_filter = (0.92,0.98),
    #power_bin_mad_thresh = (4.0, 10.0),
    max_power_filter = 0.95,
    power_bin_mad_thresh = power_bin_mad_thresh,
    use_power_coeff = False
    )
    
    
    for i, t in enumerate(yaw_mis.turbine_ids):
        print(f"Overall yaw misalignment for Turbine {t}: {np.round(yaw_mis.yaw_misalignment[i],1)} degrees")

    axes_dict = yaw_mis.plot_yaw_misalignment_by_turbine(return_fig = True)
    plt.show()