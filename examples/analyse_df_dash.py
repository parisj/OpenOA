import os


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import pickle
from dash_bootstrap_templates import load_figure_template

os.chdir("/home/OST/anton.paris/WeDoWind")
# Generate dropdowns based on DataFrame indices
def generate_dropdowns(df):
    dropdowns = []
    for level in df.index.names:
        dropdowns.append(
            html.Div([
                dcc.Dropdown(
                    id=f'{level}-dropdown',
                    placeholder=f"Select {level}",  # Placeholder text
                    options=[{'label': i, 'value': i} for i in df.index.get_level_values(level).unique()],
                    value=None,
                    style={'width': '200px', 'color': '#000'}  # Set a fixed width and text color
                )
            ], style={'padding': '10px'})
        )
    return dropdowns

# Run the app
if __name__ == '__main__':
    
    with open(f"data/penmanshiel/result_pt_dict.pkl", "rb") as f:
        summary_df = pickle.load(f)
    ws_bins = [5.0, 6.0, 7.0, 8.0, 9.0]
    empty_cols = [col for col in summary_df.columns if summary_df[col].isnull().all()]
    summary_df.drop(empty_cols, axis=1, inplace=True)

    app = dash.Dash(external_stylesheets=[dbc.themes.ZEPHYR])
    load_figure_template('ZEPHYR')



    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    sidebar = html.Div(
    [
        html.H5("Analyse Results", className="display-9"),
        html.Hr(),
        html.P(
            "Filter ", className="display-15"
        ),
        dbc.Col(generate_dropdowns(summary_df),width = 2,   style = {'margin-left':'0px', 'margin-to':'0px', 'margin-right': '0px'}),

    ],
    style=SIDEBAR_STYLE,
)
    navbar = dbc.NavbarSimple(brand="Static Yaw Misalignment Analysis",
                                  brand_href="#",
                                  color="primary",
                                  dark=True,
                                )
    
    content = html.Div([  
      
                        dbc.Row([   
                                    dbc.Col(dcc.Graph(id="facetgrid-plot"),   width = 10, style = {'width': '100%','margin-left':'5px', 'margin-top':'7px', 'margin-right':'5px'}),
                                ],
                                 style = {'margin-left':"16rem"}
                                ),
                        ],
                        style = {'margin-left':'0px', 'margin-top':'0px', 'margin-right':'0px'}
                        )
    




    app.layout = html.Div([ dbc.Row([
        sidebar,
        content])])
    # Define the layout


    # Define the callback to update the plot
    @app.callback(
        Output('facetgrid-plot', 'figure'),
        [Input(f'{level}-dropdown', 'value') for level in summary_df.index.names]
    )
    def update_plot(*args):
        """
        Update the plot based on the dropdown values.
        """
        # Start with the original DataFrame
        filtered_df = summary_df.reset_index()

        # Apply filters based on dropdown values
        for level, value in zip(summary_df.index.names, args):
            if value is not None:  # Filter only if a value is selected
                dtype = summary_df.reset_index()[level].dtype
                if dtype != object:  # Convert to the same dtype as in DataFrame if not string
                    value = dtype.type(value)
                filtered_df = filtered_df[filtered_df[level] == value]

        # Create a new DataFrame for plotting
        plot_data = []
        for _, row in filtered_df.iterrows():
            for ws, vane, yaw in zip(ws_bins, row['avg_vane'], row['bin_yaw_ws']):
                new_row = row.to_dict()
                new_row['ws_bin'] = ws
                new_row['avg_vane'] = vane
                new_row['bin_yaw_ws'] = yaw
                plot_data.append(new_row)

        plot_df = pd.DataFrame(plot_data)

        # Sort the DataFrame based on the x-axis and facetting columns
        plot_df.sort_values(by=['turbine', 'ws_bin'], inplace=True)

        num_turbines = plot_df['turbine'].nunique()
        
        num_columns_per_row = 3  # You set this to 3 earlier
        num_rows = -(-num_turbines // num_columns_per_row)  # Ceiling division

        # Calculate the height needed for each row and the total height for the figure
        height_per_row = 300  # You can adjust this value based on your needs
        total_height = num_rows * height_per_row + 200
        
        # Create the plot
        fig = px.line(
            plot_df,
            x='ws_bin',
            y=['avg_vane', 'bin_yaw_ws'],
            color='variable',
            facet_col='turbine',
            facet_col_wrap=3, 
            title='Overview of filtered data',
            height=total_height,
            labels={
                    'ws_bin': 'Wind Speed [m/s]',  # x-axis label
                    'avg_vane': 'Average Vane [°]',  # y-axis label for 'avg_vane'
                    'bin_yaw_ws': 'Yaw Misalignment [°]'   # y-axis label for 'bin_yaw_ws'
                    },
            range_y =[-8, 8]
        )
        for trace in fig['data']:
            if trace.name == 'avg_vane':
                trace.name = 'Average Vane'
            elif trace.name == 'bin_yaw_ws':
                trace.name = 'Yaw Misalignment'
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        #fig.for_each_xaxis(lambda axis: axis.update(title_text= "Wind Speed [m/s]"))

        fig.update_yaxes(title_text="Offset Turbine [°]", row=1, col=1)
        fig.update_layout(
            margin=dict(l=20, r=40, t=80, b=0),  # Adjust margins
            autosize=False # You can adjust the height and width as needed
        )

        
        
        return fig
    app.run_server(debug=True)