import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import pickle
from dash_bootstrap_templates import load_figure_template
from typing import List, Union


os.chdir("/home/OST/anton.paris/WeDoWind")


def generate_dropdowns(df: pd.DataFrame) -> List[html.Div]:
    """
    Generate dropdown elements based on DataFrame index names.

    Args:
        df (pd.DataFrame): The DataFrame to generate dropdown options from.

    Returns:
        List[html.Div]: A list of Div elements containing dropdowns.
    """
    dropdowns = []
    for level in df.index.names:
        dropdowns.append(
            html.Div(
                [
                    dcc.Dropdown(
                        id=f"{level}-dropdown",
                        placeholder=f"Select {level}",
                        options=[
                            {"label": i, "value": i}
                            for i in df.index.get_level_values(level).unique()
                        ],
                        value=None,
                        style={
                            "width": "200px",
                            "color": "#000",
                        },
                    )
                ],
                style={"padding": "10px"},
            )
        )
    return dropdowns


# Run the app
if __name__ == "__main__":
    with open(f"data/penmanshiel/result_pt_dict.pkl", "rb") as f:
        summary_df = pickle.load(f)

    # Removed code for brevity

    @app.callback(
        Output("facetgrid-plot", "figure"),
        [Input(f"{level}-dropdown", "value") for level in summary_df.index.names],
    )
    def update_plot(*args: Union[str, int, float, None]) -> px.Figure:
        """
        Update the plot based on the dropdown values.

        Args:
            *args (Union[str, int, float, None]): Selected dropdown values.

        Returns:
            px.Figure: Updated Plotly figure.
        """
        # Rest of your code
