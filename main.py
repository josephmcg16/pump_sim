import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from piping_components import Pump, ControlValve, Pipe, SimplePipingSystem


def design_of_experiment(
    pump_speeds=np.linspace(0, 1, 100), valve_positions=np.linspace(0, 1, 100)
) -> pd.DataFrame:
    """Design of experiment for pump and control valve components.

    Parameters
    ----------
        pump_speeds: np.ndarray, optional
            Array of pump speeds.
        valve_positions: np.ndarray, optional
            Array of valve positions.

    Returns
    -------
        pd.DataFrame
            Design of experiment for pump and control valve components.
    """
    pump_speeds_grid, valve_positions_grid = np.meshgrid(pump_speeds, valve_positions)
    df = pd.DataFrame(
        {
            "Pump Speed [-]": pump_speeds_grid.flatten(),
            "Valve Position [-]": valve_positions_grid.flatten(),
        }
    )
    return df


def plot(df):
    df_pivot = df.pivot(
        index="Pump Speed [-]", columns="Valve Position [-]", values="Flowrate [m3/s]"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=df_pivot.index,
            y=df_pivot.columns,
            z=df_pivot.values,
            colorscale="Viridis",
        )
    )
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Pump Speed [-]",
            yaxis_title="Valve Position [-]",
            zaxis_title="Flowrate [m3/s]",
            xaxis=dict(autorange="reversed"),
            yaxis=dict(autorange="reversed"),
        ),
    )
    fig.update_traces(
        hovertemplate="""Pump Speed: %{x:.2f}
        <br>Valve Position: %{y:.2f}
        <br>Flowrate: %{z:.2e} m3/s<extra></extra>"""
    )
    return fig


def main():
    flow_system = SimplePipingSystem(
        pump=Pump([75, 2e5]), pipe=Pipe(10), valve=ControlValve(20)
    )
    df = design_of_experiment()
    df["Flowrate [m3/s]"] = np.nan
    for row in tqdm(df.itertuples(), total=len(df)):
        pump_speed, valve_position = row._1, row._2
        flowrate = flow_system.calculate_flowrate(pump_speed, valve_position)
        df.at[row.Index, "Flowrate [m3/s]"] = flowrate
    fig = plot(df)
    fig.show()


if __name__ == "__main__":
    main()
