import plotly.express as px
import pandas as pd


def plot_time_series(t, z, title="Signal over Time"):
    """
    Plots a signal in the time domain.
    z: signal (1D array-like)
    t: time axis (1D array-like)
    """
    df = pd.DataFrame({"time": t, "signal": z})

    fig = px.line(
        df,
        x="time",
        y="signal",
        labels={"value": "Amplitude", "time": "Time (s)"},
        title=title
    )
    return fig


# Helper function to maintain zoom level
def maintain_zoom(relayout_data, fig):
    if relayout_data and "xaxis.range[0]" in relayout_data:
        fig.update_layout(
            xaxis=dict(
                range=[relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
            )
        )
    if relayout_data and "yaxis.range[0]" in relayout_data:
        fig.update_layout(
            yaxis=dict(
                range=[relayout_data["yaxis.range[0]"], relayout_data["yaxis.range[1]"]]
            )
        )