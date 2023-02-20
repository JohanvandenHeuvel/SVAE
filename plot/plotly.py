import time

import plotly
from plotly.subplots import make_subplots


def save_plotly_figure(fig, title):
    """
    Save plotly figure to pdf.
    """
    # save to file
    # see https://github.com/plotly/plotly.py/issues/3469
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    time.sleep(2)
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    print(f"save plot to {title}.pdf")
    return fig


def single_plotly_figure(
    x_axes=None,
    y_axes=None,
):
    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=600,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    if y_axes is None:
        # turn off y-axis
        fig.update_yaxes(title="y", visible=False, showticklabels=False)
    else:
        fig.update_yaxes(range=y_axes, row=1, col=1)

    if x_axes is None:
        # turn off x-axis
        fig.update_xaxes(title="x", visible=False, showticklabels=False)
    else:
        fig.update_xaxes(range=x_axes, row=1, col=1)
    return fig


def dual_plotly_figure(
    x_axes1=None,
    y_axes1=None,
    x_axes2=None,
    y_axes2=None,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Latent", "Observation"),
        horizontal_spacing=0.025,
    )
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1600,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    if y_axes1 is None:
        # turn off y-axis
        fig.update_yaxes(title="y", visible=False, showticklabels=False)
    else:
        fig.update_yaxes(range=y_axes1, row=1, col=1)

    if y_axes2 is None:
        # turn off y-axis
        fig.update_yaxes(title="y", visible=False, showticklabels=False)
    else:
        fig.update_yaxes(range=y_axes2, row=1, col=2)

    if x_axes1 is None:
        # turn off x-axis
        fig.update_xaxes(title="x", visible=False, showticklabels=False)
    else:
        fig.update_xaxes(range=x_axes1, row=1, col=1)

    if x_axes2 is None:
        # turn off x-axis
        fig.update_xaxes(title="x", visible=False, showticklabels=False)
    else:
        fig.update_xaxes(range=x_axes2, row=1, col=2)

    return fig
