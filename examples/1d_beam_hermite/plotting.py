"""Plotly visualisation of the deformed von Karman beam (optional extra `plot`)."""


def plot_deformed_beam(
    x_hat,
    w_hat,
    u_hat,
    *,
    x_nodes=None,
    w_nodes=None,
    u_nodes=None,
    amplification=1.0,
    show_undeformed=True,
):
    """Plot the beam in its plane, deformed position (x_hat + a u_hat, a w_hat).

    All inputs are dimensional [m]. The curve (`x_hat`, `w_hat`, `u_hat`) is a
    fine sampling of the solution (e.g. Hermite interpolation of w); the
    optional node arrays are drawn as markers.

    Args:
        x_hat, w_hat, u_hat (array-like): positions, transverse and axial
            displacements along the curve, shape (N_pts,).
        x_nodes, w_nodes, u_nodes (array-like, optional): same at mesh nodes.
        amplification (float): displacement magnification factor a.
        show_undeformed (bool): also draw the initial straight configuration.

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "plot_deformed_beam needs plotly: `uv sync --extra plot`."
        ) from err

    import numpy as np

    x_hat, w_hat, u_hat = (np.asarray(v, dtype=float) for v in (x_hat, w_hat, u_hat))
    a = amplification

    fig = go.Figure()
    if show_undeformed:
        fig.add_trace(go.Scatter(
            x=[x_hat[0], x_hat[-1]], y=[0.0, 0.0], mode="lines",
            name="undeformed", line=dict(color="gray", dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=x_hat + a * u_hat, y=a * w_hat, mode="lines",
        name="deformed (Hermite)", line=dict(width=3),
    ))
    if x_nodes is not None:
        x_n, w_n, u_n = (np.asarray(v, dtype=float) for v in (x_nodes, w_nodes, u_nodes))
        fig.add_trace(go.Scatter(
            x=x_n + a * u_n, y=a * w_n, mode="markers", name="nodes",
            marker=dict(size=8),
        ))

    fig.update_layout(
        title=f"Deformed beam (amplification x{a:g})",
        xaxis_title="x + a u  [m]",
        yaxis_title="a w  [m]",
        template="plotly_white",
    )
    return fig
