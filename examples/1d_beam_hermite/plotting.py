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


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as err:
        raise ImportError("plotting needs plotly: `uv sync --extra plot`.") from err
    return go, make_subplots


def plot_loss_history(losses, stages):
    """Plot the training loss against the iteration, one colour per mode.

    Args:
        losses (array-like): loss at every iteration, all modes concatenated.
        stages (array-like): index of the mode (stage) of every iteration.

    Returns:
        plotly.graph_objects.Figure
    """
    go, _ = _plotly()
    import numpy as np

    losses, stages = np.asarray(losses, dtype=float), np.asarray(stages)
    iterations = np.arange(losses.shape[0])

    fig = go.Figure()
    for stage in np.unique(stages):
        mask = stages == stage
        fig.add_trace(go.Scatter(
            x=iterations[mask], y=losses[mask], mode="lines", name=f"mode {stage}",
        ))
    fig.update_layout(
        title="Training loss",
        xaxis_title="iteration",
        yaxis_title="energy J",
        template="plotly_white",
    )
    return fig


def plot_monoms(x, x_monoms, P, P_monoms):
    """Plot the x monom and the P monom of every mode, side by side.

    Each monom is normalised by its max |.|: a monom carries an arbitrary
    scale (w_m -> c w_m, lambda_m -> lambda_m / c leaves the mode unchanged),
    so only its shape is meaningful.

    Args:
        x (array-like): sampling in x, shape (N_x,).
        x_monoms (array-like): x monoms, shape (n_modes, N_x).
        P (array-like): sampling in P, shape (N_P,).
        P_monoms (array-like): P monoms, shape (n_modes, N_P).

    Returns:
        plotly.graph_objects.Figure
    """
    go, make_subplots = _plotly()
    import numpy as np

    def normalised(v):
        top = np.abs(v).max()
        return v / top if top > 0 else v

    fig = make_subplots(rows=1, cols=2, subplot_titles=("x monoms", "P monoms"))
    for col, (axis, monoms) in enumerate(((x, x_monoms), (P, P_monoms)), start=1):
        axis, monoms = np.asarray(axis, dtype=float), np.asarray(monoms, dtype=float)
        for m, monom in enumerate(monoms):
            fig.add_trace(go.Scatter(
                x=axis, y=normalised(monom), mode="lines", name=f"mode {m}",
                legendgroup=f"mode {m}", showlegend=col == 1,
                line=dict(color=f"hsl({60 * m}, 70%, 45%)"),
            ), row=1, col=col)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="P", row=1, col=2)
    fig.update_yaxes(title_text="normalised monom", row=1, col=1)
    fig.update_layout(title="Monoms (normalised shapes)", template="plotly_white")
    return fig


def plot_amplitude(P, w, *, P_critical=None):
    """Plot a deflection against the load, e.g. w(1/2, P).

    Args:
        P (array-like): loads, shape (N_P,).
        w (array-like): deflection at a fixed x, shape (N_P,).
        P_critical (float, optional): drawn as a vertical dashed line.

    Returns:
        plotly.graph_objects.Figure
    """
    go, _ = _plotly()
    import numpy as np

    fig = go.Figure(go.Scatter(
        x=np.asarray(P, dtype=float), y=np.asarray(w, dtype=float),
        mode="lines", name="PGD",
    ))
    if P_critical is not None:
        fig.add_vline(x=float(P_critical), line=dict(color="gray", dash="dash"))
    fig.update_layout(
        title="Deflection against the load",
        xaxis_title="P",
        yaxis_title="w",
        template="plotly_white",
    )
    return fig


def plot_pod_decay(pod, *, noise_floor=None):
    """Plot eps(n) of the POD, one panel per k, one curve per P / P_cr.

    Args:
        pod (pandas.DataFrame): rows of `kolmogorov_study/results/pod.csv`
            (columns k, ratio, sigma_1, sigma_2, ...).
        noise_floor (float, optional): eps below it is within the snapshot
            error; drawn as a grey band.

    Returns:
        plotly.graph_objects.Figure
    """
    go, make_subplots = _plotly()
    import numpy as np

    k_values = sorted(pod.k.unique())
    ratios = sorted(pod.ratio.unique())
    sigma_columns = [c for c in pod.columns if c.startswith("sigma_")]
    fig = make_subplots(rows=1, cols=len(k_values), shared_yaxes=True,
                        subplot_titles=[f"k = {k:g}" for k in k_values])
    for col, k in enumerate(k_values, start=1):
        for i, ratio in enumerate(ratios):
            sigma = pod[(pod.k == k) & (pod.ratio == ratio)][sigma_columns].to_numpy(dtype=float)[0]
            tail = np.cumsum((sigma**2)[::-1])[::-1]
            eps = np.sqrt(np.append(tail, 0.0)[1:] / tail[0])
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(eps) + 1), y=eps, mode="lines", name=f"P/P_cr = {ratio:.2f}",
                legendgroup=f"{ratio}", showlegend=col == 1,
                line=dict(color=f"hsl({240 * i / max(len(ratios) - 1, 1):.0f}, 70%, 45%)"),
            ), row=1, col=col)
        if noise_floor is not None:
            fig.add_hrect(y0=1e-16, y1=noise_floor, fillcolor="gray", opacity=0.2, line_width=0,
                          row=1, col=col)
        fig.update_xaxes(title_text="n", range=[0, 40], row=1, col=col)
    fig.update_yaxes(type="log", range=[-8, 0], title_text="eps(n)", row=1, col=1)
    fig.update_layout(title="POD: relative error of the best rank-n approximation",
                      template="plotly_white")
    return fig


def plot_pod_map(pod, column="n_1e-03"):
    """Heatmap of a POD summary (e.g. n(1e-3)) on the (k, P / P_cr) grid.

    Args:
        pod (pandas.DataFrame): rows of `kolmogorov_study/results/pod.csv`.
        column (str): summary to show.

    Returns:
        plotly.graph_objects.Figure
    """
    go, _ = _plotly()

    table = pod.pivot(index="k", columns="ratio", values=column)
    fig = go.Figure(go.Heatmap(
        z=table.to_numpy(), x=[f"{r:.2f}" for r in table.columns], y=[f"{k:g}" for k in table.index],
        text=table.to_numpy(), texttemplate="%{text:.3g}", colorscale="Viridis",
        colorbar=dict(title=column),
    ))
    fig.update_layout(title=f"POD: {column}", xaxis_title="P / P_cr", yaxis_title="k",
                      template="plotly_white")
    return fig


def plot_snapshot_map(x, a, W, *, title="w(x, a)"):
    """Heatmap of the snapshots w(x, a).

    Args:
        x (array-like): sampling in x, shape (N_x,).
        a (array-like): load positions, shape (N_a,).
        W (array-like): snapshots, shape (N_x, N_a).

    Returns:
        plotly.graph_objects.Figure
    """
    go, _ = _plotly()
    import numpy as np

    fig = go.Figure(go.Heatmap(
        z=np.asarray(W, dtype=float), x=np.asarray(a, dtype=float), y=np.asarray(x, dtype=float),
        colorscale="RdBu", zmid=0.0, colorbar=dict(title="w"),
    ))
    fig.update_layout(title=title, xaxis_title="a", yaxis_title="x", template="plotly_white")
    return fig
