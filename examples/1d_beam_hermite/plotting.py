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


def plot_pod_decay(pod, *, noise_floor=None, panel="k", curve="ratio", curve_label="P/P_cr", n_max=40):
    """Plot eps(n) of the POD, one panel per value of `panel`, one curve per value of `curve`.

    Args:
        pod (pandas.DataFrame): rows of a `pod.csv` of `experiments/kolmogorov_study`
            (columns `panel`, `curve`, sigma_1, sigma_2, ...). Default: E2
            (`results/pod.csv`, panels k, curves P / P_cr).
        noise_floor (float, optional): eps below it is within the snapshot
            error; drawn as a grey band.
        panel (str, optional): column giving the panels; None for one panel.
        curve (str): column giving the curves.
        curve_label (str): name of `curve` in the legend.
        n_max (int): largest n shown.

    Returns:
        plotly.graph_objects.Figure
    """
    go, make_subplots = _plotly()
    import numpy as np

    panels = [None] if panel is None else sorted(pod[panel].unique())
    curves = sorted(pod[curve].unique())
    sigma_columns = [c for c in pod.columns if c.startswith("sigma_")]
    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True,
                        subplot_titles=None if panel is None else [f"{panel} = {v:g}" for v in panels])
    for col, value in enumerate(panels, start=1):
        rows = pod if panel is None else pod[pod[panel] == value]
        for i, c in enumerate(curves):
            sigma = rows[rows[curve] == c][sigma_columns].to_numpy(dtype=float)[0]
            sigma = sigma[~np.isnan(sigma)]
            tail = np.cumsum((sigma**2)[::-1])[::-1]
            eps = np.sqrt(np.append(tail, 0.0)[1:] / tail[0])
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(eps) + 1), y=eps, mode="lines", name=f"{curve_label} = {c:g}",
                legendgroup=f"{c}", showlegend=col == 1,
                line=dict(color=f"hsl({240 * i / max(len(curves) - 1, 1):.0f}, 70%, 45%)"),
            ), row=1, col=col)
        if noise_floor is not None:
            fig.add_hrect(y0=1e-16, y1=noise_floor, fillcolor="gray", opacity=0.2, line_width=0,
                          row=1, col=col)
        fig.update_xaxes(title_text="n", range=[0, n_max], row=1, col=col)
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


def plot_branches_k(branches, points, P, *, mu_critical=None, series=None, title=None):
    """Bifurcation diagram w(1/4) against mu = k^(1/4) of the unloaded beam, at fixed P.

    Every branch (w = +-c phi, followed by continuity in k) is a thin line; the
    global minimum (branch of smallest P_n, the only stable one) is a thick
    line, broken where it jumps from one branch to another; the NeuROM
    solutions are markers.

    Args:
        branches (pandas.DataFrame): rows of `kolmogorov_study/results/c1/branches.csv`.
        points (pandas.DataFrame or None): rows of `kolmogorov_study/results/c1/neurom_points.csv`.
        P (float): axial load.
        mu_critical (float, optional): mu_cr = k_cr^(1/4), where P_1(k) = P
            (unbuckled beyond it); drawn as a vertical dashed line.
        series (dict, optional): label -> DataFrame (columns mu, w_quarter,
            lambda_min) of other NeuROM solutions, e.g. one sweep in k; the
            saddles (lambda_min <= 0) are drawn as red crosses.
        title (str, optional): replaces the default title.

    Returns:
        plotly.graph_objects.Figure
    """
    go, _ = _plotly()
    import numpy as np

    branches = branches[branches.P == P]
    fig = go.Figure()
    for label, df in branches.groupby("branch"):
        df = df.sort_values("mu")
        for sign in (1, -1):
            fig.add_trace(go.Scatter(
                x=df.mu, y=sign * df.w_quarter, mode="lines", name=f"branch {label}",
                legendgroup=f"{label}", showlegend=sign == 1,
                line=dict(color=f"hsl({(47 * label) % 360}, 60%, 55%)", width=1),
            ))
    best = branches[branches["global"]].sort_values("mu")
    jump = best.branch.diff().fillna(0).to_numpy() != 0
    for sign in (1, -1):
        x, y = [], []
        for mu, w, cut in zip(best.mu, sign * best.w_quarter, jump):
            if cut:
                x.append(np.nan)
                y.append(np.nan)
            x.append(mu)
            y.append(w)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name="global minimum", legendgroup="global",
            showlegend=sign == 1, line=dict(color="black", width=3),
        ))
    if points is not None:
        points = points[points.P == P]
        for label, df, symbol in (("NeuROM, start 0.1 phi_n (lowest energy of n = 1, 2, 3)", points[points.kept], "circle-open"),
                                  ("NeuROM, start from a noise", points[points.start == "noise"], "x")):
            fig.add_trace(go.Scatter(
                x=df.mu, y=df.w_quarter, mode="markers", name=label,
                marker=dict(symbol=symbol, size=9, color="crimson"),
            ))
    for label, df in (series or {}).items():
        for name, part, marker in ((label, df[df.lambda_min > 0], dict(symbol="circle", size=5, color="royalblue")),
                                   (f"{label}: saddle", df[df.lambda_min <= 0], dict(symbol="x", size=10, color="red"))):
            if len(part):
                fig.add_trace(go.Scatter(x=part.mu, y=part.w_quarter, mode="markers", name=name, marker=marker))
    if mu_critical is not None:
        fig.add_vline(x=float(mu_critical), line=dict(color="gray", dash="dash"),
                      annotation_text=f"mu_cr = {mu_critical:.3f} (k_cr = {mu_critical**4:.4g})")
    fig.update_layout(
        title=title or f"Unloaded beam, P = {P:g}: branches against the foundation stiffness",
        xaxis_title="mu = k^(1/4)  (k: foundation stiffness)",
        yaxis_title="w(1/4): deflection at x = 1/4", template="plotly_white",
    )
    return fig


def plot_deflection_stack(x, mu, deflections, *, jumps=(), counts=None, n_show=15, title=None):
    """Deflections w(x) stacked by increasing mu, one panel per variant.

    Each curve is shifted upwards by its rank among the n_show values of mu
    shown (evenly picked in `mu`) and labelled with its mu; a grey horizontal
    line marks each jump of the global minimum, placed between the curves
    around it.

    Args:
        x (array-like): sampling in x, shape (N_x,).
        mu (array-like): parameter values, shape (N_mu,).
        deflections (dict): variant name -> deflections, shape (N_x, N_mu).
        jumps (sequence of float): values of mu where the global minimum jumps.
        counts (dict, optional): variant name -> number shown under its panel
            (e.g. side changes within a segment).
        n_show (int): number of deflections drawn per panel.
        title (str, optional): figure title.

    Returns:
        plotly.graph_objects.Figure
    """
    go, make_subplots = _plotly()
    import numpy as np

    x, mu = np.asarray(x, dtype=float), np.asarray(mu, dtype=float)
    shown = np.unique(np.linspace(0, len(mu) - 1, n_show).round().astype(int))
    W_all = {name: np.asarray(W, dtype=float) for name, W in deflections.items()}
    step = 2.2 * max(np.abs(W[:, shown]).max() for W in W_all.values())
    titles = [name if counts is None else f"{name}: {counts[name]} side change(s) within a segment"
              for name in W_all]
    fig = make_subplots(rows=1, cols=len(W_all), shared_yaxes=True, subplot_titles=titles)
    for col, W in enumerate(W_all.values(), start=1):
        for rank, j in enumerate(shown):
            fig.add_trace(go.Scatter(x=x, y=W[:, j] + rank * step, mode="lines", showlegend=False,
                                     line=dict(color="black", width=1.5)), row=1, col=col)
            fig.add_annotation(x=1.0, y=rank * step, text=f"mu = {mu[j]:.2f}", showarrow=False,
                               xanchor="left", font=dict(size=10), row=1, col=col)
        for jump in jumps:
            level = (np.interp(jump, mu[shown], np.arange(len(shown)))) * step
            fig.add_hline(y=level, line=dict(color="gray", width=1, dash="dot"), row=1, col=col)
        fig.update_xaxes(title_text="x", row=1, col=col)
    fig.update_yaxes(showticklabels=False, title_text="w(x), shifted by increasing mu", row=1, col=1)
    fig.update_layout(title=title, template="plotly_white", height=900)
    return fig
