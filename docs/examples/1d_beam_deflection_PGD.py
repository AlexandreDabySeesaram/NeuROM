"""Parametric 1D bar deflection via CP-PGD -- runnable example.

Solves u(x, E) for a bi-clamped bar under a constant axial load, with the
Young's modulus E treated as an extra (parametric) coordinate. The solution is
sought in separated form u(x, E) = sum_m w_m^x(x) * w_m^E(E) and built greedily,
one mode at a time. See the accompanying ``1d_beam_deflection_PGD.md`` for the
maths. Run directly to train and produce the two figures:

    python 1d_beam_deflection_PGD.py
"""

import torch

from neurom.decompositions import Axis, CPPGD
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import MidPoint1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.interpolation.integration_domain import IntegrationDomain
from neurom.constraints import Dirichlet, NoConstraint
from neurom.differential import jacobian_field
from neurom.inner import inner
from neurom.integrate import integrate
from neurom.field_layout import FieldLayout

torch.set_default_dtype(torch.float32)


def main(n_iter_training=150):
    ## Axis
    # Shape function
    sf = LinearSegment()
    # Quadrature strategy: one Gauss point per element (mid-point rule).
    quad = MidPoint1D()
    # Mapping from/to reference/physical coordinates
    mapping = IsoparametricMapping1D(sf)

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    ## Space
    # Dimensions
    x_min = 0.0
    x_max = 10.0
    N_space = 30

    # Generate vertices and connectivity
    x_array = torch.linspace(x_min, x_max, N_space).unsqueeze(-1)
    nodes_space = torch.arange(0, N_space)
    elements_space = torch.vstack([torch.arange(0, N_space - 1), torch.arange(1, N_space)]).T

    topology_space = Topology(nodes_space, elements_space)
    nodes_positions_space = Field(name=f"space_positions", topology=topology_space, values=x_array)

    # Initialize displacement values
    u_init = 0.5 * torch.ones(N_space, 1)

    axis_space = Axis(name = "space",
                    nodes_positions=nodes_positions_space,
                    sf= sf,
                    mapping=mapping,
                    quad=quad,
                    constraint=Dirichlet( nodes=[0, N_space - 1], values_imposed=torch.zeros(2, 1)),
                    init_values=u_init)

    ## E
    # Dimensions
    E_min = 10.0
    E_max = 100.0
    N_E = 20

    # Generate vertices and connectivity
    E_array = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)
    nodes_E = torch.arange(0, N_E)
    elements_E = torch.vstack([torch.arange(0, N_E - 1), torch.arange(1, N_E)]).T

    topology_E = Topology(nodes_E, elements_E)
    nodes_positions_E = Field(name=f"E_positions", topology=topology_E, values=E_array)

    # Initialize E mode values
    E_init = 0.5 * torch.ones(N_E, 1)

    axis_E = Axis(name = "E",
                    nodes_positions=nodes_positions_E,
                    sf= sf,
                    mapping=mapping,
                    quad=quad,
                    constraint=NoConstraint(),
                    init_values=E_init)

    ## CP PGD object
    pgd_approx = CPPGD(axes=[axis_space, axis_E], n_modes_max=3, name="pgd", n_modes_ini=1)
    # print(pgd_approx.directory())

    ###### Define constant load.
    # The load is a *field* f(x), not a raw nodal vector: it has to be
    # sampled at the SAME quadrature points as u so that inner(f, u) aligns
    # (this is exactly what neurom.physics.LoadPotential does). We give it its
    # own nodal values on the space mesh, then interpolate it once on the
    # space axis's quadrature (same sf / quad / mapping / topology as u).
    # For a load that is constant in E, this is the single rank-1 spatial
    # factor f_0(x) of the separated source f(x, E) = f_0(x) ⊗ 1(E); the E
    # factor "1" is what the Gm = ∫ lmbda dE term below carries implicitly

    load_value = 1000.0 # x^2 ou une autre expression mathématique
    load_field = field_layout.add(Field(name="load", topology=topology_space, values=load_value * torch.ones(N_space, 1)))
    context_f = axis_space.context # le même context que la partie spatiale
    assembly_f = QuadratureAssembly(context_f, sf, load_field)

    # Construction of the shared domain for the whole problem
    domain = IntegrationDomain([*pgd_approx.assemblies(), assembly_f]) # do not forget * to unpack

    # Creer le modele
    model = NeuROMModel(field_layout=field_layout,
                        decomposition=pgd_approx,
                        integration_domain= domain,
                        loss = lambda out: energy(out, pgd_approx, load_name="load"))

    ## add training
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.1,
    )


    def closure():
        optimizer.zero_grad()
        out = model()
        # print(out)
        loss = model.loss(out)
        loss.backward(retain_graph=True)
        return loss

    ### Training (the most basic for now)
    loss_history = []
    # Mode 0
    for _ in range(n_iter_training):
        loss = optimizer.step(closure)
        loss_history.append(loss.detach().item())

    # Mode 1
    pgd_approx.freeze_mode(0)
    pgd_approx.add_mode()                     # active le mode 1
    model.add_mode_to_optimizer(optimizer)

    for _ in range(n_iter_training):
        loss = optimizer.step(closure)
        loss_history.append(loss.detach().item())

    # Mode 2
    pgd_approx.freeze_mode(1)
    pgd_approx.add_mode()                     # active le mode 1
    model.add_mode_to_optimizer(optimizer)

    for _ in range(n_iter_training):
        loss = optimizer.step(closure)
        loss_history.append(loss.detach().item())
    print("Successfully trained!")

    ## Plotting
    plot_convergence(loss_history) # OK
    plot_solution(                              # investigate how to get the information monom per monom
        model, pgd_approx,
        x_min=x_min, x_max=x_max,
        E_min=E_min, E_max=E_max,
        load_value=load_value,
    )


## Plotting helpers
def plot_convergence(loss_history, save_path="pgd_convergence.png"):
    """Plot the (minimised) energy against the training iteration.

    Args:
        loss_history (list[float]): energy value returned by the optimizer at each
            iteration (the PGD functional we minimise).
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(loss_history)), loss_history, "b-")
    ax.set_title("Training convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy (loss)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()

def plot_solution(model, pgd_approx, *, x_min, x_max, E_min, E_max, load_value,
                  save_path="pgd_vs_analytical.png"):
    """Compare the PGD solution to the analytical beam deflection.

    Four panels: (1) the full solution u(x, E) at a fixed E, PGD vs analytical;
    (2) same at a fixed x, swept over E; (3) the space factor of every mode;
    (4) the E factor of every mode. See the per-panel comments for the scaling
    caveat on the mode-by-mode factors.

    Args:
        model (NeuROMModel): trained model (put in eval mode here).
        pgd_approx (CPPGD): the decomposition, to read per-mode / per-axis factors.
        x_min, x_max, E_min, E_max (float): axis bounds.
        load_value (float): constant load q used in the analytical formula.
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    model.eval()

    # --- helper: evaluate a single factor w_m^k(.) at arbitrary points ------
    # A PGD mode is u_m(x, E) = w_m^space(x) * w_m^E(E). Each factor is one
    # monom field of the decomposition; PointWiseInterpolator evaluates it on
    # its own axis mesh at arbitrary query points (same tool the decomposition
    # uses internally in evaluate()/assemble()).
    def factor(m, k, pts):
        pwi = PointWiseInterpolator(
            pgd_approx.axes[k].mesh,
            pgd_approx.axes[k].sf,
            pgd_approx.monoms[m][k],
            pgd_approx.axes[k].mapping,
        )
        return pwi.at_position(pts.reshape(-1)).reshape(-1)

    def norm(v):
        m = v.abs().max()
        return v / m if m > 0 else v

    n_modes = pgd_approx.n_modes_truncated

    # --- 1) FULL solution: PGD (sum of modes) vs analytical, at fixed E -----
    # This is the unambiguous comparison. The analytical deflection is
    #   u(x, E) = 0.5 q (x - x_min)(x - x_max) / E,
    # which is itself rank-1 separable: a space parabola times 1/E.
    E_fixed = E_max / 2
    x_plot = torch.linspace(x_min, x_max, 200)
    E_plot = E_fixed * torch.ones_like(x_plot)
    u_pgd_full = model(torch.stack([x_plot, E_plot], dim=1)).reshape(-1)  # sum_m w_m^x w_m^E
    u_ana_full = 0.5 * load_value * (x_plot - x_min) * (x_plot - x_max) / E_plot

    # Counterpart of panel 1: fix x at mid-span, sweep E. Same diagonal
    # evaluate (each E paired with the same x_fixed), so we see the parametric
    # dependence u(x_fixed, .) -- a 1/E curve -- rather than the spatial parabola.
    x_fixed = (x_min + x_max) / 2
    E_sweep = torch.linspace(E_min, E_max, 200)
    x_sweep = x_fixed * torch.ones_like(E_sweep)
    u_pgd_E = model(torch.stack([x_sweep, E_sweep], dim=1)).reshape(-1)
    u_ana_E = 0.5 * load_value * (x_fixed - x_min) * (x_fixed - x_max) / E_sweep

    # --- 2) MODE per MODE, AXIS per AXIS -----------------------------------
    # Caveat: individual factors carry an arbitrary scale (w_m^x -> c w_m^x,
    # w_m^E -> w_m^E / c leaves the product unchanged), so only their SHAPE is
    # comparable per axis, not the amplitude. We therefore normalise each
    # curve by its max |.|. The analytical solution is rank-1, so its space
    # shape is (x-x_min)(x-x_max) and its E shape is 1/E -- overlaid as the
    # reference each PGD mode is trying to capture (mode 0 should match it,
    # higher modes should be ~0 since the truth has a single mode).
    x_fac = torch.linspace(x_min, x_max, 200)
    E_fac = torch.linspace(E_min, E_max, 200)
    ana_x_shape = (x_fac - x_min) * (x_fac - x_max)
    ana_E_shape = 1.0 / E_fac

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))

    # Full pgd vs analytical at fixed E (sweep x)
    ax[0].plot(x_plot.numpy(), u_ana_full.numpy(), "k-", lw=2, label="analytical")
    ax[0].plot(x_plot.numpy(), u_pgd_full.detach().numpy(), "r--", lw=2,
               label="PGD (sum of modes)")
    ax[0].set_title(f"Full solution u(x, E={E_fixed:.0f})")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("u"); ax[0].legend()

    # Full pgd vs analytical at fixed x (sweep E) -- counterpart of panel 0
    ax[1].plot(E_sweep.numpy(), u_ana_E.numpy(), "k-", lw=2, label="analytical")
    ax[1].plot(E_sweep.numpy(), u_pgd_E.detach().numpy(), "r--", lw=2,
               label="PGD (sum of modes)")
    ax[1].set_title(f"Full solution u(x={x_fixed:.0f}, E)")
    ax[1].set_xlabel("E"); ax[1].set_ylabel("u"); ax[1].legend()

    # approx space (per mode)
    for m in range(n_modes):
        ax[2].plot(x_fac.numpy(), norm(factor(m, 0, x_fac)).detach().numpy(),
                   label=f"PGD mode {m}")

    # analytical space
    ax[2].plot(x_fac.numpy(), norm(ana_x_shape).numpy(), "k:", lw=2,
               label="analytical shape")
    ax[2].set_title("Space factor w_m^x(x)  (normalised shape)")
    ax[2].set_xlabel("x"); ax[2].legend()

    # approx E (per mode)
    for m in range(n_modes):
        ax[3].plot(E_fac.numpy(), norm(factor(m, 1, E_fac)).detach().numpy(),
                   label=f"PGD mode {m}")

    # analytical E
    ax[3].plot(E_fac.numpy(), norm(ana_E_shape).numpy(), "k:", lw=2,
               label="analytical 1/E shape")
    ax[3].set_title("E factor w_m^E(E)  (normalised shape)")
    ax[3].set_xlabel("E"); ax[3].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()


 ## Energy

def energy(field_layout:FieldLayout, decomposition: any, load_name : str):
    # on va chercher les noms des champs {'space': ['pgd_dimspace_mode0'], 'E': ['pgd_dimE_mode0']}
    directory = decomposition.directory()
    n_modes = len(directory['space'])

    # on les récupère dans le field_layout
    space_modes_names =  directory['space']
    E_modes_names = directory['E']
    space_modes = [field_layout[name] for name in space_modes_names]
    E_modes = [field_layout[name] for name in E_modes_names]

    # et le load
    load_field = field_layout[load_name]


    ## Elastic
    elastic = 0.0
    # param part
    E_val = [E_mode_field.x for E_mode_field in E_modes]
    lmbdas = [E_mode_field.u for E_mode_field in E_modes]
    J_E = [E_mode_field.measure for E_mode_field in E_modes]

    # space part
    u = [space_mode_field.u for space_mode_field in space_modes]
    x_val = [space_mode_field.x for space_mode_field in space_modes]
    # jacobian_field returns (N_e, N_q, *u_shape, d): one extra trailing axis of
    # size d (the physical dimension) compared to u's own shape (N_e, N_q,
    # *u_shape). That axis must be *contracted away* -- the elastic term is the
    # scalar product grad(u_m) . grad(u_n) summed over the spatial directions --
    # so we keep the raw jacobian_field output here (no reshape) and let inner()
    # perform the contraction in the loop below. See Kx there.
    #
    # This replaces an earlier `.reshape(u[n].shape)` on the line below. That
    # reshape only appeared to work in 1D: with d=1 the extra axis has size 1 and
    # reshape could drop it, but it left `grad_u[m] * grad_u[n] * J_u[m]` as a
    # plain element-wise product -- a broadcasting trap. With MidPoint1D (N_q=1)
    # that silently built an (N_e, N_e, 1, 1) outer product over *elements*
    # (numerically wrong energy, no exception); with N_q>1 (e.g. TwoPoints1D) the
    # shapes don't broadcast and it hard-failed with "size of tensor a (2) must
    # match size of tensor b (N_e)"; and in 2D/3D (d>1) reshape can't collapse the
    # axis at all. Using inner() instead is correct AND dimension-agnostic.
    grad_u = [jacobian_field(x=x_val[n], u=u[n]) for n in range(n_modes)]
    J_u = [space_mode_field.measure for space_mode_field in space_modes]

    # NB: cross terms (m, n) below assume modes m and n share the same mesh
    # (measure/coords indexed by m are used for both). Once modes can live on
    # independent (e.g. r-adapted) meshes, these products need to be
    # integrated on a common intersection mesh with a recomputed measure.
    for m in range(n_modes):
        for n in range(n_modes):

            # inner() contracts grad(u_m) . grad(u_n) over the field and d axes,
            # returning (N_e, N_q, 1) -- same rank as the measure J_u -- so the
            # `* J_u[m]` below aligns element-wise as intended (no reshape needed).
            Kx = integrate(inner(grad_u[m], grad_u[n]) * J_u[m])
            AE = integrate(E_val[m] * lmbdas[m] * lmbdas[n] * J_E[m])
            elastic = elastic + Kx * AE
    elastic = 0.5 * elastic
    load = 0.0
    # load_interp.u is the load field sampled at the space quadrature points, so
    # it has the same (N_e, N_q, *u_shape) shape as u[m]: inner() contracts them
    # into (N_e, N_q, 1) and the * J_u[m] measure aligns element-wise -- unlike
    # the old raw nodal `external_load_values` (N_space, 1), which broadcast wrong
    # against quadrature-point values (silently with N_q=1, crashing with N_q>1).
    # Gm = ∫ lmbda dE carries the constant-in-E factor of the separated load.
    load_f = load_field.u
    for m in range(n_modes):
        Fx = integrate(inner(load_f, u[m]) * J_u[m])
        Gm = integrate(lmbdas[m] * J_E[m])
        load = load + Fx * Gm

    return elastic + load

if __name__ == "__main__":
    main(150)
