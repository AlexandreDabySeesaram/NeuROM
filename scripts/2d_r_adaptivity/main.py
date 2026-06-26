"""2D plate-with-hole simulation with r-adaptivity (moving mesh).

This example mirrors ``scripts/2d/main.py`` (a linear-elastic plate-with-hole
under a prescribed top displacement), but additionally makes the **mesh node
positions trainable**. Minimising the elastic potential energy with respect to
the interior node positions performs *r-adaptivity*: the mesh relocates its
nodes to reduce the energy, without changing the mesh topology.

Compared to the fixed-mesh example, three things change:

1. ``positions`` is a :class:`~neurom.fields.trainable_field.TrainableField`
   (instead of a fixed :class:`~neurom.fields.field.Field`), with all boundary
   nodes pinned by a :class:`~neurom.constraints.dirichlet.Dirichlet`
   constraint so the domain shape (outer edges and the hole) is preserved.
2. A ``FlipLoss`` barrier penalises elements whose signed area becomes
   non-positive, preventing the moving mesh from tangling (inverted triangles).
3. Because the geometry changes every step, ``domain.update_contexts()`` is
   called before each forward pass to recompute the mapping, quadrature
   positions and integration measure.

Training proceeds in two stages: first the displacement and the mesh are
optimised jointly (r-adaptivity), then the mesh is frozen and the displacement
is refined on the adapted mesh.

The mesh file is produced by ``scripts/2d/generate_mesh.py``; run it first, e.g.::

    uv run python scripts/2d/generate_mesh.py -o scripts/2d_r_adaptivity/
    uv run python scripts/2d_r_adaptivity/main.py -i scripts/2d_r_adaptivity/plate_with_hole.xdmf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Import library modules
from neurom.quadratures import MidPoint2D
from neurom.shape_functions import LinearTriangle
from neurom.geometry import IsoparametricMapping2D
from neurom.meshes import Mesh
from neurom.constraints import Dirichlet
from neurom.fields import TrainableField, ElementField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import SolidElasticEnergy
from neurom.physics.tensors import (
    jacobian,
    green_lagrange_strain,
    linear_elastic_stress,
    linear_elastic_stress_point,
    stress_deviator,
    stress_von_mises,
)
from neurom.samplings import Sampling
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

from neurom.meshes.io import read_mesh, write_mesh
from neurom.meshes.validity import is_valid_mesh

torch.set_default_dtype(torch.float32)


class FlipLoss(nn.Module):
    """Barrier penalising inverted (tangled) triangular elements.

    For each linear triangle with node positions ``a``, ``b``, ``c`` the signed
    area is proportional to the cross product ``(b - a) x (c - a)``, which equals
    the Jacobian determinant of the reference-to-physical mapping. A valid mesh
    keeps this quantity strictly positive; the loss adds a quadratic penalty
    whenever it drops below ``margin``, acting as a barrier against element
    inversion when the mesh moves.

    Attributes:
        x (TrainableField): The trainable positions field whose per-element
            node coordinates define the element geometry.
        mu (float): Penalty weight of the barrier.
        margin (float): Lower bound below which the signed area is penalised.
    """

    def __init__(self, x: TrainableField, mu: float, margin: float = 0.0):
        """Initialise the flip barrier.

        Args:
            x (TrainableField): The trainable positions field.
            mu (float): Penalty weight of the barrier.
            margin (float): Signed-area threshold below which elements are
                penalised. Use ``0.0`` to penalise only actual inversions.
        """
        super().__init__()
        self.x = x
        self.mu = mu
        self.margin = margin

    def forward(self) -> torch.Tensor:
        """Compute the flip-barrier penalty.

        Returns:
            torch.Tensor: Scalar penalty summed over all elements whose signed
            area is below ``margin``.
        """
        # Per-element node positions, shape (N_e, 3, 2).
        x_nodes = self.x.at_elements()
        ab = x_nodes[:, 1, :] - x_nodes[:, 0, :]
        ac = x_nodes[:, 2, :] - x_nodes[:, 0, :]
        # Signed area * 2 = det of the Jacobian, shape (N_e,).
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        return self.mu * torch.sum(torch.relu(self.margin - det) ** 2)


class TotalLoss(nn.Module):
    """Sum of several loss modules evaluated together.

    Attributes:
        losses (list): The loss modules whose scalar outputs are summed.
    """

    def __init__(self, losses):
        """Initialise the composite loss.

        Args:
            losses (list): Callable loss modules returning scalar tensors.
        """
        super().__init__()
        self.losses = losses

    def forward(self) -> torch.Tensor:
        """Evaluate and sum all sub-losses.

        Returns:
            torch.Tensor: The summed scalar loss.
        """
        return sum(loss_term.forward() for loss_term in self.losses)


def boundary_nodes(connectivity) -> torch.Tensor:
    """Find the indices of nodes lying on the mesh boundary.

    A boundary edge of a triangular mesh belongs to exactly one element. The
    boundary nodes are the unique endpoints of all such edges (this captures
    both the outer edges and the hole boundary).

    Args:
        connectivity (Connectivity): Mesh connectivity with triangular elements.

    Returns:
        torch.Tensor: Sorted, unique indices of the boundary nodes.
    """
    elements = connectivity.element_connectivity  # (N_e, 3)
    edges = torch.cat(
        [elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]], dim=0
    )
    edges = torch.sort(edges, dim=1).values
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    return torch.unique(boundary.flatten())


def project_to_elements(s: Sampling) -> torch.Tensor:
    """Average quadrature-point values per element.

    Useful for VTK cell-data output of quadrature-level fields (e.g. stress).

    Args:
        s (Sampling): A quadrature-level sampling of shape ``(N_e, N_q, ...)``.

    Returns:
        torch.Tensor: Per-element averaged values of shape ``(N_e, ...)``.
    """
    return s.values.mean(dim=1)


def main():
    """Run the r-adaptive 2D plate-with-hole simulation."""
    parser = argparse.ArgumentParser(description="2D r-adaptivity simulation.")
    parser.add_argument(
        "-i",
        "--input-mesh",
        type=Path,
        default="./plate_with_hole.xdmf",
        help="Mesh in xdmf format (see scripts/2d/generate_mesh.py).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="./",
        help="Output directory where the files will be written.",
    )
    parser.add_argument(
        "--epochs-adapt",
        type=int,
        default=300,
        help="Joint displacement + mesh (r-adaptivity) optimisation steps.",
    )
    parser.add_argument(
        "--epochs-refine",
        type=int,
        default=5,
        help="Displacement-only refinement steps on the frozen adapted mesh.",
    )
    parser.add_argument(
        "--flip-penalty",
        type=float,
        default=1e6,
        help="Weight of the FlipLoss barrier against inverted elements.",
    )
    parser.add_argument(
        "--lr-adapt",
        type=float,
        default=1e-2,
        help="Adam learning rate for the joint r-adaptivity stage.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show plots.")

    args = parser.parse_args()
    output_dir = args.output_dir

    lame_lambda = 1.25
    lame_mu = 1.0

    # Read mesh
    connectivity, data = read_mesh(args.input_mesh)

    # Get positions - restrict to 2D
    points = data["x"][:, 0:2].to(torch.float32)
    N = points.shape[0]

    # --- Displacement boundary conditions (same as the fixed-mesh example) ---
    # Tags: 3 == top, 1 == bottom
    dim_tags = data["point_data"]["gmsh:dim_tags"]
    mask_top = torch.logical_and(dim_tags[:, 1] == 3, dim_tags[:, 0] == 1)
    mask_bottom = torch.logical_and(dim_tags[:, 1] == 1, dim_tags[:, 0] == 1)

    nodes_top = connectivity.nodes_indices[mask_top]
    nodes_bottom = connectivity.nodes_indices[mask_bottom]

    u_top = torch.tensor([0.0, -1.0]).expand(nodes_top.shape[0], 2)
    u_bottom = torch.tensor([0.0, 0.0]).expand(nodes_bottom.shape[0], 2)

    # Dirichlet.expand fills constrained DOFs in ascending node-index order,
    # so the imposed values must be sorted accordingly.
    nodes_u_bc = torch.cat([nodes_top, nodes_bottom])
    u_bc = torch.cat([u_top, u_bottom])
    order = torch.argsort(nodes_u_bc)
    nodes_u_bc = nodes_u_bc[order]
    u_bc = u_bc[order]

    # --- Position boundary conditions (r-adaptivity) ---
    # Pin every boundary node to its initial position so the domain shape (outer
    # edges and the hole) is preserved while interior nodes are free to move.
    nodes_x_bc = boundary_nodes(connectivity)  # already sorted ascending
    x_bc = points[nodes_x_bc]

    # Initialize displacement value
    u_init = 0.1 * torch.ones(N, 2)

    # Define shape function to use
    sf = LinearTriangle()
    # Define quadrature method
    quad = MidPoint2D()

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    # Displacement (trainable, with prescribed top/bottom displacement)
    u = field_layout.add(
        TrainableField(
            name="displacement",
            connectivity=connectivity,
            init_values=u_init,
            constraint=Dirichlet(nodes=nodes_u_bc, values_imposed=u_bc),
        )
    )

    # Positions (trainable -> r-adaptivity, with boundary nodes pinned)
    x = field_layout.add(
        TrainableField(
            name="positions",
            connectivity=connectivity,
            init_values=points,
            constraint=Dirichlet(nodes=nodes_x_bc, values_imposed=x_bc),
        )
    )

    # Generate mesh
    mesh = Mesh(connectivity=connectivity, nodes_positions=x)
    if not is_valid_mesh(mesh):
        raise ValueError("There is an issue with the processed mesh.")

    # Write init mesh with all fields
    write_mesh(output_dir / "init.xdmf", mesh, field_layout)

    # Define mapping (depends on the - now trainable - positions)
    mapping = IsoparametricMapping2D(sf, mesh)

    # Define interpolation at quadrature points
    ctx = QuadratureContext(mesh, quad, mapping)
    assembly_u = QuadratureAssembly(ctx, sf, u)
    domain = IntegrationDomain([assembly_u])

    def stress(strain):
        return linear_elastic_stress_point(strain, lame_lambda, lame_mu)

    # Physics: elastic potential energy of the displacement field
    physics = SolidElasticEnergy(
        field=u,
        strain=green_lagrange_strain,
        stress_point=stress,
    )
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    # Total loss = elastic energy + flip barrier (keeps the moving mesh valid)
    flip_loss = FlipLoss(x=x, mu=args.flip_penalty)
    total_loss = TotalLoss([physics_loss, flip_loss])

    # FEM model with the composite loss
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=total_loss,
    )

    loss_history = []

    # --- Stage 1: joint displacement + mesh optimisation (r-adaptivity) ---
    print("* Stage 1: r-adaptivity (displacement + mesh)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_adapt)
    for i in range(args.epochs_adapt):
        # Positions are trainable: recompute the geometry before interpolation.
        domain.update_contexts()
        loss = model()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_history.append(loss.item())
        print(f"  {i=} loss={loss.item():.3e}", end="\r")
    print()

    # --- Stage 2: freeze the adapted mesh and refine the displacement ---
    print("* Stage 2: displacement refinement on the adapted mesh")
    x.freeze()
    domain.update_contexts()

    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=physics_loss,
    )
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1e-1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = model()
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        return loss

    for i in range(args.epochs_refine):
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        print(f"  {i=} loss={loss.item():.3e}", end="\r")
    print()

    # --- Post-processing: recover stress fields on the adapted mesh ---
    print("* Evaluation")
    result = assembly_u.interpolate()
    x_final = result.x
    u_final = result.u

    grad_u = jacobian(x_final, u_final)
    strain = green_lagrange_strain(x_final, u_final)
    sigma = linear_elastic_stress(strain, lame_lambda, lame_mu)
    sigma_dev = stress_deviator(sigma)
    von_mises = stress_von_mises(sigma_dev)

    field_layout.add(ElementField(name="grad_u", values=project_to_elements(grad_u)))
    field_layout.add(ElementField(name="strain", values=project_to_elements(strain)))
    field_layout.add(ElementField(name="sigma", values=project_to_elements(sigma)))
    field_layout.add(
        ElementField(name="sigma_dev", values=project_to_elements(sigma_dev))
    )
    field_layout.add(
        ElementField(name="von_mises", values=project_to_elements(von_mises))
    )

    # Write final (adapted) mesh with all fields
    write_mesh(output_dir / "result.xdmf", mesh, field_layout)

    plt.figure()
    plt.semilogy(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss (r-adaptivity then refinement)")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
