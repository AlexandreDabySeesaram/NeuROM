# NeuROM

**NeuROM** is a finite-element / neural-network library built on the **HiDeNN**
(Hierarchical Deep-learning Neural Network) framework. It expresses a finite
element (FE) interpolation as a neural network whose trainable parameters are
the *nodal values* of the discretised fields, so that **solving a boundary-value
problem becomes minimising a loss** with a standard PyTorch optimiser.

This page introduces the main ideas and the data-flow of the library. The full
API is documented under {doc}`API Reference <api/index>`.

---

## 1. The core idea

In a classical finite element method, a field $\underline{u}$ is approximated on
a mesh by

$$
\underline{u}(\underline{x}) \;=\; \sum_{i=1}^{N_\text{nodes}} N_i(\underline{x})\, \underline{u}_i ,
$$ (eq-interp)

where

- $\underline{x}$ is the physical position,
- $N_\text{nodes}$ is the number of mesh nodes,
- $N_i(\underline{x})$ is the **shape function** attached to node $i$,
- $\underline{u}_i$ is the **nodal value** (degree of freedom) of the field at node $i$.

In the HiDeNN view, equation {eq}`eq-interp` is read as a neural network: the
shape functions $N_i$ form the first layers, and the last (*interpolation*)
layer combines them with weights $\underline{u}_i$. Because the weights of the
interpolation layer **are** the nodal values, training that layer is exactly
equivalent to solving the FE problem on a fixed mesh. Two consequences follow:

- **Dirichlet boundary conditions** are imposed by *freezing* the weights of the
  constrained nodes — see {py:class}`~neurom.constraints.dirichlet.Dirichlet`.
- **Mesh adaptation (r-adaptivity)** is obtained by *also* making the nodal
  *positions* trainable, which is supported by treating positions as a field
  too (see {py:class}`~neurom.fields.field.Field` versus
  {py:class}`~neurom.fields.trainable_field.TrainableField`).

---

## 2. The variational problem solved

Rather than assembling and solving a linear system, NeuROM minimises a
**variational energy**. For a solid-mechanics problem the total potential energy
of the displacement field $\underline{u}$ is

$$
\Pi(\underline{u}) \;=\;
\underbrace{\int_{\Omega} \tfrac{1}{2}\,
\underline{\underline{\sigma}} : \underline{\underline{\varepsilon}}\;\mathrm{d}\Omega}_{\text{stored elastic energy}}
\;-\;
\underbrace{\int_{\Omega} \underline{f}\cdot\underline{u}\;\mathrm{d}\Omega}_{\text{load potential}} ,
$$ (eq-energy)

where

- $\Omega$ is the domain occupied by the body,
- $\underline{\underline{\varepsilon}}$ is the (small-strain) strain tensor,
- $\underline{\underline{\sigma}}$ is the Cauchy stress tensor,
- $\underline{f}$ is the prescribed body-force (load) density,
- $\,:\,$ denotes the double contraction $\underline{\underline{a}}:\underline{\underline{b}} = \sum_{k,l} a_{kl}\,b_{kl}$.

The two terms of {eq}`eq-energy` map directly onto library classes: the elastic
energy is {py:class}`~neurom.physics.solid_elastic_energy.SolidElasticEnergy`
and the load potential is
{py:class}`~neurom.physics.load_potential.LoadPotential`. Both derive from the
common base {py:class}`~neurom.physics.term.Term`.

### Kinematics and constitutive law

The strain is the symmetric part of the displacement gradient,

$$
\underline{\underline{\varepsilon}}(\underline{u}) \;=\;
\tfrac{1}{2}\!\left(\nabla\underline{u} + (\nabla\underline{u})^{\mathsf{T}}\right),
\qquad
\varepsilon_{kl} = \tfrac{1}{2}\!\left(\frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k}\right),
$$ (eq-strain)

computed by {py:func}`~neurom.physics.tensors.green_lagrange_strain`, where the
gradient $\nabla\underline{u}$ is obtained by automatic differentiation through
{py:func}`~neurom.math.jacobian.jacobian`.

The stress follows the linear-elastic (isotropic) constitutive law

$$
\underline{\underline{\sigma}} \;=\;
\lambda\,\operatorname{tr}(\underline{\underline{\varepsilon}})\,\underline{\underline{I}}
\;+\; 2\mu\,\underline{\underline{\varepsilon}} ,
$$ (eq-stress)

implemented by {py:func}`~neurom.physics.tensors.linear_elastic_stress`, where

- $\lambda$ and $\mu$ are the **Lamé parameters** ($\mu$ being the shear modulus),
- $\operatorname{tr}(\underline{\underline{\varepsilon}}) = \sum_k \varepsilon_{kk}$ is the trace (see {py:func}`~neurom.math.trace.trace`),
- $\underline{\underline{I}}$ is the identity tensor (see {py:func}`~neurom.math.identity.identity`).

---

## 3. From integrals to quadrature

The domain integrals in {eq}`eq-energy` are evaluated element-by-element with a
quadrature rule. For any integrand $g$,

$$
\int_{\Omega} g\,\mathrm{d}\Omega \;\approx\;
\sum_{e=1}^{N_e}\;\sum_{q=1}^{N_q}
g\!\left(\underline{x}_{eq}\right)\, w_q\, \bigl|\det \underline{\underline{J}}_e(\underline{\xi}_q)\bigr| ,
$$ (eq-quadrature)

where

- $N_e$ is the number of elements and $N_q$ the number of quadrature points per element,
- $\underline{\xi}_q$ are the quadrature points in the **reference element** with weights $w_q$,
- $\underline{x}_{eq} = \underline{\varphi}_e(\underline{\xi}_q)$ is the physical position obtained from the **isoparametric mapping** $\underline{\varphi}_e$ of element $e$,
- $\underline{\underline{J}}_e = \partial \underline{x}/\partial \underline{\xi}$ is the **Jacobian** of that mapping, so $w_q\,|\det\underline{\underline{J}}_e|$ is the integration **measure** of the quadrature point.

These ingredients correspond to the following components:

| Symbol | Meaning | Class / function |
| --- | --- | --- |
| $N_i,\ N_i(\underline{\xi})$ | shape functions | {py:class}`~neurom.shape_functions.shape_function.ShapeFunction` (e.g. {py:class}`~neurom.shape_functions.linear_triangle.LinearTriangle`) |
| $\underline{\xi}_q,\ w_q$ | quadrature points and weights | {py:class}`~neurom.quadratures.quadrature_rule.QuadratureRule` (e.g. {py:class}`~neurom.quadratures.mid_point_2d.MidPoint2D`) |
| reference geometry | reference element | {py:class}`~neurom.reference_elements.reference_element.ReferenceElement` (e.g. {py:class}`~neurom.reference_elements.triangle.Triangle`) |
| $\underline{\varphi}_e,\ \underline{\underline{J}}_e$ | reference $\to$ physical map | {py:class}`~neurom.geometry.iso_parametric_mapping_2d.IsoparametricMapping2D` |
| $w_q\,\lvert\det\underline{\underline{J}}_e\rvert$ | integration measure | {py:class}`~neurom.interpolation.quadrature_context.QuadratureContext` |
| $\sum_{e,q}$ | the sum in {eq}`eq-quadrature` | {py:func}`~neurom.math.integrate.integrate` |

---

## 4. Fields, samplings and the data layout

Every quantity in the model (displacement, position, load, …) is a **field**.
Fields are stored by name in a {py:class}`~neurom.field_layout.FieldLayout` and
come in three flavours:

- {py:class}`~neurom.fields.field.Field` — fixed nodal values (e.g. coordinates),
- {py:class}`~neurom.fields.trainable_field.TrainableField` — learnable nodal values, optionally subject to a {py:class}`~neurom.constraints.constraint.Constraint`,
- {py:class}`~neurom.fields.element_field.ElementField` — element-wise (per-element) values.

As a field is evaluated through the pipeline it is carried by a
{py:class}`~neurom.samplings.Sampling`, whose tensor has shape
`(*batch_shape, *f_shape)`:

- {py:class}`~neurom.samplings.NodalSampling` — one value per mesh node, batch shape `(N_nodes,)`,
- {py:class}`~neurom.samplings.ElementSampling` — one value per element, batch shape `(N_e,)`,
- {py:class}`~neurom.samplings.QuadratureSampling` — one value per quadrature point, batch shape `(N_e, N_q)`.

Point-wise tensor operations (trace, inner product, transpose, identity,
jacobian, …) are written for a *single* point and lifted to a whole sampling by
{py:func}`~neurom.apply.apply`, which batches the function over the leading
`batch_shape` dimensions with `torch.vmap`.

---

## 5. The end-to-end pipeline

A simulation wires the components above into a single trainable
{py:class}`~neurom.fem_model.FEMModel`:

1. A {py:class}`~neurom.meshes.mesh.Mesh` (built from a {py:class}`~neurom.meshes.connectivity.Connectivity` and a positions field) describes the geometry.
2. A {py:class}`~neurom.interpolation.quadrature_context.QuadratureContext` combines the mesh, a {py:class}`~neurom.quadratures.quadrature_rule.QuadratureRule` and an isoparametric mapping to produce quadrature positions and the integration measure of {eq}`eq-quadrature`.
3. A {py:class}`~neurom.interpolation.quadrature_assembly.QuadratureAssembly` interpolates a field at those quadrature points via {py:class}`~neurom.interpolation.field_interpolator.FieldInterpolator` and equation {eq}`eq-interp`.
4. An {py:class}`~neurom.interpolation.integration_domain.IntegrationDomain` groups the assemblies and refreshes every interpolation on each forward pass.
5. A {py:class}`~neurom.physics_loss.PhysicsLoss` integrates the chosen physics {py:class}`~neurom.physics.term.Term` (equations {eq}`eq-energy`–{eq}`eq-quadrature`) into the scalar energy $\Pi$.

Calling the model returns $\Pi$; an optimiser then minimises it with respect to
the free nodal values (and, for r-adaptivity, the node positions).

### Minimal example

```python
import torch
from neurom.quadratures import MidPoint2D
from neurom.shape_functions import LinearTriangle
from neurom.geometry import IsoparametricMapping2D
from neurom.meshes import Mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import QuadratureContext, QuadratureAssembly, IntegrationDomain
from neurom.physics import SolidElasticEnergy
from neurom.physics.tensors import green_lagrange_strain, linear_elastic_stress_point
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

sf = LinearTriangle()          # shape functions N_i
quad = MidPoint2D()            # quadrature points xi_q and weights w_q

layout = FieldLayout()
u = layout.add(TrainableField(name="displacement", connectivity=connectivity,
                              init_values=u_init,
                              constraint=Dirichlet(nodes=bc_nodes, values_imposed=bc_values)))
x = layout.add(Field(name="positions", connectivity=connectivity, values=points))

mesh = Mesh(connectivity=connectivity, nodes_positions=x)
mapping = IsoparametricMapping2D(sf, mesh)
ctx = QuadratureContext(mesh, quad, mapping)
domain = IntegrationDomain([QuadratureAssembly(ctx, sf, u)])

# Stress sigma = lambda tr(eps) I + 2 mu eps  (Lame parameters lambda, mu)
def stress_point(eps):
    return linear_elastic_stress_point(eps, lame_lambda=1.25, lame_mu=1.0)

physics = SolidElasticEnergy(field=u, strain=green_lagrange_strain, stress_point=stress_point)
model = FEMModel(mesh, layout, domain, PhysicsLoss(physics, layout))

optimizer = torch.optim.LBFGS(model.parameters())

def closure():
    optimizer.zero_grad()
    energy = model()          # Pi(u) from eq. (2)
    energy.backward()
    return energy

optimizer.step(closure)
```

---

```{toctree}
:maxdepth: 2
:caption: Contents

api/index
```
