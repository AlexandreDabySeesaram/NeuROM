# The Piola maps: pushing vector fields from the reference cell to the physical cell

This chapter derives, from first principles, the three *push-forward* rules the library uses to
turn a basis value defined on the **reference cell** into a field value on a **physical cell**:

$$
\underbrace{A_e^{\mathrm{id}} = I}_{\text{point values}}, \qquad
\underbrace{A_e^{\mathrm{cov}} = J_e^{-\top}}_{\text{tangential (curl)}}, \qquad
\underbrace{A_e^{\mathrm{con}} = \tfrac{1}{\det J_e}\,J_e}_{\text{normal flux (div)}} .
$$

The companion chapter {doc}`dof_transformations` treated the *scalar / Hermite* case — how a
**value or a derivative** DOF transforms under the mapping, giving the DOF matrix
$\underline{\underline{M}}_e$. This chapter is the **vector-field** generalisation: when the quantity
being interpolated is a *direction-carrying* field (a velocity whose circulation matters, a flux
whose normal component matters), the naive "compose with $\phi_e^{-1}$" rule is **wrong**, and the
correct rule is a *Piola map*. These are exactly the maps encoded by
{py:class}`~neurom.elements.finite_element.MapType` and applied by
{py:func}`~neurom.dof_transformations.reference_to_physical.reference_to_physical`.

Everything below assumes the library's geometry convention
$\phi_e:\hat K\to K$, $\;J_e=\partial x/\partial\xi$ (as in deal.II / DOLFINx / UFL), with the affine
simplex mapping {py:class}`~neurom.geometry.affine_simplex_mapping.AffineSimplexMapping`, so
$J_e$ is **constant on each cell**.

---

## 1. The problem: why composition is not enough

For a **scalar** field $f$ the push-forward is just composition: the physical field and the reference
field are the same number at corresponding points,

$$
f(x) = \hat f\big(\phi_e^{-1}(x)\big), \qquad \hat f(\xi) = f\big(\phi_e(\xi)\big).
$$ (eq-scalar)

That is the `IDENTITY` map, $A_e = I$: a vertex value on $\hat K$ *is* the vertex value on $K$.

Now take a **vector** field $v$. We could try to transport it componentwise by the same rule,
$v_i(x) = \hat v_i(\xi)$. This keeps the Cartesian components fixed, but it does **not** keep the
quantities that make a vector field physically meaningful on a mesh:

- for an **edge (H(curl)) element** the degree of freedom is a *circulation* $\int_E v\cdot t\,\mathrm ds$
  along an edge — the **tangential** component must match across a shared edge;
- for a **face (H(div)) element** the degree of freedom is a *flux* $\int_E v\cdot n\,\mathrm ds$
  through an edge — the **normal** component must match across a shared edge.

Under a general (even affine) $\phi_e$, edges are stretched and sheared, so tangents and normals on
$K$ are **not** the Cartesian images of those on $\hat K$. If we transported components naively, two
cells sharing an edge would disagree on the tangential/normal component and the discrete space would
**not be conforming**. The Piola maps are precisely the transformation rules that keep circulation
(covariant) or flux (contravariant) invariant. To derive them we first need to know how the three
geometric objects — tangents, gradients, and area/normal elements — transform.

---

## 2. First principles: how geometry transforms under $\phi_e$

Throughout, a reference object wears a hat ($\hat t,\ \hat n,\ \hat\nabla$) and its physical image does
not. Because the mapping is affine,

$$
x = \phi_e(\xi) = x_0 + J_e\,\xi, \qquad J_e = \frac{\partial x}{\partial\xi}\ \text{constant},
\qquad \xi = \phi_e^{-1}(x) = J_e^{-1}(x-x_0).
$$ (eq-affine)

### 2.1 Tangent vectors transform by $J_e$

Let $\hat\gamma(s)$ be a curve in $\hat K$ and $\gamma(s)=\phi_e(\hat\gamma(s))$ its image in $K$. By the
chain rule and {eq}`eq-affine`,

$$
\gamma'(s) = \frac{\mathrm d}{\mathrm ds}\,\phi_e(\hat\gamma(s)) = J_e\,\hat\gamma'(s).
$$ (eq-tangent)

So a reference tangent $\hat t$ pushes forward to the physical tangent

$$
\boxed{\,t = J_e\,\hat t\,}.
$$

A line element follows: $\mathrm d l = J_e\,\mathrm{d}\hat l$. **Tangents ride along with $J_e$.**

### 2.2 Gradients (covectors) transform by $J_e^{-\top}$

Let $f:K\to\mathbb R$ and $\hat f=f\circ\phi_e$ its pullback. Component by component,

$$
\frac{\partial\hat f}{\partial\xi_j}
= \sum_i \frac{\partial f}{\partial x_i}\,\frac{\partial x_i}{\partial\xi_j}
= \sum_i (J_e)_{ij}\,\frac{\partial f}{\partial x_i}
\quad\Longleftrightarrow\quad
\hat\nabla\hat f = J_e^{\top}\,\nabla f .
$$ (eq-grad-pull)

Inverting,

$$
\boxed{\,\nabla f = J_e^{-\top}\,\hat\nabla\hat f\,}.
$$

A gradient is the archetypal *covector*: it transforms **contravariantly to a tangent** — by
$J_e^{-\top}$ instead of $J_e$. This is the seed of the covariant Piola map: any field that should
behave like a gradient (a tangential field) must transform by $J_e^{-\top}$.

### 2.3 The cofactor identity and Nanson's formula (normals and areas)

Normals are the subtle case. A normal is defined *against* tangents (it is orthogonal to the surface),
so it transforms like neither a tangent nor a plain gradient. The key algebraic fact is the
**cofactor identity**, valid for any invertible $J$:

$$
\operatorname{cof}(J) \;=\; \det(J)\,J^{-\top},
\qquad\text{equivalently}\qquad
J^{\top}\operatorname{cof}(J) = \det(J)\,I,
$$ (eq-cofactor)

where $\operatorname{cof}(J)_{ij}$ is the signed $(i,j)$ cofactor. Consider an oriented surface element
on $\hat K$ with area-normal vector $\hat n\,\mathrm d\hat a$ (direction = unit normal, length = area).
Its image under $\phi_e$ has area-normal

$$
\boxed{\,n\,\mathrm d a = \operatorname{cof}(J_e)\,\hat n\,\mathrm d\hat a = \det(J_e)\,J_e^{-\top}\,\hat n\,\mathrm d\hat a\,}
$$ (eq-nanson)

which is **Nanson's formula**. One way to see it: in 3-D a surface element is spanned by two tangents
$\hat a,\hat b$, with $\hat n\,\mathrm d\hat a = \hat a\times\hat b$; the images are $J_e\hat a,\ J_e\hat b$,
and the vector identity $(J\hat a)\times(J\hat b) = \operatorname{cof}(J)\,(\hat a\times\hat b)$ gives
{eq}`eq-nanson` directly. The determinant makes sense dimensionally: areas scale like $\det J_e$,
while the $J_e^{-\top}$ rotates the normal back to orthogonality with the sheared tangents.

#### The 2-D specialisation used in the code

In 2-D the library builds a normal by **rotating the tangent** (Rognes–Kirby–Logg; Kirby 2018):
with the quarter-turn

$$
R = \begin{bmatrix} 0 & -1\\ 1 & 0 \end{bmatrix}, \qquad n = R\,t ,
$$ (eq-rot)

the cofactor identity becomes a clean rotation identity. For any $2\times2$ matrix
$J=\begin{bmatrix} a & b\\ c & d\end{bmatrix}$ a direct computation gives

$$
J^{\top} R\,J
= \begin{bmatrix} a & c\\ b & d\end{bmatrix}
  \begin{bmatrix} 0 & -1\\ 1 & 0\end{bmatrix}
  \begin{bmatrix} a & b\\ c & d\end{bmatrix}
= (ad-bc)\begin{bmatrix} 0 & -1\\ 1 & 0\end{bmatrix}
= \det(J)\,R .
$$ (eq-JTRJ)

So

$$
\boxed{\,J_e^{\top} R\,J_e = \det(J_e)\,R\,}
$$

is the $2\times2$ form of Nanson's formula; it is the single identity that makes the contravariant map
preserve fluxes, and it is the identity exercised by the push-forward unit tests.

---

## 3. The covariant Piola map (tangential / H(curl))

**Requirement.** The circulation of the field along any edge must be the same whether computed on $K$
or on $\hat K$:

$$
\int_{E} v\cdot t\,\mathrm d s \;=\; \int_{\hat E} \hat v\cdot \hat t\,\mathrm d\hat s
\quad\text{for every edge } E=\phi_e(\hat E).
$$ (eq-circ-req)

**Ansatz.** Look for a constant (per cell) matrix $A_e$ with $v(x)=A_e\,\hat v(\xi)$. Using the tangent
law {eq}`eq-tangent`, $t\,\mathrm d s = J_e\,\hat t\,\mathrm d\hat s$, the left-hand integrand is

$$
v\cdot t\,\mathrm d s
= (A_e\hat v)^{\top}(J_e\hat t)\,\mathrm d\hat s
= \hat v^{\top}\big(A_e^{\top} J_e\big)\hat t\,\mathrm d\hat s .
$$

For this to equal $\hat v\cdot\hat t\,\mathrm d\hat s = \hat v^{\top}\hat t\,\mathrm d\hat s$ for **all**
$\hat v,\hat t$ we need $A_e^{\top}J_e = I$, i.e.

$$
\boxed{\,A_e^{\mathrm{cov}} = J_e^{-\top}\,}, \qquad v = J_e^{-\top}\,\hat v .
$$ (eq-covariant)

This is the **covariant Piola map**. It is the vector version of the gradient law {eq}`eq-grad-pull`:
a field that should behave tangentially transforms exactly as a gradient does. It preserves
$\nabla\times$ conformity, hence it is the map of **Nédélec / edge elements**, and it is
`MapType.COVARIANT_PIOLA` in the code.

**Check (the unit test).** With $t=J_e\hat t$ the identity above reads
$(J_e^{-\top}\hat v)\cdot(J_e\hat t) = \hat v\cdot\hat t$ — the tangential inner product is invariant,
which is precisely what `test_covariant_preserves_tangential_inner_product` asserts.

---

## 4. The contravariant Piola map (normal flux / H(div))

**Requirement.** The flux of the field through any edge/face must be preserved:

$$
\int_{\partial K} v\cdot n\,\mathrm d a \;=\; \int_{\partial\hat K} \hat v\cdot \hat n\,\mathrm d\hat a .
$$ (eq-flux-req)

**Ansatz.** Again write $v=A_e\hat v$ and substitute Nanson's formula {eq}`eq-nanson`,
$n\,\mathrm d a = \det(J_e)J_e^{-\top}\hat n\,\mathrm d\hat a$:

$$
v\cdot n\,\mathrm d a
= (A_e\hat v)^{\top}\big(\det(J_e)\,J_e^{-\top}\hat n\big)\,\mathrm d\hat a
= \det(J_e)\,\hat v^{\top}\big(A_e^{\top}J_e^{-\top}\big)\hat n\,\mathrm d\hat a .
$$

For this to equal $\hat v^{\top}\hat n\,\mathrm d\hat a$ for all $\hat v,\hat n$ we need
$\det(J_e)\,A_e^{\top}J_e^{-\top} = I$, i.e. $A_e^{\top} = \tfrac{1}{\det J_e}J_e^{\top}$, hence

$$
\boxed{\,A_e^{\mathrm{con}} = \frac{1}{\det J_e}\,J_e\,}, \qquad v = \frac{1}{\det J_e}\,J_e\,\hat v .
$$ (eq-contravariant)

This is the **contravariant Piola map**: it preserves $\nabla\cdot$ conformity, so it is the map of
**Raviart–Thomas / face elements**, and it is `MapType.CONTRAVARIANT_PIOLA` in the code.

### 4.1 The 2-D flux identity (what the code and test use)

In 2-D with $n=R\,t$ {eq}`eq-rot`, the physical edge tangent is $t=J_e\hat t$, so the physical normal is
$n=R\,J_e\hat t$. The flux density of a contravariant-mapped field is

$$
v\cdot n
= \Big(\tfrac{1}{\det J_e}J_e\hat v\Big)^{\top}\!\big(R\,J_e\hat t\big)
= \tfrac{1}{\det J_e}\,\hat v^{\top}\big(J_e^{\top} R\,J_e\big)\hat t
= \tfrac{1}{\det J_e}\,\hat v^{\top}\big(\det J_e\,R\big)\hat t
= \hat v\cdot(R\,\hat t),
$$ (eq-flux-2d)

where the third equality is the identity {eq}`eq-JTRJ`. The flux through $R\,t$ is invariant —
`test_contravariant_preserves_normal_flux_signed`.

### 4.2 Why the *signed* determinant (not $|\det J_e|$)

This is the single most error-prone point, so it deserves its own argument. Equation {eq}`eq-flux-2d`
used the **signed** $\det J_e$, and the $\det J_e$ in the numerator of $A_e^{\mathrm{con}}$ cancelled the
$\det J_e$ produced by $J_e^{\top}RJ_e$. If instead we (incorrectly) used $|\det J_e|$ in the map,

$$
v\cdot n
= \tfrac{1}{|\det J_e|}\,\hat v^{\top}(\det J_e\,R)\hat t
= \operatorname{sign}(\det J_e)\;\hat v\cdot(R\hat t) .
$$

On a **counter-clockwise** cell $\det J_e>0$ and this is fine; on a **clockwise** (inverted) cell
$\det J_e<0$ and the flux is **negated** — the flux "matrix" becomes $-I$ on half the mesh, breaking
continuity on every interior edge where a CCW and a CW cell meet. This is the ablation
`test_contravariant_ablation_absdet_flips_sign_on_cw`.

A clean way to hold the two uses of the determinant apart:

> **Integration measures use $|\det J_e|$** (a volume is positive — see {eq}`eq-quadrature` in the
> overview). **The contravariant map uses the signed $\det J_e$** (it carries orientation). They are
> different roles of the same number; conflating them is the classic RT0 bug.

### 4.3 The divergence identity

The contravariant map has a second signature property, used to compute divergences by plain autograd
and to prove flux preservation through the divergence theorem. For $v=\tfrac{1}{\det J_e}J_e\hat v$ with
$J_e$ **constant** (affine), chain-rule each component
$\partial x_i/\partial\xi \Rightarrow (J_e^{-1})_{mi}$:

$$
\operatorname{div}_x v
= \sum_i \partial_{x_i}\!\Big(\tfrac{1}{\det J_e}\sum_k (J_e)_{ik}\hat v_k\Big)
= \tfrac{1}{\det J_e}\sum_{k,m}\Big(\underbrace{\textstyle\sum_i (J_e^{-1})_{mi}(J_e)_{ik}}_{=\;\delta_{mk}}\Big)\,\partial_{\xi_m}\hat v_k
= \frac{1}{\det J_e}\,\widehat{\operatorname{div}}\,\hat v .
$$ (eq-div)

$$
\boxed{\,\operatorname{div}_x\Big(\tfrac{1}{\det J_e}J_e\hat v\Big) = \frac{1}{\det J_e}\,\widehat{\operatorname{div}}\,\hat v\,}
$$

This is `test_contravariant_divergence_equals_reference_over_detJ`. (For a *non-affine* $J_e$ one extra
term appears; it vanishes by the **Piola identity** $\sum_i\partial_{x_i}\!\big(\tfrac{1}{\det J_e}(J_e)_{ik}\big)=0$,
so {eq}`eq-div` holds in general. The library's simplex mapping is affine, so the term is simply absent.)
Flux preservation {eq}`eq-flux-req` then also follows from the divergence theorem:
$\int_{\partial K}v\cdot n = \int_K\operatorname{div}_x v = \int_K \tfrac{1}{\det J_e}\widehat{\operatorname{div}}\hat v
= \int_{\hat K}\widehat{\operatorname{div}}\hat v = \int_{\partial\hat K}\hat v\cdot\hat n$, where the volume
measure change $\mathrm d x = \det J_e\,\mathrm d\xi$ cancels the $1/\det J_e$.

---

## 5. Where the map lives relative to the DOFs

A finite element is a pair: **functionals** $\{\ell_a\}$ (the DOFs — here circulations or fluxes) and a
**basis** $\{\hat H_a\}$ on $\hat K$, made dual so that on the physical cell

$$
\ell_a\big(\underbrace{A_e\,\hat H_b\circ\phi_e^{-1}}_{\text{physical basis } \Phi H_b}\big) = \delta_{ab}.
$$ (eq-duality)

The Piola factor $A_e$ sits on the **basis / reconstruction** side. Concretely it is applied *after*
the reference contraction, to the assembled value:

$$
v_h(x) = A_e\Big(\sum_a \hat q_a\,\hat H_a(\xi)\Big), \qquad \xi=\phi_e^{-1}(x),
$$ (eq-reconstruct)

which is exactly what {py:func}`~neurom.dof_transformations.reference_to_physical.reference_to_physical`
computes, wired into point evaluation and quadrature assembly.

This also explains an asymmetry documented in the code: **interpolation of a known function does *not*
apply $A_e$.** Interpolation *measures* a physical field, $q_a=\ell_a(f)$; because the basis is dual to
the functionals {eq}`eq-duality`, that measurement already yields the correct coefficient, with no $A_e$.
The push-forward only appears where a field is *reconstructed* from coefficients (assembly, point
evaluation) and where a functional is applied to the pushed-forward basis (the rows of $L_e$). See the
contrast between `interpolate` and `PointWiseInterpolator` in {doc}`dof_transformations`-adjacent code.

### Orientation (the sign $T_e$)

One subtlety remains for a *shared* edge: the two cells that meet there may traverse it in opposite
directions, so their locally built tangent/normal point oppositely. The relative sign
$\sigma_{e}\in\{+1,-1\}$ is stored in the topology (`entity_orientation`), and the DOF transformation
carries an extra diagonal factor $T_e=\operatorname{diag}(\sigma_e)$ so both cells agree on the globally
oriented flux/circulation. Because the contravariant map already uses the **signed** $\det J_e$
(§4.2), this sign is then *purely topological*. The full runtime DOF transformation is
$M_e = G_e\,T_e$ (with the constant unisolvence factor $C=V^{-1}$ folded into the tabulated basis), of
which the covariant/contravariant $A_e$ derived here is the geometric part $G_e$ acting on the value
components.

---

## 6. Summary

$$
\begin{array}{llll}
\textbf{map type} & \textbf{rule } v=A_e\hat v & \textbf{preserves} & \textbf{conforming space / elements}\\[2pt]
\texttt{IDENTITY} & v=\hat v & \text{point values} & H^1\ \text{(Lagrange, Hermite)}\\
\texttt{COVARIANT\_PIOLA} & v=J_e^{-\top}\hat v & \text{circulation }\textstyle\int v\cdot t & H(\mathrm{curl})\ \text{(Nédélec)}\\
\texttt{CONTRAVARIANT\_PIOLA} & v=\tfrac{1}{\det J_e}J_e\hat v & \text{flux }\textstyle\int v\cdot n & H(\mathrm{div})\ \text{(Raviart–Thomas)}
\end{array}
$$

The through-line of the derivation:

1. **Tangents push forward by $J_e$** {eq}`eq-tangent`; **gradients by $J_e^{-\top}$** {eq}`eq-grad-pull`;
   **area-normals by $\det(J_e)J_e^{-\top}$** (Nanson, {eq}`eq-nanson`; in 2-D the identity
   $J_e^{\top}RJ_e=\det(J_e)R$, {eq}`eq-JTRJ`).
2. Demand **circulation invariance** $\Rightarrow$ $A_e=J_e^{-\top}$ (covariant, {eq}`eq-covariant`).
3. Demand **flux invariance** $\Rightarrow$ $A_e=\tfrac1{\det J_e}J_e$ (contravariant, {eq}`eq-contravariant`),
   with the **signed** determinant (§4.2) and the divergence identity {eq}`eq-div`.
4. The map acts on the **reconstructed value** {eq}`eq-reconstruct`; interpolation needs no map because
   DOFs and basis are dual {eq}`eq-duality`; orientation adds a topological sign $T_e$.

### Code map

| Object | Symbol | Code |
| --- | --- | --- |
| map-type declaration | $A_e^{\text{id/cov/con}}$ | {py:class}`~neurom.elements.finite_element.MapType` |
| the push-forward $v=A_e\hat v$ | {eq}`eq-covariant`, {eq}`eq-contravariant` | {py:func}`~neurom.dof_transformations.reference_to_physical.reference_to_physical` |
| apply it with per-point $J_e,\det J_e$ | {eq}`eq-reconstruct` | {py:func}`~neurom.dof_transformations.reference_to_physical.reference_to_physical_at` |
| $J_e$, signed $\det J_e$ | {eq}`eq-affine` | {py:class}`~neurom.geometry.affine_simplex_mapping.AffineSimplexMapping` |
| DOF-side transform $M_e=G_e T_e$ | §5 | {py:class}`~neurom.dof_transformations.transformation.Transformation` |

### References

- F. Brezzi and M. Fortin, *Mixed and Hybrid Finite Element Methods*, Springer, 1991.
- M. Rognes, R. C. Kirby and A. Logg, *Efficient assembly of H(div) and H(curl) conforming finite
  elements*, SIAM J. Sci. Comput. 31(6), 2009.
- R. C. Kirby, *A general approach to transforming finite elements*, SMAI J. Comput. Math. 4, 2018.
- F. R. A. Aznaran, P. E. Farrell and R. C. Kirby, *Transformations for Piola-mapped elements*, 2022.
