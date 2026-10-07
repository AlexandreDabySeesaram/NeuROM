# Physical DOFs vs reference DOFs (Hermite elements)

## Problem

Until now, physical and reference DOFs were stored without distinction. This
did not matter because only Lagrange elements were used, i.e. affine-equivalent
elements.

- The DOFs were only values of the interpolated function.
- Hermite elements revealed the problem: their DOFs also include values of the
  derivative of the function.

---

## Physical DOFs

$$
q_e = \big[\, w(x_e)\,;\ w'(x_e)\,;\ w(x_{e+1})\,;\ w'(x_{e+1}) \,\big]
$$

$\rightsquigarrow$ physical DOFs: evaluated on the physical geometry.

$w(x)$ is the physical variable,

$$
w(x) = \hat{w}\big(\phi_e^{-1}(x)\big) \quad \text{where} \quad \hat{w}(\xi) = w\big(\phi_e(\xi)\big), \quad \xi \in [-1, 1]
$$

The whole question is whether the stored value/derivative DOFs are those of
$\boxed{\hat{w}(\xi) = w\big(\phi_e(\xi)\big)}$ or those of
$\boxed{w(x) = \hat{w}\big(\phi_e^{-1}(x)\big)}$.
Indeed, the interpolation on $[x_e, x_{e+1}]$ reads:

$$
w(x) = \Big[\hat{H}_1\big(\phi_e^{-1}(x)\big),\ \hat{H}_2\big(\phi_e^{-1}(x)\big),\ \hat{H}_3\big(\phi_e^{-1}(x)\big),\ \hat{H}_4\big(\phi_e^{-1}(x)\big)\Big]
\underbrace{\begin{bmatrix} \hat{w}_e(-1) \\ \hat{w}_e'(-1) \\ \hat{w}_e(1) \\ \hat{w}_e'(1) \end{bmatrix}}_{\hat{q}_e}
$$

$$
= \Big[\hat{H}_1\big(\phi_e^{-1}(x)\big),\ \dots,\ \hat{H}_4\big(\phi_e^{-1}(x)\big)\Big]
\,\underline{\underline{M}}_e\,
\underbrace{\begin{bmatrix} w(x_e) \\ w'(x_e) \\ w(x_{e+1}) \\ w'(x_{e+1}) \end{bmatrix}}_{=\, q_e}
$$

Because

$$
\hat{w}_e'(-1) = \frac{\partial \hat{w}}{\partial \xi}(-1) = J_e(-1)\cdot \frac{\partial w}{\partial x}(x_e) = J_e(-1)\cdot w'(x_e)
$$

i.e.

$$
w(x) = \underline{\hat{H}} \cdot \underline{\underline{M}}_e\, q_e = \underline{\hat{H}}\, \hat{q}_e
$$

with

$$
\underline{\underline{M}}_e = \operatorname{diag}\big(1,\ J_e(-1),\ 1,\ J_e(1)\big)
$$

---

## Without $\underline{\underline{M}}_e$

If $\underline{\underline{M}}_e$ is omitted and the physical DOFs are stored
(confusing them with the reference DOFs, $\hat{q}_e = q_e$):

$$
w(x) = \underline{\hat{H}}(x)\cdot q_e
$$

with, writing $l_e = x_{e+1} - x_e$,

$$
x = \phi_e(\xi) = \frac{x_e + x_{e+1}}{2} + \frac{l_e}{2}\,\xi
\quad \Rightarrow \quad
\phi_e^{-1}(x) = \frac{2}{l_e}\left(x - \frac{x_e + x_{e+1}}{2}\right)
$$

Then

$$
w'(x) = \frac{\partial}{\partial x}\underline{\hat{H}}(x)\cdot q_e
$$

But

$$
\frac{\partial}{\partial x}\underline{\hat{H}}(x) = \frac{\partial}{\partial x}\hat{H}_i\big(\phi_e^{-1}(x)\big)
= \frac{\partial \phi_e^{-1}}{\partial x}(x)\cdot \hat{H}_i'\big(\phi_e^{-1}(x)\big)
= \boxed{\frac{2}{l_e}}\ \hat{H}_i'\big(\phi_e^{-1}(x)\big)
$$

So, evaluating from elements $e$ and $e+1$:

$$
w'(x_{e+1}) = \frac{2}{l_e}\, \hat{w}_e'(1) = \frac{2}{l_{e+1}}\, \hat{w}_{e+1}'(-1) \leadsto \quad\text{absurd as soon as the mesh is not uniform}
$$

since the physical derivative must be continuous at the nodes; that is the
whole point of storing these additional DOFs.

Whereas with $\underline{\underline{M}}_e$:

$$
w'(x_{e+1}) = \underbrace{\frac{2}{l_e}\, J_e(1)}_{=\,1}\, \hat{w}_e'(1) \quad \text{OK!}
$$

**Conclusion: the physical DOFs must be stored, and correctly transported
according to how each kind of DOF transforms under the geometric mapping.**
