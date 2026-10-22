# Physical Problems and Modeling Objectives

This document summarizes the two coursework applications implemented in the repository. It provides context for readers who do not have the original assignment handout. The notebooks contain the detailed derivations, experimental choices, and results.

## 1. Chemical dispersion in a three-cylinder mixer

### Engineering question

A chemical enters a two-dimensional mixing chamber containing three rotating cylinders. Changing the cylinder velocities changes the flow and the resulting concentration field. Repeated finite-element solves are costly when exploring operating conditions or minimizing concentration at a target point.

The goal is to construct a fast reduced model that predicts the stationary concentration field and supports optimization of the three cylinder velocities.

### Mathematical model

The stationary advection-diffusion equation is

$$
-\varepsilon\Delta u_\mu + \mathbf b_\mu\cdot\nabla u_\mu = 0,
\qquad \varepsilon=0.005.
$$

The parameter vector is $\mu=(c_1,c_2,c_3)\in[-5,5]^3$. The velocity field is assembled from precomputed Stokes solutions:

$$
\mathbf b_\mu=\mathbf b_0+c_1\mathbf b_1+c_2\mathbf b_2+c_3\mathbf b_3.
$$

The inlet concentration is $u=\exp[-25(y-1)^2]$; the implementation applies natural conditions on the remaining concentration boundaries. The domain is a rectangular chamber with three circular obstacles.

### Computational tasks

1. Generate training snapshots and independent test solutions with the FEM full-order model.
2. Lift the nonhomogeneous Dirichlet condition and compute a POD basis.
3. Project affine operators to build a POD-Galerkin model with an inexpensive online solve.
4. Assess projection error, complete ROM error, and computational speedup.
5. Minimize concentration at $x_0=(1.5,0.25)$ over cylinder velocities and verify the candidate with the full-order model.

Global field accuracy and pointwise objective accuracy are assessed separately: a small average field error does not guarantee an equally small optimization error.

Entry point: [01-FEM-Surrogate-Modelling/Results.ipynb](01-FEM-Surrogate-Modelling/Results.ipynb).

## 2. Damage propagation in a U-bolt

### Engineering question

A localized damaged region evolves over a U-shaped component. Diffusion spreads the damaged region while a reaction term increases local damage. The project asks whether a reduced surrogate can predict this evolution efficiently, infer unknown parameters from measurements, and forecast damage beyond the observed interval.

### Mathematical model

The damage field follows the Fisher-KPP equation

$$
\frac{\partial u}{\partial t}-D\Delta u=ru(1-u),
\qquad \nabla u\cdot\mathbf n=0\quad\text{on the boundary}.
$$

The normalized parameters $\mu\in[0,1]^3$ determine

$$
D=10^{\mu_1},\qquad r=10^{\mu_2+2},\qquad\delta=\mu_3+1.
$$

The initial damaged strip is described by $g_\delta$: it is one where $|x-\delta|<0.1$ and $y\geq4$, and zero elsewhere. The intended physical damage range is $0\leq u\leq1$.

The supplied dataset contains 100 trajectories, each with 41 snapshots over $[0,T]$, where $T=0.02$ and $\Delta t=5\times10^{-4}$. Each field has 6,636 FEM degrees of freedom. The forward POD-NN study uses 60 training, 15 validation, and 25 test trajectories.

### Tasks I-II: forward surrogate

Construct a POD basis using the FEM mass-matrix inner product, then train a neural network mapping physical parameters and normalized time to POD coefficients:

$$
(\mu,t/T)\longmapsto\widehat a(\mu,t)\longmapsto V_N\widehat a(\mu,t).
$$

Fit the basis and coefficient scaling on training data, choose hyperparameters using validation data, and evaluate the frozen model on the independent test set. Report spatial L2 errors, their temporal evolution, and inference costs.

Entry point: [02-Dynamical_Systems-ROM/I&II_Results.ipynb](02-Dynamical_Systems-ROM/I%26II_Results.ipynb).

### Task III: parameter identification

Use the trained forward surrogate to identify the parameters that best match the supplied pointwise damage measurements. Compare predicted observations with measurements and visualize the reconstructed damage field.

Entry point: [02-Dynamical_Systems-ROM/III_Results.ipynb](02-Dynamical_Systems-ROM/III_Results.ipynb).

### Task IV: temporal extrapolation

Identify reduced dynamics and evaluate how reliably they predict future damage. The final question concerns damage at $x^*=(3.5,2)$ and $t^*=2T=0.04$, using the parameters inferred in Task III.

Compare DMD, SINDy variants, and alternative reduced dynamics. Distinguish numerical integration success, physical admissibility, spatial error, and pointwise forecast error. Full-order reference calculations and temporal refinement diagnostics help assess the credibility of forecasts beyond the training interval.

Entry point: [02-Dynamical_Systems-ROM/IV_Results.ipynb](02-Dynamical_Systems-ROM/IV_Results.ipynb).

## Provenance of the original assignment

These are academic modeling exercises, not validated industrial deployment studies. The original assignment handout is not included here. If it can be redistributed, it can be added alongside this summary with its author, course, academic year, and source clearly identified. Otherwise, retain this summary and cite an authorized public source for the original assignment. Attribution of supplied code and data should distinguish those materials from the analysis and extensions developed in this repository.
