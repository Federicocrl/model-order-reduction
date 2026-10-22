"""FEM solvers and affine operators for the three-cylinder mixing problem.

Importing this module installs the process-local conversion compatibility
patch, loads or generates the mesh, solves four Stokes velocity basis
problems, and creates the concentration spaces. This initialization requires
the project's legacy FEniCS/dlroms environment."""
import numpy as np
import dlroms.fespaces as fe
import os
import shutil
from pathlib import Path
from fenics_numpy_compat import install, check
import matplotlib.pyplot as plt
# Fail before generating data if the runtime cannot preserve FEM coefficients.
install(fe)


# ============================================================
# Geometry
# ============================================================

domain = (
    fe.rectangle((0, 0), (3, 2))
    - fe.circle((0.95, 0.70), 0.25)
    - fe.circle((1.50, 1.35), 0.25)
    - fe.circle((2.05, 0.70), 0.25)
)

def load_mixer_mesh():
    """Reuse the dataset mesh; locate Gmsh even in a VS Code Python kernel."""
    from fenics import Mesh, File

    mesh_path = Path(__file__).with_name("mixer_mesh.xml")
    if mesh_path.exists():
        return Mesh(str(mesh_path))

    gmsh = shutil.which("gmsh")
    if gmsh is None:
        # dlroms is reused from the shared virtual environment via a .pth file.
        candidate = Path(fe.__file__).resolve().parents[4] / "bin" / "gmsh"
        if candidate.is_file():
            gmsh = str(candidate)
    if gmsh is None:
        raise RuntimeError("Gmsh was not found: cannot generate the mixer mesh.")
    if shutil.which("dolfin-convert") is None:
        raise RuntimeError("dolfin-convert was not found: cannot convert the mixer mesh.")

    previous_path = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = str(Path(gmsh).parent) + os.pathsep + previous_path
        result = fe.mesh(domain, stepsize=0.05)
    finally:
        os.environ["PATH"] = previous_path
    File(str(mesh_path)) << result
    return result


mesh = load_mixer_mesh()


# ============================================================
# Boundaries
# ============================================================

left = lambda x: x[0] < 1e-6
right = lambda x: 3 - x[0] < 1e-12

circle = lambda x0, r: (
    lambda x:
    ((x[0] - x0[0])**2 + (x[1] - x0[1])**2)**0.5 < r + 1e-3
)


def inflows(b0, b1, b2, b3):
    """Return boundary predicates and velocity functions for the inlet and three cylinders.

    b0 scales the Gaussian inlet profile; b1, b2, and b3 set cylinder angular
    velocities. Each returned entry is [boundary_predicate, velocity_function]."""
    return [
        [
            left,
            lambda x: [b0*np.exp(-25*(x[1]-1)**2), 0.0]
        ],
        [
            circle((0.95, 0.70), 0.25),
            lambda x: [b1*(-x[1]+0.70), b1*(x[0]-0.95)]
        ],
        [
            circle((1.50, 1.35), 0.25),
            lambda x: [b2*(-x[1]+1.35), b2*(x[0]-1.50)]
        ],
        [
            circle((2.05, 0.70), 0.25),
            lambda x: [b3*(-x[1]+0.70), b3*(x[0]-2.05)]
        ]
    ]


outflows = [right]


# ============================================================
# Stokes solver
# ============================================================

def stokes_solver(mesh, inflows, outflows, source=[0.0, 0.0]):

    """Solve the steady incompressible Stokes problem on the supplied mesh.

    inflows contains [predicate, velocity] entries, outflows contains boundary
    predicates, and source is the two-component body force. Impose prescribed
    inflow/cylinder velocities, no-slip walls, and natural outlet conditions.
    Return the flattened velocity in the project's vector CG1 DOF ordering;
    the mixed pressure solution is not returned."""
    from fenics import (
        FiniteElement, NodalEnrichedElement, FunctionSpace,
        VectorElement, TrialFunctions, TestFunctions,
        inner, grad, dx, div, assemble,
        DirichletBC, Constant
    )

    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix

    pP1 = FiniteElement("CG", mesh.ufl_cell(), 1)

    vP1B = VectorElement(
        NodalEnrichedElement(
            FiniteElement("CG", mesh.ufl_cell(), 1),
            FiniteElement(
                "Bubble",
                mesh.ufl_cell(),
                mesh.topology().dim() + 1
            )
        )
    )

    pspace, vspace = pP1, vP1B

    W = FunctionSpace(mesh, vspace * pspace)
    W0 = FunctionSpace(mesh, vspace)

    (b, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    space = fe.space(
        mesh,
        "CG",
        1,
        vector_valued=True,
        bubble=True
    )

    f = fe.interpolate(source, space)

    a = (
        inner(grad(b), grad(v))*dx
        - div(v)*p*dx
        - q*div(b)*dx
    )

    L = inner(f, v)*dx

    def outflow(x):
        """Return whether a point belongs to any configured outlet boundary."""
        result = False
        for out in outflows:
            result = result or out(x)
        return result

    def inflow(x):
        """Return whether a point belongs to any prescribed velocity boundary."""
        result = False
        for infl in inflows:
            result = result or infl[0](x)
        return result

    def walls(x):
        """Return whether a point lies outside the configured inlet and outlet predicates."""
        return not (outflow(x) or inflow(x))

    noslip = DirichletBC(
        W.sub(0),
        Constant((0.0, 0.0)),
        lambda x, on: on and walls(x)
    )

    def make_bc(i):
        """Build the mixed-space velocity Dirichlet condition for inflow entry i."""
        return DirichletBC(
            W.sub(0),
            fe.interpolate(inflows[i][1], W0),
            lambda x, on: on and inflows[i][0](x)
        )

    ins = [make_bc(i) for i in range(len(inflows))]

    A = assemble(a)
    F = assemble(L)

    for bc in [noslip, *ins]:
        bc.apply(A)
        bc.apply(F)

    A = csr_matrix(A.array())
    F = F[:]

    bp = spsolve(A, F)
    bp_f = fe.asfunction(bp, W)

    from fenics import dof_to_vertex_map

    Vh_temp = fe.space(mesh, "CG", 1)

    nvertices = mesh.coordinates().shape[0]

    b = (
        bp_f.compute_vertex_values(mesh)[:2*nvertices]
        .reshape(2, -1)
        .T
    )

    indexes = dof_to_vertex_map(Vh_temp)

    b = b[indexes].reshape(-1)

    return b


# ============================================================
# Stokes pre-computation
# ============================================================

def bi(i):
    """Return the Stokes velocity basis for unit boundary coefficient i (0=inlet, 1..3=cylinders)."""
    e = np.zeros(4)
    e[i] = 1
    return stokes_solver(mesh, inflows(*e), outflows)


b0, b1, b2, b3 = [bi(i) for i in range(4)]


# ============================================================
# Function spaces for concentration problem
# ============================================================

Vh = fe.space(mesh, "CG", 1)
Vb = fe.space(mesh, "CG", 1, vector_valued=True)
check(fe, Vh)
check(fe, Vb)


# ============================================================
# Full Order Model
# ============================================================

def FOMsolver(mu):

    """Solve stationary advection-diffusion for cylinder velocities mu=(c1,c2,c3).

    Use the global mesh, scalar CG1 space, diffusion coefficient 0.005, and
    velocity b0+c1*b1+c2*b2+c3*b3. Impose the Gaussian concentration at the
    left inlet and natural conditions elsewhere. Return a concentration DOF vector."""
    from fenics import grad, inner, dx
    from scipy.sparse.linalg import spsolve

    c1, c2, c3 = mu

    b = fe.asfunction(
        b0 + c1*b1 + c2*b2 + c3*b3,
        Vb
    )

    def a(u, v):
        """Return the diffusion-advection bilinear form for the current velocity field."""
        return (
            0.005*inner(grad(u), grad(v))*dx
            + inner(b, grad(u))*v*dx
        )

    A = fe.assemble(a, Vh)

    F = np.zeros(Vh.dim())

    bc = fe.DirichletBC(
        left,
        lambda x: np.exp(-25*(x[1]-1)**2)
    )

    A = fe.applyBCs(A, Vh, bc)
    F = fe.applyBCs(F, Vh, bc)

    u = spsolve(A, F)

    return u



def assemble_affine_operators():

    """Return unconditioned full-order matrices (A0,A1,A2,A3).

    A0 contains diffusion and fixed inlet advection; A1..A3 contain the cylinder
    advection contributions. A(mu)=A0+sum(mu[i]*A[i+1]). Boundary treatment,
    Dirichlet lifting, and reduced projection are performed by the caller."""
    from fenics import grad, inner, dx

    b_0 = fe.asfunction(b0, Vb)
    b_1 = fe.asfunction(b1, Vb)
    b_2 = fe.asfunction(b2, Vb)
    b_3 = fe.asfunction(b3, Vb)

    # Diffusion + fixed inlet flow
    def a0(u, v):
        """Return the diffusion and fixed-inlet advection bilinear form."""
        return (
            0.005 * inner(grad(u), grad(v)) * dx
            + inner(b_0, grad(u)) * v * dx
        )

    # Contribution of cylinder 1
    def a1(u, v):
        """Return the advection bilinear form induced by cylinder one."""
        return inner(b_1, grad(u)) * v * dx

    # Contribution of cylinder 2
    def a2(u, v):
        """Return the advection bilinear form induced by cylinder two."""
        return inner(b_2, grad(u)) * v * dx

    # Contribution of cylinder 3
    def a3(u, v):
        """Return the advection bilinear form induced by cylinder three."""
        return inner(b_3, grad(u)) * v * dx

    A0 = fe.assemble(a0, Vh)
    A1 = fe.assemble(a1, Vh)
    A2 = fe.assemble(a2, Vh)
    A3 = fe.assemble(a3, Vh)

    return A0, A1, A2, A3


import math

def plot_pod_modes(U_pod, Vh, modes_to_plot, ncols=3):
    """Plot selected one-based POD mode indices in a grid and call plt.show().

    U_pod stores spatial modes in columns, Vh is the scalar FEM space, and
    ncols sets the grid width. Unused axes are hidden; no value is returned."""
    nrows = math.ceil(len(modes_to_plot) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, mode in zip(axes, modes_to_plot):
        plt.sca(ax)
        fe.plot(U_pod[:, mode - 1], Vh, colorbar=True)
        ax.set_title(f"POD mode {mode}")

    for ax in axes[len(modes_to_plot):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()