import numpy as np
import dlroms.fespaces as fe


# ============================================================
# Geometry
# ============================================================

domain = (
    fe.rectangle((0, 0), (3, 2))
    - fe.circle((0.95, 0.70), 0.25)
    - fe.circle((1.50, 1.35), 0.25)
    - fe.circle((2.05, 0.70), 0.25)
)

mesh = fe.mesh(domain, stepsize=0.05)


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
        result = False
        for out in outflows:
            result = result or out(x)
        return result

    def inflow(x):
        result = False
        for infl in inflows:
            result = result or infl[0](x)
        return result

    def walls(x):
        return not (outflow(x) or inflow(x))

    noslip = DirichletBC(
        W.sub(0),
        Constant((0.0, 0.0)),
        lambda x, on: on and walls(x)
    )

    def make_bc(i):
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
    e = np.zeros(4)
    e[i] = 1
    return stokes_solver(mesh, inflows(*e), outflows)


b0, b1, b2, b3 = [bi(i) for i in range(4)]


# ============================================================
# Function spaces for concentration problem
# ============================================================

Vh = fe.space(mesh, "CG", 1)
Vb = fe.space(mesh, "CG", 1, vector_valued=True)


# ============================================================
# Full Order Model
# ============================================================

def FOMsolver(mu):

    from fenics import grad, inner, dx
    from scipy.sparse.linalg import spsolve

    c1, c2, c3 = mu

    b = fe.asfunction(
        b0 + c1*b1 + c2*b2 + c3*b3,
        Vb
    )

    def a(u, v):
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

    from fenics import grad, inner, dx

    b_0 = fe.asfunction(b0, Vb)
    b_1 = fe.asfunction(b1, Vb)
    b_2 = fe.asfunction(b2, Vb)
    b_3 = fe.asfunction(b3, Vb)

    # Diffusion + fixed inlet flow
    def a0(u, v):
        return (
            0.005 * inner(grad(u), grad(v)) * dx
            + inner(b_0, grad(u)) * v * dx
        )

    # Contribution of cylinder 1
    def a1(u, v):
        return inner(b_1, grad(u)) * v * dx

    # Contribution of cylinder 2
    def a2(u, v):
        return inner(b_2, grad(u)) * v * dx

    # Contribution of cylinder 3
    def a3(u, v):
        return inner(b_3, grad(u)) * v * dx

    A0 = fe.assemble(a0, Vh)
    A1 = fe.assemble(a1, Vh)
    A2 = fe.assemble(a2, Vh)
    A3 = fe.assemble(a3, Vh)

    return A0, A1, A2, A3