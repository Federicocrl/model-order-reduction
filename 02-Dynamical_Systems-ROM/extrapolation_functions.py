"""Temporal ROMs and FEM diagnostics for U-bolt damage propagation.

Includes DMD, SINDy, Galerkin, hybrid, and POD-LSTM utilities. No simulations
or training run at import. Historical DMDModel, fit_dmd, and fit_sindy
definitions near the top are superseded by later definitions in this file;
their older backtest helpers require API cleanup before reuse."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DMDModel:
    """Store the historical affine DMD operator, offset, singular values, and fitted rank."""
    operator: np.ndarray
    offset: np.ndarray
    singular_values: np.ndarray
    rank: int

    @property
    def eigenvalues(self):
        """Return the eigenvalues of the stored full reduced-state operator."""
        return np.linalg.eigvals(self.operator)


def fit_dmd(coefficients, rank, *, window=None, shift='zero', equilibrium=None,
            ridge=0.0, rtol=1e-10, formulation='exact'):
    """Fit a_{k+1}=c+K(a_k-c) by truncated SVD of row-wise snapshots.

    Historical API: superseded by the later fit_dmd definition in this module.
    shift selects zero, the fitting-window mean, or a supplied equilibrium.
    ridge is dimensionless, with alpha=ridge*sigma_max**2. The requested rank
    must be numerically available. formulation='exact' uses Y V Sigma^-1 U.T;
    'projected' projects this operator onto the retained left-singular subspace.
    No eigenvalue or field clipping is applied."""
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 2 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError('Servono almeno tre snapshot finiti, shape (Nt,N).')
    if int(rank) != rank or rank < 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError('Rango intero positivo e ridge finito non negativo richiesti.')
    if window is not None:
        if int(window) != window or window < 3:
            raise ValueError('Finestra intera di almeno tre snapshot richiesta.')
        a = a[-int(window):]
    if shift == 'zero':
        offset = np.zeros(a.shape[1])
    elif shift == 'mean':
        offset = a.mean(axis=0)
    elif shift == 'equilibrium':
        offset = np.asarray(equilibrium, dtype=float)
        if offset.shape != (a.shape[1],) or not np.isfinite(offset).all():
            raise ValueError('Equilibrio incompatibile con i coefficienti.')
    else:
        raise ValueError('shift deve essere zero, mean o equilibrium.')
    X, Y = (a[:-1]-offset).T, (a[1:]-offset).T
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    available = int(np.sum(sigma > rtol*sigma[0])) if sigma[0] > 0 else 0
    if rank > available:
        raise ValueError(f'Rango {rank} superiore al rango numerico {available}.')
    r = int(rank)
    inverse = sigma[:r] / (sigma[:r]**2 + ridge*sigma[0]**2)
    K = ((Y @ Vt[:r].T)*inverse) @ U[:, :r].T
    if formulation == 'projected':
        K = U[:, :r] @ (U[:, :r].T @ K)
    elif formulation != 'exact':
        raise ValueError('formulation deve essere exact o projected.')
    return DMDModel(K, offset.copy(), sigma, r)


def forecast_dmd(model, initial, steps):
    """Return the initial state and steps affine DMD updates as (steps+1,n_modes).

    Requires the historical model with an offset attribute, not the later
    projected DMDModel. Raise FloatingPointError for nonfinite predictions."""
    if int(steps) != steps or steps < 0:
        raise ValueError('steps intero non negativo richiesto.')
    initial = np.asarray(initial, dtype=float)
    if initial.shape != model.offset.shape or not np.isfinite(initial).all():
        raise ValueError('Stato iniziale non valido.')
    states = np.empty((int(steps)+1, initial.size))
    states[0] = initial
    for k in range(int(steps)):
        states[k+1] = model.offset + model.operator @ (states[k]-model.offset)
    if not np.isfinite(states).all():
        raise FloatingPointError('La previsione DMD diverge numericamente.')
    return states


def backtest_dmd(coefficients, configurations, cutoffs, gram, sensor_modes,
                 *, equilibrium=None):
    """Evaluate historical affine DMD configurations with rolling-origin folds.

    Requires the historical fit_dmd API, superseded later in this module.
    cutoffs are inclusive fitting indices. Return (rows, folds) with mean relative
    spatial errors in the POD Gram norm and sensor RMSE. Future fold data are
    excluded from fitting and offsets. Infeasible configurations receive an
    infinite score rather than a silently reduced rank."""
    a = np.asarray(coefficients, dtype=float)
    G = np.asarray(gram, dtype=float)
    h = np.asarray(sensor_modes, dtype=float)
    rows, folds = [], []
    for config_id, config in enumerate(configurations):
        errors, point_errors = [], []
        status = 'ok'
        for cutoff in cutoffs:
            if cutoff < 2 or cutoff >= len(a)-1:
                raise ValueError('Cutoff non valido.')
            try:
                model = fit_dmd(a[:cutoff+1], equilibrium=equilibrium, **config)
                pred = forecast_dmd(model, a[cutoff], len(a)-1-cutoff)[1:]
                truth = a[cutoff+1:]
                diff = pred-truth
                denom = np.sqrt(np.maximum(np.einsum('ti,ij,tj->t', truth,G,truth),0))
                if np.any(denom <= 0):
                    raise ValueError('Norma nulla nel backtest.')
                error = float(np.mean(np.sqrt(np.maximum(
                    np.einsum('ti,ij,tj->t', diff,G,diff),0))/denom))
                point_error = float(np.sqrt(np.mean((diff@h)**2)))
                errors.append(error)
                point_errors.append(point_error)
                folds.append(dict(config_id=config_id, cutoff=cutoff,
                                  relative_l2=error, sensor_rmse=point_error))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                status = str(exc)
                break
        rows.append(dict(config_id=config_id, **config, status=status,
                         score=float(np.mean(errors)) if status=='ok' else np.inf,
                         worst_fold=max(errors) if status=='ok' else np.inf,
                         sensor_rmse=float(np.mean(point_errors)) if status=='ok' else np.inf))
    return rows, folds


@dataclass
class SINDyModel:
    """Store PCA transforms and polynomial dynamics dz/dtau=Theta(z) @ coefficients.

    tau=t/T. PCA and scales are fitted on training data only. discrete_dt=None
    denotes continuous dynamics; a numeric value denotes a discrete update map."""
    center: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    feature_scale: np.ndarray
    regression_rank: int
    derivative_error: float
    discrete_dt: float | None = None

    def encode(self, a):
        """Map POD coefficients (...,n_modes) to centered, scaled local PCA coordinates."""
        return ((np.asarray(a)-self.center) @ self.basis)/self.scale

    def decode(self, z):
        """Map local PCA coordinates back to the retained POD coefficient subspace."""
        return self.center + (np.asarray(z)*self.scale) @ self.basis.T

    def rhs(self, tau, z):
        """Evaluate the polynomial derivative or discrete update; the autonomous map ignores tau."""
        return polynomial_library(np.asarray(z).reshape(1, -1), self.powers)[0] @ self.coefficients


def polynomial_powers(dimension, degree):
    """Return unique monomial exponent rows up to total degree, including the constant term."""
    from itertools import combinations_with_replacement
    powers = [np.zeros(dimension, dtype=int)]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers.append(np.bincount(indices, minlength=dimension))
    return np.stack(powers)


def polynomial_library(states, powers):
    """Evaluate Theta[i,j] = product_k states[i,k]**powers[j,k] for row-wise states."""
    states = np.atleast_2d(states)
    return np.prod(states[:, None, :]**powers[None, :, :], axis=2)


def fit_sindy(coefficients, tau, dimension, *, degree=2, threshold=0.05,
              alpha=0.01, derivative='finite', derivatives=None, window=None,
              max_iter=20):
    """Fit historical local-PCA SINDy dynamics by thresholded ridge regression.

    Historical API: superseded by the later fit_sindy definition in this module.
    coefficients contains row-wise POD states and tau contains uniform normalized
    times. derivative='finite' uses centered differences within the fitting data;
    'autograd' requires derivatives da/dtau, not da/dt; 'discrete' fits the next
    state directly. Optional window restricts fitting to trailing snapshots.
    Normalize library columns by RMS before thresholding and multiply ridge alpha
    by the sample count. Fit all transforms on the supplied data only. Return
    SINDyModel with regression rank and relative derivative/update residual."""
    a = np.asarray(coefficients, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if a.ndim != 2 or len(a)<5 or tau.shape!=(len(a),) or not np.isfinite(a).all():
        raise ValueError('Snapshot o tempi non validi.')
    if np.any(np.diff(tau)<=0) or not np.allclose(np.diff(tau),np.diff(tau)[0]):
        raise ValueError('Tempi uniformi crescenti richiesti.')
    if degree not in (1,2,3) or alpha<0 or threshold<0:
        raise ValueError('Grado o regolarizzazione non validi.')
    da = None if derivatives is None else np.asarray(derivatives, dtype=float)
    if window is not None:
        if window<5:
            raise ValueError('Finestra troppo corta.')
        a,tau = a[-window:],tau[-window:]
        if da is not None:
            da=da[-window:]
    center = a.mean(axis=0)
    _,sigma,Vt = np.linalg.svd(a-center,full_matrices=False)
    if dimension<1 or dimension>sum(sigma>1e-10*sigma[0]):
        raise ValueError('Dimensione PCA superiore al rango numerico.')
    basis = Vt[:dimension].T
    reduced = (a-center) @ basis
    scale = np.maximum(np.std(reduced,axis=0),1e-12)
    z = reduced/scale
    if derivative=='finite':
        zdot = (z[2:]-z[:-2])/(tau[2:]-tau[:-2])[:,None]
    elif derivative=='discrete':
        zdot=z[1:]
    elif derivative=='autograd':
        if da is None or da.shape!=a.shape or not np.isfinite(da).all():
            raise ValueError('Derivate autograd mancanti o incompatibili.')
        zdot = (da[1:-1] @ basis)/scale
    else:
        raise ValueError('Derivata finite, autograd o discrete richiesta.')
    powers = polynomial_powers(dimension,degree)
    theta = polynomial_library(z[:-1] if derivative=='discrete' else z[1:-1],powers)
    feature_scale = np.maximum(np.sqrt(np.mean(theta**2,axis=0)),1e-12)
    design = theta/feature_scale
    active = np.ones((design.shape[1],dimension),dtype=bool)
    normalized = np.zeros_like(active,dtype=float)
    for _ in range(max_iter):
        previous=active.copy()
        for target in range(dimension):
            mask = active[:,target]
            normalized[:,target]=0
            if mask.any():
                D=design[:,mask]
                # Risoluzione aumentata stabile anche con librerie sottodeterminate.
                lhs=np.vstack([D, np.sqrt(alpha*len(D))*np.eye(D.shape[1])])
                rhs=np.concatenate([zdot[:,target],np.zeros(D.shape[1])])
                normalized[mask,target]=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
        active &= abs(normalized)>=threshold
        if np.array_equal(active,previous):
            break
    normalized[~active]=0
    Xi=normalized/feature_scale[:,None]
    error=float(np.linalg.norm(theta@Xi-zdot)/max(np.linalg.norm(zdot),1e-15))
    return SINDyModel(center,basis,scale,powers,Xi,feature_scale,
                      int(np.linalg.matrix_rank(design)),error,
                      float(tau[1]-tau[0]) if derivative=='discrete' else None)


def forecast_sindy(model, initial, tau_elapsed, *, rtol=1e-7, atol=1e-9,
                   max_normalized_state=100., max_step=0.025):
    """Forecast a historical SINDyModel at elapsed normalized times starting at zero.

    Integrate continuous dynamics or iterate the discrete map. Preserve the
    initial residual orthogonal to the local PCA subspace as a constant so that
    the first prediction equals initial exactly. Abort divergence at the
    max_normalized_state limit (default 100) without clipping. Return POD states
    with shape (n_times,n_modes)."""
    from scipy.integrate import solve_ivp
    times=np.asarray(tau_elapsed,dtype=float)
    if times.ndim!=1 or len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Tempi trascorsi crescenti da zero richiesti.')
    z0=model.encode(initial)
    residual=np.asarray(initial)-model.decode(z0)
    if model.discrete_dt is not None:
        indices=np.rint(times/model.discrete_dt).astype(int)
        if not np.allclose(indices*model.discrete_dt,times,atol=1e-10):
            raise ValueError('Tempi incompatibili con il passo SINDy discreto.')
        z=z0.copy();states=[z.copy()]
        for _ in range(indices[-1]):
            z=model.rhs(0,z)
            if not np.isfinite(z).all() or np.max(abs(z))>max_normalized_state:
                raise FloatingPointError('SINDy discreto diverge numericamente.')
            states.append(z.copy())
        prediction=model.decode(np.stack(states)[indices])+residual
        prediction[0]=initial
        return prediction
    def stop(t,z):
        """Trigger a terminal integration event at the maximum normalized-state magnitude."""
        return max_normalized_state-np.max(np.abs(z))
    stop.terminal=True
    stop.direction=-1
    if np.max(abs(z0))>=max_normalized_state:
        raise FloatingPointError('Stato iniziale SINDy fuori dal limite numerico.')
    with np.errstate(over='ignore',invalid='ignore'):
        sol=solve_ivp(model.rhs,(0.,times[-1]),z0,t_eval=times,
                      rtol=rtol,atol=atol,max_step=max_step,events=stop)
    if not sol.success or len(sol.t)!=len(times) or not np.isfinite(sol.y).all():
        raise FloatingPointError('SINDy diverge o non completa l’orizzonte richiesto.')
    prediction=model.decode(sol.y.T)+residual
    prediction[0]=initial
    return prediction


def coefficient_errors(prediction, truth, gram, sensor_modes):
    """Return mean relative spatial Gram-norm error and pointwise sensor RMSE.

    prediction and truth contain row-wise POD coefficients; gram defines the
    spatial norm and sensor_modes maps coefficients to a scalar observation.
    Zero reference norms raise ValueError. No clipping is applied."""
    diff=np.asarray(prediction)-np.asarray(truth)
    denominator=np.sqrt(np.maximum(np.einsum('ti,ij,tj->t',truth,gram,truth),0))
    if np.any(denominator<=0):
        raise ValueError('Norma nulla nel riferimento.')
    errors=np.sqrt(np.maximum(np.einsum('ti,ij,tj->t',diff,gram,diff),0))/denominator
    return float(errors.mean()),float(np.sqrt(np.mean((diff@sensor_modes)**2)))


def backtest_sindy(coefficients, tau, configurations, cutoffs, gram,
                   sensor_modes, *, derivatives=None, evaluation_stride=1):
    """Evaluate historical local-PCA SINDy using rolling-origin fitting prefixes.

    Requires the historical fit_sindy API, superseded later in this module.
    Return (rows, folds) using the same spatial and sensor metrics as
    backtest_dmd. Fit PCA, scales, and sparsity on each prefix only;
    evaluation_stride subsamples forecast times for metric evaluation."""
    rows,folds=[],[]
    for config_id,config in enumerate(configurations):
        errors,point_errors=[],[]
        status='ok'
        for cutoff in cutoffs:
            try:
                fitted=fit_sindy(coefficients[:cutoff+1],tau[:cutoff+1],
                    derivatives=None if derivatives is None else derivatives[:cutoff+1],**config)
                pred=forecast_sindy(fitted,coefficients[cutoff],tau[cutoff:]-tau[cutoff])[1:]
                error,point_error=coefficient_errors(pred[evaluation_stride-1::evaluation_stride],
                    coefficients[cutoff+1:][evaluation_stride-1::evaluation_stride],gram,sensor_modes)
                errors.append(error); point_errors.append(point_error)
                folds.append(dict(config_id=config_id,cutoff=cutoff,
                    relative_l2=error,sensor_rmse=point_error,
                    active_terms=int(np.count_nonzero(fitted.coefficients)),
                    derivative_error=fitted.derivative_error))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                status=str(exc); break
        rows.append(dict(config_id=config_id,**config,status=status,
            score=float(np.mean(errors)) if status=='ok' else np.inf,
            worst_fold=max(errors) if status=='ok' else np.inf,
            sensor_rmse=float(np.mean(point_errors)) if status=='ok' else np.inf))
    return rows,folds


def solve_fkpp_implicit(Vh, mu, times, *, substeps=1):
    """Solve Fisher-KPP with nonlinear backward Euler and scalar CG1 FEM.

    Use natural homogeneous Neumann boundaries and nodal initial damage g_delta.
    mu contains three normalized parameters: D=10**mu[0], r=10**(mu[1]+2), and
    delta=mu[2]+1. times must be uniformly increasing; substeps refines each
    interval. Solve the nonlinear reaction-diffusion residual with Newton.
    Return (fields, elapsed_seconds), including the initial field. Intended for
    reference diagnostics rather than training a reduced model."""
    import dolfin as df
    import time
    mu=np.asarray(mu,dtype=float);times=np.asarray(times,dtype=float)
    if mu.shape!=(3,) or not np.isfinite(mu).all() or np.any((mu<0)|(mu>1)):
        raise ValueError('Parametri normalizzati non validi.')
    if len(times)<2 or np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),np.diff(times)[0]):
        raise ValueError('Griglia uniforme crescente richiesta.')
    if int(substeps)!=substeps or substeps<1:
        raise ValueError('substeps intero positivo richiesto.')
    D,r,delta=10**mu[0],10**(mu[1]+2),mu[2]+1
    dt=(times[1]-times[0])/substeps
    previous,current=df.Function(Vh),df.Function(Vh)
    coords=Vh.tabulate_dof_coordinates().reshape(-1,2)
    initial=((abs(coords[:,0]-delta)<.1)&(coords[:,1]>=4)).astype(float)
    previous.vector().set_local(initial);previous.vector().apply('insert')
    current.assign(previous)
    v=df.TestFunction(Vh)
    residual=((current-previous)*v+dt*D*df.inner(df.grad(current),df.grad(v))
              -dt*r*current*(1-current)*v)*df.dx
    jacobian=df.derivative(residual,current)
    problem=df.NonlinearVariationalProblem(residual,current,J=jacobian)
    solver=df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance']=1e-9
    solver.parameters['newton_solver']['absolute_tolerance']=1e-10
    solver.parameters['newton_solver']['maximum_iterations']=30
    solver.parameters['newton_solver']['error_on_nonconvergence']=True
    fields=[initial.copy()];start=time.perf_counter()
    for _ in times[1:]:
        for __ in range(int(substeps)):
            solver.solve();previous.assign(current)
        fields.append(current.vector().get_local().copy())
    return np.stack(fields),time.perf_counter()-start




import numpy as np
from dataclasses import dataclass


@dataclass
class DMDModel:
    """
    Store a truncated DMD model.

    Attributes
    ----------
    basis : ndarray, shape (n_modes, rank)
        Left singular vectors retained from the snapshot matrix X.
        They define the subspace in which DMD dynamics are propagated.

    operator : ndarray, shape (rank, rank)
        Reduced linear evolution operator A_tilde.

    singular_values : ndarray
        Singular values of the snapshot matrix X.

    spectral_radius : float
        Maximum absolute eigenvalue of the reduced operator.
        Values above 1 indicate an unstable discrete-time mode.
    """
    basis: np.ndarray
    operator: np.ndarray
    singular_values: np.ndarray

    @property
    def spectral_radius(self):
        """Return the largest absolute eigenvalue of the projected DMD operator."""
        return float(np.max(np.abs(np.linalg.eigvals(self.operator))))


def fit_dmd(a_fit, rank, start_idx=0):
    """
    Identify a truncated DMD operator from a single POD trajectory.

    The input trajectory contains consecutive reduced snapshots stored
    as rows. DMD constructs two time-shifted matrices:

        X = [a_start, ..., a_(m-1)]
        Y = [a_(start+1), ..., a_m]

    It then computes the truncated SVD of X:

        X ≈ U_r @ Sigma_r @ V_r.T

    and identifies the reduced least-squares evolution operator:

        A_tilde = U_r.T @ Y @ V_r @ Sigma_r^(-1)

    The model is specific to the supplied trajectory. No parameter
    dependence is learned across different trajectories.

    Parameters
    ----------
    a_fit : ndarray, shape (n_times, n_modes)
        POD coefficients in the identification interval.

    rank : int
        Number of singular vectors retained in the DMD model.

    start_idx : int, default 0
        Index of the first snapshot used for identification.
        Increasing this value excludes part of the initial transient.

    Returns
    -------
    DMDModel
        Identified reduced basis, evolution operator and singular values.
    """
    a = np.asarray(a_fit, dtype=float)

    if a.ndim != 2:
        raise ValueError("a_fit must have shape (n_times, n_modes).")

    if not np.all(np.isfinite(a)):
        raise ValueError("a_fit contains non-finite values.")

    if not 0 <= start_idx < len(a) - 1:
        raise ValueError("Invalid start_idx: at least two snapshots are required.")

    X = a[start_idx:-1].T
    Y = a[start_idx + 1:].T

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if not 1 <= rank <= len(s):
        raise ValueError(f"rank must be between 1 and {len(s)}.")

    tol = np.finfo(float).eps * max(X.shape) * s[0]

    if s[rank - 1] <= tol:
        raise ValueError(
            f"Requested rank {rank} exceeds the numerical rank of X."
        )

    Ur = U[:, :rank]
    Vr = Vt[:rank].T

    A_tilde = (Ur.T @ Y @ Vr) / s[:rank][None, :]

    return DMDModel(
        basis=Ur,
        operator=A_tilde,
        singular_values=s
    )


def predict_dmd(model, a0, n_steps):
    """
    Forecast a POD trajectory using an identified DMD model.

    The initial POD state is first projected onto the DMD subspace:

        z0 = U_r.T @ a0

    The reduced state is then propagated recursively:

        z_(k+1) = A_tilde @ z_k

    and reconstructed in the original POD coordinate space:

        a_hat_k = U_r @ z_k

    The returned array contains n_steps future states and does NOT
    include the initial state a0.

    Parameters
    ----------
    model : DMDModel
        Previously identified DMD model.

    a0 : ndarray, shape (n_modes,)
        Last observed POD coefficient vector.

    n_steps : int
        Number of future time steps to predict.

    Returns
    -------
    ndarray, shape (n_steps, n_modes)
        Predicted POD coefficients at future time steps.
    """
    a0 = np.asarray(a0, dtype=float)

    if a0.shape != (model.basis.shape[0],):
        raise ValueError("a0 has an incompatible shape.")

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    z = model.basis.T @ a0
    predictions = np.empty((n_steps, model.basis.shape[0]))

    for k in range(n_steps):
        z = model.operator @ z
        predictions[k] = model.basis @ z

    return predictions


def dmd_forecast(a, rank, split_idx, start_idx=0):
    """
    Fit DMD on the observed interval and forecast the remaining trajectory.

    The full coefficient trajectory is divided into:

        identification: a[0 : split_idx + 1]
        extrapolation:  a[split_idx + 1 :]

    Only the identification interval is used to fit the operator.
    The last observed state initializes the autonomous forecast.

    Parameters
    ----------
    a : ndarray, shape (n_times, n_modes)
        Complete POD coefficient trajectory, including the reference
        future used only for evaluation.

    rank : int
        Truncation rank of the DMD operator.

    split_idx : int
        Index of the last observed snapshot.

    start_idx : int, default 0
        First snapshot used during operator identification.

    Returns
    -------
    model : DMDModel
        Identified DMD model.

    prediction : ndarray
        Forecasted POD coefficients for all times after split_idx.
    """
    a = np.asarray(a, dtype=float)

    if a.ndim != 2 or not 1 <= split_idx < len(a) - 1:
        raise ValueError("Invalid trajectory or temporal split.")

    a_fit = a[:split_idx + 1]

    model = fit_dmd(
        a_fit,
        rank=rank,
        start_idx=start_idx
    )

    prediction = predict_dmd(
        model,
        a0=a[split_idx],
        n_steps=len(a) - split_idx - 1
    )

    return model, prediction


def relative_coefficient_errors(reference, prediction, eps=1e-12):
    """
    Compute the relative Euclidean error at each predicted time step.

    For each time instant:

        error_k = ||a_ref_k - a_pred_k||_2 / ||a_ref_k||_2

    Because the POD basis is M-orthonormal, this is equivalent to the
    relative FEM L2 error between the corresponding POD-reconstructed
    fields. It does not include the POD projection error relative to
    the original full-order solution.

    Parameters
    ----------
    reference, prediction : ndarray, shape (n_times, n_modes)
        Reference and predicted POD coefficient trajectories.

    eps : float
        Small threshold used to avoid division by zero.

    Returns
    -------
    ndarray, shape (n_times,)
        Relative errors for each predicted time step.
    """
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)

    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Reference and prediction must have equal 2D shapes.")

    numerator = np.linalg.norm(reference - prediction, axis=1)
    denominator = np.linalg.norm(reference, axis=1)

    return numerator / np.maximum(denominator, eps)



from itertools import combinations_with_replacement
from scipy.integrate import solve_ivp


def build_sindy_library(Z_train, mu_train, dt):
    """
    Build the physics-informed SINDy regression problem.

    Input:
        Z_train: (n_trajectories, n_times, n_modes)
        mu_train: (n_trajectories, 3)
        dt: temporal sampling interval

    Method:
        1. Estimate POD coefficient derivatives using second-order
           finite differences.
        2. Construct diffusion-linear features D*z_j.
        3. Construct reaction-quadratic features r*z_j*z_k.
        4. Subtract the known reaction contribution r*z
           from the derivative targets.

    The resulting regression problem is:

        Theta @ Xi ≈ dz/dt - r*z

    Returns the library, regression targets, original states,
    derivatives, parameters and quadratic index pairs.
    """
    Z_train = np.asarray(Z_train, dtype=float)
    mu_train = np.asarray(mu_train, dtype=float)

    if Z_train.ndim != 3 or Z_train.shape[1] < 3:
        raise ValueError("Z_train requires at least 3 time snapshots.")
    if mu_train.shape != (Z_train.shape[0], 3):
        raise ValueError("mu_train must have shape (n_trajectories, 3).")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_traj, n_times, n_modes = Z_train.shape

    Z_dot_train = np.gradient(
        Z_train, dt, axis=1, edge_order=2
    )

    Z = Z_train.reshape(-1, n_modes)
    Z_dot = Z_dot_train.reshape(-1, n_modes)

    D = np.repeat(10.0 ** mu_train[:, 0], n_times)
    r = np.repeat(10.0 ** (mu_train[:, 1] + 2.0), n_times)

    pairs = list(combinations_with_replacement(range(n_modes), 2))
    pair_indices = np.asarray(pairs, dtype=int)

    Theta_linear = D[:, None] * Z

    Theta_quadratic = (
        r[:, None]
        * Z[:, pair_indices[:, 0]]
        * Z[:, pair_indices[:, 1]]
    )

    Theta = np.hstack((Theta_linear, Theta_quadratic))
    Y = Z_dot - r[:, None] * Z

    if not np.all(np.isfinite(Theta)) or not np.all(np.isfinite(Y)):
        raise ValueError("SINDy regression matrices contain non-finite values.")

    return Theta, Y, Z, Z_dot, r, pair_indices


def ridge_fit(X, Y, alpha=1e-4):
    """
    Solve multi-output Ridge regression without an intercept.

    Minimizes:

        ||Y - X @ Xi||_F^2 + alpha * ||Xi||_F^2

    Ridge regularization stabilizes coefficient identification
    when library columns are highly correlated.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    gram = X.T @ X
    rhs = X.T @ Y

    return np.linalg.solve(
        gram + alpha * np.eye(X.shape[1]),
        rhs
    )


def stlsq(X, Y, alpha=1e-4, threshold=0.1, max_iter=20):
    """
    Identify sparse SINDy coefficients using Ridge + STLSQ.

    The algorithm alternates between:
        1. Ridge regression on the currently active features.
        2. Removal of coefficients below the sparsity threshold.
        3. Ridge refitting on the retained terms.

    The regression is performed separately for each state
    equation, allowing different active terms.

    IMPORTANT:
        X must already be feature-normalized.
        The threshold applies to coefficients in normalized
        feature coordinates, not physical coefficients.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    Xi = ridge_fit(X, Y, alpha=alpha)

    for _ in range(max_iter):
        active = np.abs(Xi) >= threshold
        Xi_new = np.zeros_like(Xi)

        for j in range(Y.shape[1]):
            mask = active[:, j]

            if np.any(mask):
                Xi_new[mask, j] = ridge_fit(
                    X[:, mask],
                    Y[:, j:j + 1],
                    alpha=alpha
                ).ravel()

        new_active = np.abs(Xi_new) >= threshold
        Xi = Xi_new

        if np.array_equal(new_active, active):
            break

    return Xi


def fit_sindy(Theta, Y, alpha=1e-4, threshold=0.1, max_iter=20):
    """
    Normalize SINDy features and identify the coefficient matrix.

    Each library column is divided by its Euclidean norm before
    regression so that features with different physical scales
    are treated consistently.

    STLSQ is applied to the normalized library, after which
    coefficients are transformed back to the original units.

    Returns:
        Xi: coefficients for the original (unscaled) library
        Xi_scaled: coefficients for the normalized library
        feature_norms: scaling factors
    """
    feature_norms = np.linalg.norm(Theta, axis=0)

    if np.any(feature_norms == 0):
        raise ValueError("SINDy library contains zero-norm features.")

    Theta_scaled = Theta / feature_norms[None, :]

    Xi_scaled = stlsq(
        Theta_scaled, Y,
        alpha=alpha,
        threshold=threshold,
        max_iter=max_iter
    )

    Xi = Xi_scaled / feature_norms[:, None]

    return Xi, Xi_scaled, feature_norms


def sindy_rhs(t, z, mu, Xi, pairs):
    """
    Evaluate the identified continuous-time SINDy vector field.

    The physical parameter dependence is:

        D = 10**mu1
        r = 10**(mu2 + 2)

    The reconstructed dynamical system is:

        dz/dt = r*z + Theta(z, D, r) @ Xi

    The third physical parameter affects the initial condition
    rather than entering this vector field explicitly.
    """
    D = 10.0 ** mu[0]
    r = 10.0 ** (mu[1] + 2.0)

    features = np.concatenate((
        D * z,
        r * z[pairs[:, 0]] * z[pairs[:, 1]]
    ))

    return r * z + features @ Xi


def simulate_sindy(z0, mu, Xi, t_eval, pairs,
                   rtol=1e-6, atol=1e-8, max_norm=None):
    """
    Integrate the identified SINDy model over a requested time grid.

    Uses scipy.solve_ivp with the implicit BDF solver, suitable for
    potentially stiff reaction-diffusion reduced dynamics.

    If max_norm is supplied, a terminal event stops integration
    when the reduced-state norm reaches that threshold.

    Returns:
        Predicted states (n_times, n_modes), or None if the solver
        fails, diverges, or does not reach the requested final time.
    """
    z0 = np.asarray(z0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if t_eval.ndim != 1 or len(t_eval) < 2:
        raise ValueError("t_eval must contain at least two times.")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing.")

    events = None

    if max_norm is not None:
        def divergence_event(t, z, mu, Xi, pairs):
            """Return the state-norm safety margin used to terminate divergent integration."""
            return max_norm - np.linalg.norm(z)

        divergence_event.terminal = True
        divergence_event.direction = -1
        events = divergence_event

    sol = solve_ivp(
        sindy_rhs,
        (t_eval[0], t_eval[-1]),
        z0,
        t_eval=t_eval,
        args=(mu, Xi, pairs),
        method="BDF",
        rtol=rtol,
        atol=atol,
        events=events
    )

    if not sol.success or sol.y.shape[1] != len(t_eval):
        return None

    prediction = sol.y.T

    if not np.all(np.isfinite(prediction)):
        return None

    return prediction




def tune_sindy(
    Theta, Y, Z, Z_dot, reaction_rates,
    A_fom, mu_fom, val_idx, sensor_times,
    split_idx, quadratic_pairs,
    n_modes=20,
    alphas=(1e-4, 1e-2, 1.0, 10.0),
    thresholds=(0.1, 10.0, 25.0),
    high_mu2_cutoff=0.65,
    divergence_factor=10.0,
    save_path=None
):
    """
    Tune continuous-time physics-informed SINDy via validation rollouts.

    For each combination of Ridge regularization (alpha) and STLSQ
    sparsity threshold:

    1. Normalize the physics-informed feature library.
    2. Identify the coefficient matrix using PySINDy's STLSQ.
    3. Measure training derivative-fitting error and model sparsity.
    4. Integrate the identified dynamics from T/2 to T on each
       validation trajectory, starting from its last observed state.
    5. Record integration failures and final-time prediction errors.

    Model selection prioritizes:
        1. Number of successful high-reaction integrations.
        2. Total number of successful integrations.
        3. Median final-time error on successful trajectories.

    Parameters
    ----------
    Theta : ndarray, shape (n_samples, n_features)
        Physics-informed SINDy feature matrix, built on training data.

    Y : ndarray, shape (n_samples, n_modes)
        Regression targets, corresponding to dz/dt - r*z.

    Z, Z_dot : ndarray, shape (n_samples, n_modes)
        Training states and estimated temporal derivatives.

    reaction_rates : ndarray, shape (n_samples,)
        Reaction coefficients associated with training observations.

    A_fom : ndarray, shape (n_trajectories, n_times, n_pod_modes)
        POD-projected reference trajectories.

    mu_fom : ndarray, shape (n_trajectories, 3)
        Physical parameters of each trajectory.

    val_idx : array-like
        Indices of validation trajectories.

    sensor_times : ndarray
        Temporal grid shared by the reference trajectories.

    split_idx : int
        Last observed snapshot index.

    quadratic_pairs : ndarray
        Index pairs defining quadratic SINDy features.

    n_modes : int
        Number of POD coefficients used for SINDy.

    alphas, thresholds : sequence
        Hyperparameter grids.

    high_mu2_cutoff : float
        Threshold defining the high-reaction region.

    divergence_factor : float
        Maximum allowed state norm relative to the observed trajectory.

    save_path : str or Path, optional
        If provided, save the tuning table as CSV.

    Returns
    -------
    results_df : pandas.DataFrame
        Complete tuning results, sorted by the selection criterion.

    best_Xi : ndarray
        Coefficient matrix of the selected SINDy model.

    best_config : dict
        Selected hyperparameters and validation statistics.
    """
    from itertools import product
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from pysindy import STLSQ

    val_idx = np.asarray(val_idx, dtype=int)
    high_reaction = mu_fom[val_idx, 1] >= high_mu2_cutoff

    feature_norms = np.linalg.norm(Theta, axis=0)

    if np.any(feature_norms == 0):
        raise ValueError("SINDy library contains zero-norm features.")

    Theta_scaled = Theta / feature_norms[None, :]
    t_extrap = sensor_times[split_idx:]

    results = []
    models = {}

    for alpha, threshold in product(alphas, thresholds):

        # Sparse regression in normalized feature coordinates
        optimizer = STLSQ(
            threshold=threshold,
            alpha=alpha,
            normalize_columns=False,
            unbias=False,
            max_iter=20
        )
        optimizer.fit(Theta_scaled, Y)

        Xi_scaled = optimizer.coef_.T
        Xi = Xi_scaled / feature_norms[:, None]

        # Training derivative-fitting error
        Z_dot_pred = (
            reaction_rates[:, None] * Z + Theta @ Xi
        )

        derivative_error = 100 * (
            np.linalg.norm(Z_dot - Z_dot_pred)
            / np.linalg.norm(Z_dot)
        )

        sparsity = 100 * (
            1 - np.count_nonzero(Xi_scaled) / Xi_scaled.size
        )

        # Autonomous extrapolation on validation trajectories
        errors = np.full(len(val_idx), np.nan)

        for k, idx in enumerate(val_idx):
            observed = A_fom[idx, :split_idx + 1, :n_modes]
            reference = A_fom[idx, -1, :n_modes]

            observed_scale = max(
                np.linalg.norm(observed, axis=1).max(),
                1e-12
            )

            prediction = simulate_sindy(
                z0=observed[-1],
                mu=mu_fom[idx],
                Xi=Xi,
                t_eval=t_extrap,
                pairs=quadratic_pairs,
                max_norm=divergence_factor * observed_scale
            )

            if prediction is not None:
                errors[k] = 100 * (
                    np.linalg.norm(reference - prediction[-1])
                    / max(np.linalg.norm(reference), 1e-12)
                )

        success = np.isfinite(errors)
        n_success = int(success.sum())

        results.append({
            "alpha": alpha,
            "threshold": threshold,
            "success": n_success,
            "high_success": int((success & high_reaction).sum()),
            "median_error": (
                float(np.median(errors[success]))
                if n_success else np.nan
            ),
            "mean_error": (
                float(np.mean(errors[success]))
                if n_success else np.nan
            ),
            "derivative_error": float(derivative_error),
            "sparsity": float(sparsity)
        })

        models[(alpha, threshold)] = Xi

    # Prioritize robustness, then prediction accuracy
    results_df = pd.DataFrame(results).sort_values(
        ["high_success", "success", "median_error"],
        ascending=[False, False, True],
        na_position="last"
    ).reset_index(drop=True)

    best_config = results_df.iloc[0].to_dict()

    best_Xi = models[(
        best_config["alpha"],
        best_config["threshold"]
    )]

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(path, index=False)

    return results_df, best_Xi, best_config



from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


def assemble_galerkin_operators(Vh, basis, quadrature_degree=4):
    """
    Assemble the reduced diffusion and quadratic reaction operators
    for the Fisher-KPP equation using the finite-element formulation.

    Given a POD basis V with N columns, the function computes:

        K_N[i,j] = integral grad(phi_i) · grad(phi_j) dx

        C[i,j,k] = integral phi_i * phi_j * phi_k dx

    The quadratic tensor is assembled efficiently through
    weighted FEM mass matrices:

        C[:,:,k] = V.T @ M(phi_k) @ V

    where M(phi_k) is the mass matrix weighted by POD mode phi_k.

    Since the POD basis is M-orthonormal, the reduced mass matrix
    is approximately the identity and does not need to be inverted.

    Parameters
    ----------
    Vh : dolfin.FunctionSpace
        FEM function space used to construct the POD basis.

    basis : ndarray, shape (n_dofs, n_modes)
        POD basis expressed in the original FEM DOF ordering.

    quadrature_degree : int
        Quadrature degree for the quadratic reaction tensor.

    Returns
    -------
    K_N : ndarray, shape (n_modes, n_modes)
        Reduced stiffness matrix.

    C : ndarray, shape (n_modes, n_modes, n_modes)
        Quadratic reaction tensor.
    """
    import dolfin as df


    V = np.asarray(basis, dtype=float)
    n_dofs, n_modes = V.shape

    if n_dofs != Vh.dim():
        raise ValueError("POD basis and FEM space are incompatible.")

    trial = df.TrialFunction(Vh)
    test = df.TestFunction(Vh)

    def to_csr(matrix):
        """Convert an assembled FEM matrix to SciPy CSR format for reduced projection."""
        backend = df.as_backend_type(matrix).mat()
        indptr, indices, data = backend.getValuesCSR()
        return csr_matrix(
            (data.copy(), indices.copy(), indptr.copy()),
            shape=backend.size
        )

    # Reduced FEM stiffness matrix
    K_full = to_csr(
        df.assemble(df.inner(df.grad(trial), df.grad(test)) * df.dx)
    )

    K_N = V.T @ (K_full @ V)
    K_N = 0.5 * (K_N + K_N.T)

    # Quadratic reaction tensor via weighted mass matrices
    C = np.empty((n_modes, n_modes, n_modes))

    dx_q = df.dx(
        metadata={"quadrature_degree": quadrature_degree}
    )

    for k in range(n_modes):
        phi_k = df.Function(Vh)
        phi_k.vector().set_local(V[:, k].copy())
        phi_k.vector().apply("insert")

        weighted_mass = df.assemble(phi_k * trial * test * dx_q)
        M_k = to_csr(weighted_mass)

        C[:, :, k] = V.T @ (M_k @ V)

    # Enforce the analytical permutation symmetry of C
    C = (
        C
        + C.transpose(0, 2, 1)
        + C.transpose(1, 0, 2)
        + C.transpose(1, 2, 0)
        + C.transpose(2, 0, 1)
        + C.transpose(2, 1, 0)
    ) / 6.0

    return K_N, C



def simulate_galerkin(z0, mu, K_N, C, t_eval,
                      divergence_limit=None,
                      rtol=1e-6, atol=1e-8):
    """
    Predict the reduced Fisher-KPP state by integrating POD-Galerkin.

    The autonomous reduced dynamics are:

        dz/dt = -D*K_N@z + r*(z - C(z,z))

    where:
        D = 10**mu[0]
        r = 10**(mu[1] + 2)

    Integration starts from the supplied POD state z0 and uses BDF,
    an implicit solver suitable for potentially stiff dynamics.

    A terminal event optionally stops trajectories whose reduced-state
    norm exceeds divergence_limit. This is a numerical diagnostic,
    not an enforced physical positivity or saturation constraint.

    Returns
    -------
    ndarray, shape (len(t_eval), n_modes), or None
        Complete predicted trajectory, or None if integration fails
        or terminates before the final requested time.
    """
    z0 = np.asarray(z0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    D = 10.0 ** mu[0]
    r = 10.0 ** (mu[1] + 2.0)

    def rhs(t, z):
        """Evaluate the projected Fisher-KPP diffusion and quadratic reaction terms."""
        q = np.einsum("ijk,j,k->i", C, z, z, optimize=True)
        return -D * (K_N @ z) + r * (z - q)

    events = None

    if divergence_limit is not None:
        def divergence_event(t, z):
            """Return the state-norm safety margin used to terminate divergent integration."""
            return divergence_limit - np.linalg.norm(z)

        divergence_event.terminal = True
        divergence_event.direction = -1
        events = divergence_event

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        z0,
        t_eval=t_eval,
        method="BDF",
        rtol=rtol,
        atol=atol,
        events=events
    )

    success = (
        sol.status == 0
        and sol.y.shape == (len(z0), len(t_eval))
        and np.all(np.isfinite(sol.y))
    )

    return sol.y.T if success else None



def benchmark_galerkin(
    mode_counts, K_full, C_full,
    A_fom, u_fom, mu_fom, indices,
    Vn, M, sensor_times, split_idx,
    high_mu2_cutoff=0.65,
    divergence_factor=10.0
):
    """
    Benchmark POD-Galerkin temporal extrapolation for multiple POD sizes.

    For every reduced dimension N:
      1. Extract the N-mode Galerkin operators from the largest operators.
      2. Initialize each trajectory at the last observed POD state T/2.
      3. Integrate autonomously from T/2 to T.
      4. Evaluate final-time projection, coefficient and full FEM L2 errors.
      5. Record numerical failures and high-reaction success rates.

    All candidate dimensions use identical validation trajectories,
    initial observation times and prediction horizons.

    Projection errors are available even when an integration fails.

    Returns
    -------
    dict
        Per-dimension results including individual errors and
        success masks for subsequent comparison and plotting.
    """
    import support_functions as sf

    indices = np.asarray(indices, dtype=int)
    t_eval = np.asarray(sensor_times[split_idx:], dtype=float)

    results = {}

    for n in mode_counts:
        K_N = K_full[:n, :n]
        C_N = C_full[:n, :n, :n]
        basis = Vn[:, :n]

        projection = np.empty(len(indices))
        prediction = np.full(len(indices), np.nan)
        dynamics = np.full(len(indices), np.nan)
        success = np.zeros(len(indices), dtype=bool)

        for k, idx in enumerate(indices):
            u_ref = u_fom[idx, -1]
            a_ref = A_fom[idx, -1, :n]

            # Best spatial approximation within the N-mode POD space
            u_projected = sf.reconstruct_pod(a_ref, basis)

            projection[k] = 100 * sf.relative_l2_error(
                u_ref[None, :],
                u_projected[None, :],
                M
            )[0]

            # Known POD history provides the initial reduced state
            observed = A_fom[idx, :split_idx + 1, :n]
            z0 = observed[-1]

            reference_scale = max(
                np.linalg.norm(observed, axis=1).max(),
                1e-12
            )

            z_pred = simulate_galerkin(
                z0, mu_fom[idx], K_N, C_N, t_eval,
                divergence_limit=divergence_factor * reference_scale
            )

            if z_pred is None:
                continue

            success[k] = True
            a_pred = z_pred[-1]
            u_pred = sf.reconstruct_pod(a_pred, basis)

            prediction[k] = 100 * sf.relative_l2_error(
                u_ref[None, :],
                u_pred[None, :],
                M
            )[0]

            dynamics[k] = 100 * (
                np.linalg.norm(a_ref - a_pred)
                / max(np.linalg.norm(a_ref), 1e-12)
            )

        high_reaction = mu_fom[indices, 1] >= high_mu2_cutoff

        results[n] = {
            "projection_error": projection,
            "prediction_error": prediction,
            "dynamics_error": dynamics,
            "success": success,
            "high_success": int(np.sum(success & high_reaction)),
            "high_total": int(np.sum(high_reaction))
        }

    return results




def fit_hybrid_correction(A_train, mu_train, K, C, dt, alpha=1e-3):
    """
    Identify a parameter-dependent linear correction to POD-Galerkin.

    Method
    ------
    1. Estimate time derivatives of the observed POD trajectories.
    2. Evaluate the known POD-Galerkin vector field on those states.
    3. Compute the residual: dz/dt - f_Galerkin(z, mu).
    4. Fit the correction D*B_D*z + r*B_r*z using normalized
       library features and Ridge regression.

    Parameters
    ----------
    A_train : (n_trajectories, n_times, n_modes)
        Observed training coefficients only.
    mu_train : (n_trajectories, 3)
        Physical parameters associated with the trajectories.
    K, C : ndarray
        Reduced stiffness matrix and quadratic Galerkin tensor.
    dt : float
        Snapshot sampling interval.
    alpha : float
        Ridge regularization on normalized feature coefficients.

    Returns
    -------
    B_D, B_r : (n_modes, n_modes)
        Identified diffusion- and reaction-scaled corrections.
    diagnostics : dict
        Training derivative errors before and after correction.
    """
    A_train = np.asarray(A_train, dtype=float)
    n_traj, n_times, n = A_train.shape

    Z_dot = np.gradient(A_train, dt, axis=1, edge_order=2)
    Z = A_train.reshape(-1, n)
    Z_dot = Z_dot.reshape(-1, n)

    D = np.repeat(10.0 ** mu_train[:, 0], n_times)
    r = np.repeat(10.0 ** (mu_train[:, 1] + 2.0), n_times)

    quadratic = np.einsum(
        "ijk,bj,bk->bi", C, Z, Z, optimize=True
    )

    F_galerkin = (
        -D[:, None] * (Z @ K.T)
        + r[:, None] * (Z - quadratic)
    )

    residual = Z_dot - F_galerkin
    Theta = np.hstack((D[:, None] * Z, r[:, None] * Z))

    norms = np.linalg.norm(Theta, axis=0)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm hybrid feature.")

    X = Theta / norms
    Xi_scaled = np.linalg.solve(
        X.T @ X + alpha * np.eye(X.shape[1]),
        X.T @ residual
    )
    Xi = Xi_scaled / norms[:, None]

    B_D = Xi[:n].T
    B_r = Xi[n:].T

    F_hybrid = F_galerkin + Theta @ Xi
    denominator = max(np.linalg.norm(Z_dot), 1e-12)

    diagnostics = {
        "galerkin_derivative_error":
            100 * np.linalg.norm(Z_dot - F_galerkin) / denominator,
        "hybrid_derivative_error":
            100 * np.linalg.norm(Z_dot - F_hybrid) / denominator
    }

    return B_D, B_r, diagnostics


def simulate_hybrid(z0, mu, K, C, B_D, B_r, t_eval,
                    correction_strength=1.0,
                    divergence_limit=None,
                    rtol=1e-6, atol=1e-8):
    """
    Integrate POD-Galerkin plus a learned residual correction.

    Dynamics:
        dz/dt = -D*K*z + r*(z-C(z,z))
                + lambda*(D*B_D*z + r*B_r*z)

    BDF is used for numerical integration. Failed, incomplete,
    or divergent integrations return None.

    lambda=0 recovers the pure POD-Galerkin dynamics.
    """
    from scipy.integrate import solve_ivp

    z0 = np.asarray(z0, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    D = 10.0 ** mu[0]
    r = 10.0 ** (mu[1] + 2.0)

    def rhs(t, z):
        """Evaluate Galerkin dynamics plus the fitted parameter-dependent closure correction."""
        q = np.einsum("ijk,j,k->i", C, z, z, optimize=True)

        physics = -D * (K @ z) + r * (z - q)
        correction = D * (B_D @ z) + r * (B_r @ z)

        return physics + correction_strength * correction

    events = None
    if divergence_limit is not None:
        def divergence_event(t, z):
            """Return the state-norm safety margin used to terminate divergent integration."""
            return divergence_limit - np.linalg.norm(z)

        divergence_event.terminal = True
        divergence_event.direction = -1
        events = divergence_event

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), z0,
        t_eval=t_eval, method="BDF",
        rtol=rtol, atol=atol, events=events
    )

    success = (
        sol.status == 0
        and sol.y.shape == (len(z0), len(t_eval))
        and np.all(np.isfinite(sol.y))
    )

    return sol.y.T if success else None



def prepare_pod_lstm_data(A, mu, train_idx, split_idx, n_modes=20):
    """Standardize POD states and parameters using observed training data only."""
    A = np.asarray(A, dtype=float)[..., :n_modes]
    mu = np.asarray(mu, dtype=float)

    observed = A[train_idx, :split_idx + 1]
    a_mean = observed.reshape(-1, n_modes).mean(axis=0)
    a_std = np.maximum(observed.reshape(-1, n_modes).std(axis=0), 1e-8)

    mu_mean = mu[train_idx].mean(axis=0)
    mu_std = np.maximum(mu[train_idx].std(axis=0), 1e-8)

    scaling = {
        "a_mean": a_mean,
        "a_std": a_std,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
    }

    return (
        (A - a_mean) / a_std,
        (mu - mu_mean) / mu_std,
        scaling
    )


class ParametricPODLSTM:
    """Factory for a parameter-conditioned, increment-predicting PyTorch LSTM."""

    @staticmethod
    def build(n_modes, n_params=3, hidden_size=64):
        """Return a PyTorch LSTM that predicts the next standardized POD state.

        Concatenate n_params parameters to each history state of n_modes values.
        The network adds a learned increment to the final history state; hidden_size
        sets the recurrent and head widths."""
        import torch
        from torch import nn

        class Model(nn.Module):
            """Parameter-conditioned LSTM with a residual next-state prediction head."""
            def __init__(self):
                """Initialize the recurrent layer and nonlinear increment-prediction head."""
                super().__init__()
                self.lstm = nn.LSTM(
                    n_modes + n_params,
                    hidden_size,
                    batch_first=True
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, n_modes)
                )

            def forward(self, history, parameters):
                """Predict the next state from batched history and standardized parameters.

                history has shape (batch,history_length,n_modes), parameters has shape
                (batch,n_params), and the result has shape (batch,n_modes)."""
                p = parameters[:, None, :].expand(
                    -1, history.shape[1], -1
                )
                out, _ = self.lstm(torch.cat([history, p], dim=-1))
                return history[:, -1] + self.head(out[:, -1])

        return Model()


def rollout_pod_lstm(model, history, mu, scaling, n_steps, device="cpu"):
    """
    Autoregressively forecast n_steps beyond the supplied observed history.

    history: physical (unscaled) POD coefficients, shape (seq_len, n_modes).
    Returns physical POD coefficients, including the initial anchor state.
    """
    import torch

    model.eval()
    h = (np.asarray(history) - scaling["a_mean"]) / scaling["a_std"]
    p = (np.asarray(mu) - scaling["mu_mean"]) / scaling["mu_std"]

    h = torch.as_tensor(h, dtype=torch.float32, device=device)[None]
    p = torch.as_tensor(p, dtype=torch.float32, device=device)[None]

    predictions = [np.asarray(history[-1], dtype=float)]

    with torch.no_grad():
        for _ in range(n_steps):
            next_state = model(h, p)
            physical = (
                next_state[0].cpu().numpy() * scaling["a_std"]
                + scaling["a_mean"]
            )
            predictions.append(physical)
            h = torch.cat([h[:, 1:], next_state[:, None]], dim=1)

    return np.asarray(predictions)


def train_pod_lstm(
    A_scaled, mu_scaled, mu_original, scaling,
    train_idx, val_idx, split_idx,
    Vn, M, u_fom,
    seq_len=5, hidden_size=64, weight_strength=0.0,
    high_mu2_cutoff=0.65, epochs=200, patience=12,
    validation_every=5, batch_size=128, learning_rate=1e-3,
    seed=42, device="cpu"
):
    """
    Train a parameter-conditioned POD-LSTM on observed snapshots only.

    The objective is weighted one-step MSE on standardized POD coefficients.
    Checkpoints are selected by full-rollout final FEM L2 validation error,
    prioritizing the high-reaction subset. No validation targets enter
    gradient-based training.

    Returns the best model, training history and validation diagnostics.
    """
    import copy
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import support_functions as sf

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_modes = A_scaled.shape[-1]
    X, P, Y, W = [], [], [], []

    for idx in train_idx:
        weight = 1.0 + weight_strength * (
            mu_original[idx, 1] >= high_mu2_cutoff
        )
        for k in range(seq_len, split_idx + 1):
            X.append(A_scaled[idx, k-seq_len:k])
            P.append(mu_scaled[idx])
            Y.append(A_scaled[idx, k])
            W.append(weight)

    dataset = TensorDataset(*[
        torch.tensor(np.asarray(a), dtype=torch.float32)
        for a in (X, P, Y, W)
    ])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ParametricPODLSTM.build(
        n_modes, mu_scaled.shape[1], hidden_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    high_mask = mu_original[val_idx, 1] >= high_mu2_cutoff

    history = []
    best_score = np.inf
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x, p, y, w in loader:
            x, p, y, w = [
                v.to(device) for v in (x, p, y, w)
            ]

            optimizer.zero_grad()
            pred = model(x, p)

            sample_loss = ((pred - y)**2).mean(dim=1)
            loss = (w * sample_loss).sum() / w.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if epoch % validation_every != 0 and epoch != 1:
            continue

        errors = []
        for idx in val_idx:
            observed = (
                A_scaled[idx, split_idx-seq_len+1:split_idx+1]
                * scaling["a_std"] + scaling["a_mean"]
            )

            forecast = rollout_pod_lstm(
                model, observed,
                mu_original[idx], scaling,
                A_scaled.shape[1] - split_idx - 1,
                device=device
            )

            u_pred = sf.reconstruct_pod(forecast[-1], Vn[:, :n_modes])
            err = sf.relative_l2_error(
                u_fom[idx, -1:],
                u_pred[None, :],
                M
            )[0]
            errors.append(100 * err)

        errors = np.asarray(errors)
        score = (
            np.median(errors[high_mask])
            if high_mask.any() else np.median(errors)
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "high_median": float(score),
            "overall_median": float(np.median(errors))
        })

        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid POD-LSTM checkpoint was found.")

    model.load_state_dict(best_state)
    model.eval()

    return model, history, {
        "best_high_median": best_score,
        "best_epoch": history[
            next(i for i, h in enumerate(history)
                 if h["high_median"] == best_score)
        ]["epoch"]
    }


def evaluate_pod_lstm(
    model, A_fom, mu_fom, indices, split_idx,
    seq_len, scaling, Vn, M, u_fom,
    high_mu2_cutoff=0.65, device="cpu"
):
    """Evaluate final FEM L2 errors after autonomous POD-LSTM rollout."""
    import support_functions as sf

    indices = np.asarray(indices)
    n_modes = len(scaling["a_mean"])
    n_steps = A_fom.shape[1] - split_idx - 1

    errors = np.full(len(indices), np.nan)
    predictions = {}

    for j, idx in enumerate(indices):
        history = A_fom[
            idx, split_idx-seq_len+1:split_idx+1, :n_modes
        ]

        forecast = rollout_pod_lstm(
            model, history, mu_fom[idx], scaling, n_steps,
            device=device
        )

        if not np.all(np.isfinite(forecast)):
            continue

        predictions[int(idx)] = forecast

        u_pred = sf.reconstruct_pod(forecast[-1], Vn[:, :n_modes])
        errors[j] = 100 * sf.relative_l2_error(
            u_fom[idx, -1:],
            u_pred[None, :],
            M
        )[0]

    high = mu_fom[indices, 1] >= high_mu2_cutoff
    success = np.isfinite(errors)

    return {
        "errors": errors,
        "success": success,
        "high_mask": high,
        "predictions": predictions,
        "median": np.median(errors[success]) if success.any() else np.nan,
        "high_median": (
            np.median(errors[success & high])
            if (success & high).any() else np.nan
        )
    }



def discrete_sindy_library(z, p):
    """Build a constant, linear, quadratic and parameter-dependent library."""
    from itertools import combinations_with_replacement

    z = np.atleast_2d(np.asarray(z, dtype=float))
    p = np.atleast_2d(np.asarray(p, dtype=float))

    if len(p) == 1 and len(z) > 1:
        p = np.repeat(p, len(z), axis=0)

    if len(z) != len(p):
        raise ValueError("State and parameter sample counts differ.")

    n, rank = z.shape
    parts = [np.ones((n, 1)), z]

    pairs = list(combinations_with_replacement(range(rank), 2))
    parts.append(np.column_stack([z[:, i] * z[:, j] for i, j in pairs]))
    parts.append(p)
    parts.append(np.concatenate([p[:, j:j+1] * z
                                 for j in range(p.shape[1])], axis=1))

    return np.column_stack(parts)


def fit_discrete_sindy(A_train, mu_train, alpha=1.0, threshold=0.01):
    """
    Identify a global parametric one-step increment map.

    A_train contains observed POD trajectories only, shape (n_mu, n_t, N).
    Both state and parameter normalization use training data exclusively.
    Features are RMS-normalized before Ridge/STLSQ fitting.
    """
    from sklearn.linear_model import Ridge

    A_train = np.asarray(A_train, dtype=float)
    mu_train = np.asarray(mu_train, dtype=float)

    n_traj, n_times, rank = A_train.shape

    z_mean = A_train.reshape(-1, rank).mean(axis=0)
    z_std = np.maximum(A_train.reshape(-1, rank).std(axis=0), 1e-8)
    p_mean = mu_train.mean(axis=0)
    p_std = np.maximum(mu_train.std(axis=0), 1e-8)

    Z = (A_train - z_mean) / z_std
    P = (mu_train - p_mean) / p_std

    Z0 = Z[:, :-1].reshape(-1, rank)
    Z1 = Z[:, 1:].reshape(-1, rank)
    P0 = np.repeat(P, n_times - 1, axis=0)

    target = Z1 - Z0
    Theta = discrete_sindy_library(Z0, P0)

    feature_scale = np.maximum(
        np.sqrt(np.mean(Theta**2, axis=0)), 1e-10
    )
    X = Theta / feature_scale

    Xi_scaled = np.zeros((X.shape[1], rank))

    for j in range(rank):
        coef = Ridge(alpha=alpha, fit_intercept=False).fit(
            X, target[:, j]
        ).coef_

        for _ in range(5):
            active = np.abs(coef) >= threshold
            active[0] = True

            new_coef = np.zeros_like(coef)
            new_coef[active] = Ridge(
                alpha=alpha, fit_intercept=False
            ).fit(X[:, active], target[:, j]).coef_

            if np.array_equal(
                np.abs(new_coef) >= threshold,
                np.abs(coef) >= threshold
            ):
                coef = new_coef
                break

            coef = new_coef

        Xi_scaled[:, j] = coef

    Xi = Xi_scaled / feature_scale[:, None]

    return {
        "rank": rank,
        "alpha": alpha,
        "threshold": threshold,
        "Xi": Xi,
        "z_mean": z_mean,
        "z_std": z_std,
        "p_mean": p_mean,
        "p_std": p_std,
        "active_terms": int(np.count_nonzero(Xi))
    }


def rollout_discrete_sindy(model, a0, mu, n_steps,
                           divergence_limit=1e4):
    """
    Autoregressively predict POD coefficients from the last observed state.

    Returns an array including the initial state, or None if the forecast
    diverges or becomes nonfinite. The bound is applied to standardized states.
    """
    rank = model["rank"]
    z = (np.asarray(a0, dtype=float)[:rank] - model["z_mean"]) / model["z_std"]
    p = (np.asarray(mu, dtype=float) - model["p_mean"]) / model["p_std"]

    forecast = [np.asarray(a0, dtype=float)[:rank].copy()]

    for _ in range(n_steps):
        Theta = discrete_sindy_library(z[None, :], p[None, :])
        z = z + (Theta @ model["Xi"]).ravel()

        if (
            not np.all(np.isfinite(z))
            or np.max(np.abs(z)) > divergence_limit
        ):
            return None

        forecast.append(z * model["z_std"] + model["z_mean"])

    return np.asarray(forecast)


def evaluate_discrete_sindy(model, A_fom, mu_fom, indices,
                            split_idx, Vn, M, u_fom,
                            high_mu2_cutoff=0.65):
    """Compute final-time FEM L2 prediction errors after autonomous rollout."""
    import support_functions as sf

    indices = np.asarray(indices, dtype=int)
    n = model["rank"]
    n_steps = A_fom.shape[1] - split_idx - 1

    errors = np.full(len(indices), np.nan)
    forecasts = {}

    for j, idx in enumerate(indices):
        predicted = rollout_discrete_sindy(
            model,
            A_fom[idx, split_idx, :n],
            mu_fom[idx],
            n_steps
        )

        if predicted is None:
            continue

        forecasts[int(idx)] = predicted

        u_pred = sf.reconstruct_pod(predicted[-1], Vn[:, :n])

        errors[j] = 100 * sf.relative_l2_error(
            u_fom[idx, -1:],
            u_pred[None, :],
            M
        )[0]

    success = np.isfinite(errors)
    high = mu_fom[indices, 1] >= high_mu2_cutoff

    return {
        "errors": errors,
        "success": success,
        "high_mask": high,
        "forecasts": forecasts,
        "median": np.median(errors[success]) if success.any() else np.nan,
        "high_median": (
            np.median(errors[success & high])
            if np.any(success & high) else np.nan
        ),
        "high_max": (
            np.max(errors[success & high])
            if np.any(success & high) else np.nan
        )
    }

def benchmark_temporal_models(
    forecasters, A_fom, u_fom, mu_fom, test_idx,
    Vn, M, split_idx, mode_counts, high_mu2_cutoff=0.65
):
    """
    Evaluate frozen reduced-dynamics models on unseen test trajectories.

    Each forecaster is a callable:
        forecast = forecaster(trajectory_index)

    It must return an array of shape (n_forecast_times, n_modes),
    including the POD state at T/2, or None on integration failure.

    Returns per-model time-dependent FEM L2 errors, final errors,
    completion masks, high-reaction masks and POD projection errors.
    """
    import support_functions as sf

    indices = np.asarray(test_idx, dtype=int)
    n_times = u_fom.shape[1] - split_idx
    high_reaction = mu_fom[indices, 1] >= high_mu2_cutoff

    results = {}

    for name, forecast_fn in forecasters.items():
        n = int(mode_counts[name])
        basis = Vn[:, :n]

        errors = np.full((len(indices), n_times), np.nan)
        projection_errors = np.full(len(indices), np.nan)
        completed = np.zeros(len(indices), dtype=bool)
        forecasts = {}
        failures = {}

        for j, idx in enumerate(indices):
            reference = u_fom[idx, split_idx:]

            # Irreducible spatial approximation error at the final time.
            u_projection = sf.reconstruct_pod(
                A_fom[idx, -1, :n], basis
            )
            projection_errors[j] = 100 * sf.relative_l2_error(
                u_fom[idx, -1:],
                u_projection[None, :],
                M
            )[0]

            try:
                coefficients = forecast_fn(int(idx))

                if coefficients is None:
                    failures[int(idx)] = "Incomplete or failed rollout"
                    continue

                coefficients = np.asarray(coefficients, dtype=float)

                if coefficients.shape != (n_times, n):
                    raise ValueError(
                        f"Expected {(n_times, n)}, got {coefficients.shape}"
                    )

                if not np.all(np.isfinite(coefficients)):
                    raise ValueError("Nonfinite POD coefficients")

                u_pred = sf.reconstruct_pod(coefficients, basis)

                trajectory_errors = 100 * sf.relative_l2_error(
                    reference, u_pred, M
                )

                if not np.all(np.isfinite(trajectory_errors)):
                    raise ValueError("Nonfinite FEM L2 error")

                errors[j] = trajectory_errors
                completed[j] = True
                forecasts[int(idx)] = coefficients

            except (ValueError, RuntimeError, FloatingPointError,
                    OverflowError, np.linalg.LinAlgError) as exc:
                failures[int(idx)] = f"{type(exc).__name__}: {exc}"

        results[name] = {
            "errors": errors,
            "final_errors": errors[:, -1],
            "success": completed,
            "high_mask": high_reaction.copy(),
            "projection_errors": projection_errors,
            "forecasts": forecasts,
            "failures": failures,
            "indices": indices.copy(),
            "n_modes": n
        }

    return results



def refit_pod_lstm(
    A_scaled, mu_scaled, mu_original, train_idx,
    seq_len=5, hidden_size=128, weight_strength=0.0,
    high_mu2_cutoff=0.65, epochs=200,
    batch_size=128, learning_rate=1e-3,
    seed=42, device="cpu"
):
    """
    Refit a previously selected POD-LSTM on complete training trajectories.

    Architecture, loss weighting, optimization settings and training
    budget are kept fixed. No validation or test trajectory is used
    for training or checkpoint selection.

    A_scaled: (n_trajectories, n_times, n_modes), normalized with
              statistics estimated on full-window training data.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_modes = A_scaled.shape[-1]
    n_times = A_scaled.shape[1]

    X, P, Y, W = [], [], [], []

    for idx in train_idx:
        weight = 1.0 + weight_strength * (
            mu_original[idx, 1] >= high_mu2_cutoff
        )

        for k in range(seq_len, n_times):
            X.append(A_scaled[idx, k-seq_len:k])
            P.append(mu_scaled[idx])
            Y.append(A_scaled[idx, k])
            W.append(weight)

    dataset = TensorDataset(*[
        torch.tensor(np.asarray(a), dtype=torch.float32)
        for a in (X, P, Y, W)
    ])

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model = ParametricPODLSTM.build(
        n_modes=n_modes,
        n_params=mu_scaled.shape[1],
        hidden_size=hidden_size
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for x, p, y, w in loader:
            x, p, y, w = [
                v.to(device) for v in (x, p, y, w)
            ]

            optimizer.zero_grad()
            predicted = model(x, p)

            sample_loss = ((predicted - y) ** 2).mean(dim=1)
            loss = (w * sample_loss).sum() / w.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            epoch_losses.append(loss.item())

        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    return model, np.asarray(losses)



def tune_full_window_lstm_config(
    A_scaled, mu_scaled, mu_original, scaling,
    train_idx, val_idx, A_fom, u_fom, Vn, M,
    split_idx, seq_len, hidden_size,
    weight_strength, learning_rate,
    max_epochs=350, eval_every=25,
    batch_size=128, seed=42, device="cpu"
):
    """
    Train one full-window POD-LSTM configuration and select its best epoch.

    Training:
        All available time transitions of training trajectories.

    Checkpoint selection:
        Autonomous validation rollout from T/2 to T, prioritizing
        numerical completion and median FEM L2 error at high reaction.

    Returns:
        Best PyTorch model, best-checkpoint metrics, and epoch history.
    """
    import copy
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_modes = A_scaled.shape[-1]
    n_times = A_scaled.shape[1]
    train_idx = np.asarray(train_idx, dtype=int)

    X, P, Y, W = [], [], [], []

    for idx in train_idx:
        weight = 1.0 + weight_strength * (
            mu_original[idx, 1] >= 0.65
        )

        for k in range(seq_len, n_times):
            X.append(A_scaled[idx, k-seq_len:k])
            P.append(mu_scaled[idx])
            Y.append(A_scaled[idx, k])
            W.append(weight)

    dataset = TensorDataset(
        torch.tensor(np.asarray(X), dtype=torch.float32),
        torch.tensor(np.asarray(P), dtype=torch.float32),
        torch.tensor(np.asarray(Y), dtype=torch.float32),
        torch.tensor(np.asarray(W), dtype=torch.float32)
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model = ParametricPODLSTM.build(
        n_modes=n_modes,
        n_params=mu_scaled.shape[1],
        hidden_size=hidden_size
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    history = []
    best_key = None
    best_state = None
    best_metrics = None

    high_mask = mu_original[val_idx, 1] >= 0.65

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []

        for x, p, y, w in loader:
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            prediction = model(x, p)

            per_sample = ((prediction - y) ** 2).mean(dim=1)
            loss = (per_sample * w).sum() / w.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            batch_losses.append(float(loss.item()))

        if epoch % eval_every != 0 and epoch != max_epochs:
            continue

        result = evaluate_pod_lstm(
            model=model,
            A_fom=A_fom,
            mu_fom=mu_original,
            indices=val_idx,
            split_idx=split_idx,
            seq_len=seq_len,
            scaling=scaling,
            Vn=Vn,
            M=M,
            u_fom=u_fom,
            device=device
        )

        errors = np.asarray(result["errors"])
        success = result["success"] & np.isfinite(errors)
        high_success = success & high_mask

        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "success": int(success.sum()),
            "high_success": int(high_success.sum()),
            "median": (
                float(np.median(errors[success]))
                if success.any() else np.inf
            ),
            "high_median": (
                float(np.median(errors[high_success]))
                if high_success.any() else np.inf
            ),
            "high_max": (
                float(np.max(errors[high_success]))
                if high_success.any() else np.inf
            )
        }

        history.append(metrics)

        # Selection prioritizes high-reaction completion and accuracy.
        key = (
            -metrics["high_success"],
            -metrics["success"],
            metrics["high_median"],
            metrics["high_max"],
            metrics["median"]
        )

        if best_key is None or key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics.copy()

    if best_state is None:
        raise RuntimeError("No valid LSTM checkpoint was produced.")

    model.load_state_dict(best_state)
    model.eval()

    return model, best_metrics, history



def run_final_full_window_tuning(
    A_fom, u_fom, mu_fom, train_idx, val_idx,
    Vn, M, time_grid, split_idx,
    cache_dir="artifacts/final_dynamics",
    max_epochs=350,
    eval_every=25,
    seed=42,
    device="cpu",
    version="v1",
):
    """
    Final cached hyperparameter search for the temporal ROMs.

    - DMD: trajectory-specific rolling-origin validation.
    - Continuous SINDy: full-window parametric identification.
    - Discrete SINDy: full-window parametric identification.
    - POD-LSTM: full-window training with validation checkpoints.
    - POD-Galerkin: unchanged; no data-driven training required.

    Model selection uses validation trajectories exclusively.
    All configurations and selected models are cached.
    """

    import hashlib
    import json
    from itertools import product
    from pathlib import Path
    from time import perf_counter

    import numpy as np
    import pandas as pd
    import torch
    from pysindy.optimizers import STLSQ
    import support_functions as sf

    root = Path(cache_dir)
    candidates_dir = root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    time_grid = np.asarray(time_grid, dtype=float)

    full_end = len(time_grid) - 1
    dt = float(np.diff(time_grid).mean())
    high_mask = mu_fom[val_idx, 1] >= 0.65

    if not np.allclose(np.diff(time_grid), dt):
        raise ValueError("A uniformly spaced time grid is required.")

    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Train and validation indices overlap.")

    if full_end <= split_idx:
        raise ValueError("No validation forecasting interval available.")

    # Versioned signature: changing data or search version
    # invalidates the old candidate cache.
    digest = hashlib.sha256()
    for arr in (
        train_idx, val_idx,
        A_fom[np.r_[train_idx, val_idx]],
        mu_fom[np.r_[train_idx, val_idx]],
        Vn,
        time_grid,
    ):
        a = np.ascontiguousarray(arr)
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())

    data_id = digest.hexdigest()[:16]

    def candidate_path(method, config):
        """Create the method cache directory and return a path hashed from data and configuration."""
        payload = json.dumps({
            "method": method,
            "config": config,
            "data_id": data_id,
            "version": version,
        }, sort_keys=True)

        uid = hashlib.sha256(
            payload.encode()
        ).hexdigest()[:16]

        folder = candidates_dir / method
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{uid}.pt"

    def metrics_from_errors(errors, success=None):
        """Summarize successful validation forecasts and the high-damage subset; empty groups use infinity."""
        errors = np.asarray(errors, dtype=float).reshape(-1)

        if errors.size != len(val_idx):
            raise ValueError(
                "Validation error array has an unexpected length."
            )

        if success is None:
            success = np.isfinite(errors)
        else:
            success = np.asarray(success, dtype=bool).reshape(-1)

        ok = success & np.isfinite(errors)
        high_ok = ok & high_mask

        return {
            "success": int(ok.sum()),
            "high_success": int(high_ok.sum()),
            "median": (
                float(np.median(errors[ok]))
                if ok.any() else float("inf")
            ),
            "high_median": (
                float(np.median(errors[high_ok]))
                if high_ok.any() else float("inf")
            ),
            "high_max": (
                float(np.max(errors[high_ok]))
                if high_ok.any() else float("inf")
            ),
        }

    def rank_key(metrics):
        """Rank candidates by high-damage successes, total successes, then increasing validation errors."""
        return (
            -metrics["high_success"],
            -metrics["success"],
            metrics["high_median"],
            metrics["high_max"],
            metrics["median"],
        )

    all_rows = []
    selected = {}
    selected_payloads = {}

    def save_tables():
        """Write accumulated tuning records to tuning_results.csv and return the DataFrame."""
        table = pd.DataFrame(all_rows)
        table.to_csv(root / "tuning_results.csv", index=False)
        return table

    def run_cached(method, config, train_evaluate):
        """Load or train a candidate, cache its payload, and return the payload and logged metric row."""
        path = candidate_path(method, config)

        if path.exists():
            payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
            status = "CACHE HIT"
        else:
            tic = perf_counter()
            payload = train_evaluate()
            payload["training_time"] = perf_counter() - tic
            payload["config"] = config
            payload["data_id"] = data_id
            payload["version"] = version

            # Atomic replacement avoids keeping a partial checkpoint.
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

            status = "TRAINED"

        metrics = payload["metrics"]

        row = {
            "Model": method,
            **config,
            "Success": metrics["success"],
            "High success": metrics["high_success"],
            "Median [%]": metrics["median"],
            "High median [%]": metrics["high_median"],
            "High max [%]": metrics["high_max"],
            "Time [s]": payload["training_time"],
            "Cache": str(path),
        }

        all_rows.append(row)
        save_tables()

        print(
            f"{status:9s} | {method:17s} | "
            f"{config} | "
            f"high={metrics['high_median']:.2f}%"
        )

        return payload, row

    def select_method(method, records):
        """Select the best ranked candidate, record its configuration and payload, and return its table row."""
        best_payload, best_row = min(
            records,
            key=lambda item: rank_key(item[0]["metrics"])
        )

        selected[method] = best_payload["config"]
        selected_payloads[method] = best_payload

        return best_row

    # ==========================================================
    # 1. DMD — prioritize longer autonomous forecasting horizons
    # ==========================================================

    print("\n========== DMD ==========")

    dmd_records = []
    dmd_primary = []

    # 20-step and 10-step forecasts, not the previous 5-step test.
    dmd_windows = [split_idx, 30]
    dmd_ranks = [5, 10, 15, 18, 20, 25, 30]
    dmd_starts = [0, 3, 5]

    for cut, start, rank in product(
        dmd_windows, dmd_starts, dmd_ranks
    ):
        if rank > cut - start or cut >= full_end:
            continue

        config = {
            "rank": rank,
            "start_idx": start,
            "fit_end": cut,
        }

        def train_dmd():
            """Fit and validate the current DMD candidate using the enclosing tuning configuration."""
            errors = np.full(len(val_idx), np.nan)

            for j, idx in enumerate(val_idx):
                try:
                    trajectory = A_fom[idx, :cut+1, :50]

                    dmd = fit_dmd(
                        trajectory,
                        rank=rank,
                        start_idx=start
                    )

                    prediction = predict_dmd(
                        dmd,
                        A_fom[idx, cut, :50],
                        full_end - cut
                    )

                    u_pred = sf.reconstruct_pod(
                        prediction[-1], Vn[:, :50]
                    )

                    errors[j] = 100 * sf.relative_l2_error(
                        u_fom[idx, -1:],
                        u_pred[None, :],
                        M
                    )[0]

                except (
                    ValueError, RuntimeError, FloatingPointError,
                    OverflowError, np.linalg.LinAlgError
                ):
                    continue

            return {
                "metrics": metrics_from_errors(errors),
                "errors": errors,
            }

        payload, row = run_cached(
            "DMD", config, train_dmd
        )
        dmd_records.append((payload, row))

        if cut == split_idx:
            dmd_primary.append((payload, row))

    # Select using the longest available forecast.
    best_dmd_row = select_method("DMD", dmd_primary)

    # fit_end describes validation only, not the final fit horizon.
    selected["DMD"]["fit_end"] = full_end

    # ==========================================================
    # 2. Continuous-time SINDy
    # ==========================================================

    print("\n========== CONTINUOUS SINDy ==========")

    continuous_records = []
    sindy_libraries = {}

    for n in [10, 20]:
        Theta, Y, _, _, _, pairs = build_sindy_library(
            A_fom[train_idx, :, :n],
            mu_fom[train_idx],
            dt
        )

        norms = np.maximum(
            np.linalg.norm(Theta, axis=0), 1e-12
        )

        sindy_libraries[n] = (
            Theta / norms, Y, norms, pairs
        )

    for n, alpha, threshold in product(
        [10, 20],
        [0.01, 1.0, 10.0],
        [1.0, 10.0, 25.0]
    ):
        config = {
            "n_modes": n,
            "alpha": alpha,
            "threshold": threshold,
        }

        def train_continuous():
            """Fit and validate the current continuous-time SINDy candidate."""
            X, Y, norms, pairs = sindy_libraries[n]

            optimizer = STLSQ(
                alpha=alpha,
                threshold=threshold,
                max_iter=20,
                unbias=False
            )
            optimizer.fit(X, Y)

            Xi = optimizer.coef_.T / norms[:, None]
            errors = np.full(len(val_idx), np.nan)

            for j, idx in enumerate(val_idx):
                try:
                    forecast = simulate_sindy(
                        A_fom[idx, split_idx, :n],
                        mu_fom[idx],
                        Xi,
                        time_grid[split_idx:],
                        pairs
                    )

                    # Failed integrations are counted as unsuccessful forecasts.
                    if forecast is None:
                        continue

                    forecast = np.asarray(forecast, dtype=float)

                    if (
                        forecast.shape != (full_end - split_idx + 1, n)
                        or not np.all(np.isfinite(forecast))
                    ):
                        continue

                    u_pred = sf.reconstruct_pod(
                        forecast[-1], Vn[:, :n]
                    )

                    errors[j] = 100 * sf.relative_l2_error(
                        u_fom[idx, -1:],
                        u_pred[None, :],
                        M
                    )[0]

                except (
                    ValueError, RuntimeError, FloatingPointError,
                    OverflowError, np.linalg.LinAlgError
                ):
                    continue

            return {
                "metrics": metrics_from_errors(errors),
                "Xi": Xi,
                "quadratic_pairs": np.asarray(pairs),
                "active_terms": int(np.count_nonzero(Xi)),
            }

        continuous_records.append(
            run_cached(
                "Continuous_SINDy",
                config,
                train_continuous
            )
        )

    best_continuous_row = select_method(
        "Continuous_SINDy", continuous_records
    )

    # ==========================================================
    # 3. Discrete-time SINDy
    # ==========================================================

    print("\n========== DISCRETE SINDy ==========")

    discrete_records = []

    for n, alpha, threshold in product(
        [5, 8, 10],
        [0.01, 1.0, 10.0, 100.0],
        [0.001, 0.01, 0.1]
    ):
        config = {
            "n_modes": n,
            "alpha": alpha,
            "threshold": threshold,
        }

        def train_discrete():
            """Fit and validate the current discrete-time SINDy candidate."""
            model = fit_discrete_sindy(
                A_fom[train_idx, :, :n],
                mu_fom[train_idx],
                alpha=alpha,
                threshold=threshold
            )

            result = evaluate_discrete_sindy(
                model, A_fom, mu_fom, val_idx,
                split_idx, Vn, M, u_fom
            )

            return {
                "metrics": metrics_from_errors(
                    result["errors"], result["success"]
                ),
                "model": model,
                "active_terms": int(model["active_terms"]),
            }

        discrete_records.append(
            run_cached(
                "Discrete_SINDy",
                config,
                train_discrete
            )
        )

    best_discrete_row = select_method(
        "Discrete_SINDy", discrete_records
    )

    # ==========================================================
    # 4. POD-LSTM — joint HP and checkpoint selection
    # ==========================================================

    print("\n========== POD-LSTM ==========")

    if "tune_full_window_lstm_config" not in globals():
        raise RuntimeError(
            "Add tune_full_window_lstm_config to "
            "extrapolation_functions.py before running this search."
        )

    lstm_records = []
    scaled_data = {}

    for n in [10, 20, 30]:
        scaled_data[n] = prepare_pod_lstm_data(
            A_fom, mu_fom, train_idx,
            split_idx=full_end,
            n_modes=n
        )

    lstm_space = product(
        [10, 20, 30],         # POD modes
        [3, 5],               # History length
        [64, 128],            # Hidden size
        [0.0, 4.0],           # High-reaction weighting
        [3e-4, 1e-3],         # Learning rate
    )

    for n, p, h, weight, lr in lstm_space:
        config = {
            "n_modes": n,
            "seq_len": p,
            "hidden_size": h,
            "weight_strength": weight,
            "learning_rate": lr,
            "max_epochs": max_epochs,
            "eval_every": eval_every,
            "seed": seed,
        }

        def train_lstm():
            """Train and validate the current POD-LSTM candidate."""
            A_scaled, mu_scaled, scaling = scaled_data[n]

            model, metrics, history = (
                tune_full_window_lstm_config(
                    A_scaled=A_scaled,
                    mu_scaled=mu_scaled,
                    mu_original=mu_fom,
                    scaling=scaling,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    A_fom=A_fom,
                    u_fom=u_fom,
                    Vn=Vn,
                    M=M,
                    split_idx=split_idx,
                    seq_len=p,
                    hidden_size=h,
                    weight_strength=weight,
                    learning_rate=lr,
                    max_epochs=max_epochs,
                    eval_every=eval_every,
                    seed=seed,
                    device=device,
                )
            )

            return {
                "metrics": metrics,
                "history": history,
                "scaling": scaling,
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
            }

        lstm_records.append(
            run_cached("POD_LSTM", config, train_lstm)
        )

    best_lstm_row = select_method(
        "POD_LSTM", lstm_records
    )

    # ==========================================================
    # 5. Save the selected models and complete search results
    # ==========================================================

    final_models = root / "selected"
    final_models.mkdir(exist_ok=True)

    continuous_best = selected_payloads["Continuous_SINDy"]
    discrete_best = selected_payloads["Discrete_SINDy"]
    lstm_best = selected_payloads["POD_LSTM"]

    np.savez_compressed(
        final_models / "continuous_sindy.npz",
        Xi=continuous_best["Xi"],
        quadratic_pairs=continuous_best["quadratic_pairs"],
        n_modes=selected["Continuous_SINDy"]["n_modes"],
    )

    np.savez_compressed(
        final_models / "discrete_sindy.npz",
        **discrete_best["model"]
    )

    torch.save({
        "state_dict": lstm_best["state_dict"],
        "scaling": lstm_best["scaling"],
        "config": selected["POD_LSTM"],
        "best_epoch": lstm_best["metrics"]["epoch"],
    }, final_models / "pod_lstm.pt")

    with open(final_models / "selected_configs.json", "w") as f:
        json.dump(selected, f, indent=2)

    table = save_tables()

    # Record the complete checkpoint-validation histories.
    histories = []
    for payload, row in lstm_records:
        for record in payload["history"]:
            histories.append({
                "N": row["n_modes"],
                "History": row["seq_len"],
                "Hidden": row["hidden_size"],
                "Weight": row["weight_strength"],
                "LR": row["learning_rate"],
                **record,
            })

    pd.DataFrame(histories).to_csv(
        root / "lstm_epoch_history.csv", index=False
    )

    print("\n========== FINAL SELECTION ==========")
    for method, config in selected.items():
        print(f"{method}: {config}")

    print(f"\nSaved to: {final_models}")

    return {
        "selected": selected,
        "results": table,
        "paths": {
            "root": root,
            "models": final_models,
        },
    }
