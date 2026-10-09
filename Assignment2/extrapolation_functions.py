"""DMD dei coefficienti POD per il punto IV; nessun training all'import."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DMDModel:
    operator: np.ndarray
    offset: np.ndarray
    singular_values: np.ndarray
    rank: int

    @property
    def eigenvalues(self):
        return np.linalg.eigvals(self.operator)


def fit_dmd(coefficients, rank, *, window=None, shift='zero', equilibrium=None,
            ridge=0.0, rtol=1e-10, formulation='exact'):
    """Fit a_{k+1}=c+K(a_k-c) con SVD troncata, snapshot sulle righe.

    shift: zero, mean (media della sola finestra di fit), equilibrium.
    ridge è adimensionale: alpha=ridge*sigma_max**2. Nessun clipping
    degli autovalori o del campo. Il rango richiesto deve essere disponibile.
    formulation='exact': Y V Sigma^-1 U.T; 'projected': U S U.T,
    S=U.T Y V Sigma^-1, come nel Lab 9 (dinamica nello spazio di U).
    """
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
    """Restituisce steps+1 righe: stato iniziale e steps avanzamenti dt."""
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
    """Rolling-origin backtest: fit sul prefisso, previsione fino all'ultimo dato.

    cutoffs sono indici inclusivi. La media degli errori relativi spaziali
    (norma Gram POD) sui fold sceglie il modello. Riporta anche RMSE puntuale.
    I dati futuri di ogni fold non entrano nel fit né nel calcolo dell'offset.
    Restituisce righe di risultati e righe per fold; configurazioni non
    fattibili sono marcate, senza abbassarne implicitamente il rango.
    """
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
    """dz/dtau = Theta(z) Xi, tau=t/T; PCA e scale fit solo sul training."""
    center: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    feature_scale: np.ndarray
    regression_rank: int
    derivative_error: float

    def encode(self, a):
        return ((np.asarray(a)-self.center) @ self.basis)/self.scale

    def decode(self, z):
        return self.center + (np.asarray(z)*self.scale) @ self.basis.T

    def rhs(self, tau, z):
        return polynomial_library(np.asarray(z).reshape(1, -1), self.powers)[0] @ self.coefficients


def polynomial_powers(dimension, degree):
    """Monomi unici totali fino a degree, bias compreso."""
    from itertools import combinations_with_replacement
    powers = [np.zeros(dimension, dtype=int)]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers.append(np.bincount(indices, minlength=dimension))
    return np.stack(powers)


def polynomial_library(states, powers):
    """Theta_ij = prodotto_k states_ik**powers_jk."""
    states = np.atleast_2d(states)
    return np.prod(states[:, None, :]**powers[None, :, :], axis=2)


def fit_sindy(coefficients, tau, dimension, *, degree=2, threshold=0.05,
              alpha=0.01, derivative='finite', derivatives=None, window=None,
              max_iter=20):
    """SINDy continuo con PCA locale e sequential thresholded least squares.

    derivative: finite usa differenze centrali sui soli snapshot del fit;
    autograd riceve da derivatives da/dtau della rete, non da/dt.
    Colonne Theta normalizzate in RMS; soglia e ridge agiscono sui coefficienti
    nella libreria normalizzata. Alpha moltiplica n_samples, per indipendenza
    dalla densità temporale. Nessun dato futuro nelle trasformazioni.
    """
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
    elif derivative=='autograd':
        if da is None or da.shape!=a.shape or not np.isfinite(da).all():
            raise ValueError('Derivate autograd mancanti o incompatibili.')
        zdot = (da[1:-1] @ basis)/scale
    else:
        raise ValueError('Derivata finite o autograd richiesta.')
    powers = polynomial_powers(dimension,degree)
    theta = polynomial_library(z[1:-1],powers)
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
                      int(np.linalg.matrix_rank(design)),error)


def forecast_sindy(model, initial, tau_elapsed, *, rtol=1e-7, atol=1e-9,
                   max_normalized_state=100., max_step=0.025):
    """Integra da tau=0; include lo stato iniziale esatto anche fuori dalla PCA.

    Il residuo iniziale ortogonale alla PCA è mantenuto costante, rendendo
    continuo il raccordo al ROM senza attribuirgli una dinamica identificata.
    Interrompe divergenze, senza clipping, con un evento a |z|=100.
    """
    from scipy.integrate import solve_ivp
    times=np.asarray(tau_elapsed,dtype=float)
    if times.ndim!=1 or len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Tempi trascorsi crescenti da zero richiesti.')
    z0=model.encode(initial)
    residual=np.asarray(initial)-model.decode(z0)
    def stop(t,z):
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
    """Errore relativo spaziale medio e RMSE puntuale, senza clipping."""
    diff=np.asarray(prediction)-np.asarray(truth)
    denominator=np.sqrt(np.maximum(np.einsum('ti,ij,tj->t',truth,gram,truth),0))
    if np.any(denominator<=0):
        raise ValueError('Norma nulla nel riferimento.')
    errors=np.sqrt(np.maximum(np.einsum('ti,ij,tj->t',diff,gram,diff),0))/denominator
    return float(errors.mean()),float(np.sqrt(np.mean((diff@sensor_modes)**2)))


def backtest_sindy(coefficients, tau, configurations, cutoffs, gram,
                   sensor_modes, *, derivatives=None):
    """Stessi fold/metriche della DMD, PCA, scale e sparsità fit sui prefissi."""
    rows,folds=[],[]
    for config_id,config in enumerate(configurations):
        errors,point_errors=[],[]
        status='ok'
        for cutoff in cutoffs:
            try:
                fitted=fit_sindy(coefficients[:cutoff+1],tau[:cutoff+1],
                    derivatives=None if derivatives is None else derivatives[:cutoff+1],**config)
                pred=forecast_sindy(fitted,coefficients[cutoff],tau[cutoff:]-tau[cutoff])[1:]
                error,point_error=coefficient_errors(pred,coefficients[cutoff+1:],gram,sensor_modes)
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


def solve_fkpp_reference(Vh, mass, mu, times, *, substeps=4, initial=None, scheme="imex"):
    """Riferimento FEM indipendente: CG1, Neumann naturale, IMEX Euler.

    (M+dt*D*L)u_next=M*u+dt*r*load(u*(1-u)). Il termine di reazione
    è integrato come prodotto FEM, non applicato ai soli nodi. Initial=None
    interpola g_delta ai DOF; initial può essere u_ROM(T) per isolare il
    prolungamento temporale. Non pretende di replicare il solver del dataset.
    Restituisce campi ai times e la durata di sola integrazione in secondi.
    """
    import time
    import dolfin as df
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu
    from dolfin import as_backend_type
    mu=np.asarray(mu,dtype=float)
    times=np.asarray(times,dtype=float)
    if mu.shape!=(3,) or np.any((mu<0)|(mu>1)):
        raise ValueError('Mu normalizzato non valido.')
    if len(times)<2 or np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),np.diff(times)[0]):
        raise ValueError('Griglia temporale uniforme richiesta.')
    if int(substeps)!=substeps or substeps<1:
        raise ValueError('Substeps intero positivo richiesto.')
    D,r,delta=10**mu[0],10**(mu[1]+2),mu[2]+1
    dt=(times[1]-times[0])/substeps
    u,v=df.TrialFunction(Vh),df.TestFunction(Vh)
    stiffness=df.assemble(df.inner(df.grad(u),df.grad(v))*df.dx)
    indptr,indices,values=as_backend_type(stiffness).mat().getValuesCSR()
    L=csr_matrix((values,indices,indptr),shape=mass.shape)
    if scheme == 'imex':
        advance=splu((mass+dt*D*L).tocsc())
    elif scheme == 'linear_implicit':
        if dt*r>=1:
            raise ValueError('La reazione lineare implicita richiede dt*r<1.')
        advance=splu(((1-dt*r)*mass+dt*D*L).tocsc())
    else:
        raise ValueError('Schema imex o linear_implicit richiesto.')
    old=df.Function(Vh)
    if initial is None:
        coords=Vh.tabulate_dof_coordinates().reshape(-1,2)
        state=((abs(coords[:,0]-delta)<.1)&(coords[:,1]>=4)).astype(float)
    else:
        state=np.asarray(initial,dtype=float).copy()
        if state.shape!=(Vh.dim(),) or not np.isfinite(state).all():
            raise ValueError('Stato FEM iniziale non valido.')
    reaction=(old*(1-old) if scheme=='imex' else -old**2)*v*df.dx
    fields=[state.copy()]
    start=time.perf_counter()
    for _ in times[1:]:
        for __ in range(int(substeps)):
            old.vector().set_local(state)
            old.vector().apply('insert')
            load=df.assemble(reaction).get_local()
            state=advance.solve(mass@state+dt*r*load)
        if not np.isfinite(state).all():
            raise FloatingPointError('Il riferimento FEM diverge dallo stato iniziale fornito.')
        fields.append(state.copy())
    return np.stack(fields),time.perf_counter()-start


def solve_fkpp_implicit(Vh, mu, times, *, substeps=1):
    """Euler implicito nonlineare FEM CG1, Neumann naturale; solo diagnostica.

    Integra (u_new-u_old,v)+dt*D*(grad u_new,grad v)
             -dt*r*(u_new*(1-u_new),v)=0 con Newton.
    Condizione iniziale nodale g_delta; restituisce campi e durata.
    """
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
