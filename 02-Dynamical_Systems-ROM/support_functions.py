"""POD-NN utilities for U-bolt damage propagation in the FEM L2 norm.

Arrays use trajectory-major order: fields u=(n_trajectories,n_times,n_dofs),
snapshots S=(n_snapshots,n_dofs), basis V=(n_dofs,n_modes), and coefficients
A=(n_snapshots,n_modes). The mass matrix must be symmetric positive definite
and use exactly the dataset DOF ordering. Importing this module does not run
FEM calculations, downloads, or training. NumPy/SciPy are required at import;
PyTorch, Matplotlib, and dlroms are imported by the functions that need them.
See support_functions_guide.md for the workflow."""
from dataclasses import dataclass
import copy
import random
import time
import numpy as np
from scipy.linalg import eigh


@dataclass
class PODResult:
    """Store the POD basis, singular values, cumulative energy fractions, and total energy."""
    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    total_energy: float


@dataclass
class CoefficientScaler:
    """Store per-mode minima and ranges; constant modes use a range of one."""
    minimum: np.ndarray
    scale: np.ndarray


def build_ubolt_mesh(stepsize=0.05, knots=30):
    """Build the course U-bolt mesh and return (mesh, scalar CG1 space).

    stepsize must be positive and knots must be at least three. Use stepsize=0.05
    and knots=30 for the supplied dataset. Construct the mesh before computing
    POD. Matching DOF counts does not guarantee matching DOF ordering: use the
    dataset mesh pipeline and inspect the reconstructed fields."""
    import dlroms.fespaces as fe
    import os
    import shutil
    from pathlib import Path

    # Selecting a Python environment in VS Code bypasses kernelspec PATH.
    # dlroms and its gmsh executable live in the same installed environment.
    if shutil.which('gmsh') is None:
        gmsh_bin = Path(fe.__file__).resolve().parents[4] / 'bin'
        if (gmsh_bin / 'gmsh').is_file():
            os.environ['PATH'] = str(gmsh_bin) + os.pathsep + os.environ.get('PATH', '')
        else:
            raise RuntimeError('gmsh executable not found: cannot construct the U-bolt mesh')
    if shutil.which('dolfin-convert') is None:
        raise RuntimeError('dolfin-convert executable not found: cannot convert the U-bolt mesh')
    if stepsize <= 0 or knots < 3:
        raise ValueError('stepsize>0 e knots>=3 richiesti')
    points = [np.array([0., 0.])]
    def step(z):
        """Append a boundary vertex using the real and imaginary parts of displacement z."""
        points.append(points[-1] + [z.real, z.imag])
    def arch(r, thetas):
        """Append a circular boundary arc with complex radius r and angular limits thetas."""
        center = points[-1] + [r.real, r.imag]
        shift = abs(r)*np.exp(1j*np.linspace(*thetas, knots)[1:])
        points.extend(np.stack([center[0]+shift.real, center[1]+shift.imag], axis=1))
    step(3j)
    arch(2, (np.pi, 0))
    step(-3j)
    step(-1)
    step(3j)
    arch(-1, (0, np.pi))
    step(-3j)
    step(-1)
    mesh = fe.mesh(fe.polygon(np.stack(points)), stepsize=stepsize)
    return mesh, fe.space(mesh, 'CG', 1)


def assemble_mass_matrix(Vh):
    """Assemble M[i,j] = integral(phi_i * phi_j) on the scalar FEM space Vh.

    Return the representation provided by dlroms. Numerical routines expect a
    NumPy array or a SciPy sparse matrix with the dataset DOF ordering."""
    import dlroms.fespaces as fe
    return fe.assemble(lambda u, v: fe.inner(u, v)*fe.dx, Vh)


def check_fem_conversion(values, Vh):
    """Fail if the NumPy-to-FEniCS conversion silently changes field values."""
    import dlroms.fespaces as fe
    expected = np.ascontiguousarray(values, dtype=np.float64)
    if expected.shape != (Vh.dim(),):
        raise ValueError('Expected one field with Vh.dim() DOFs')
    actual = fe.asfunction(expected, Vh).vector().get_local()
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            'NumPy-to-FEniCS conversion changed the field values. '
            'Select the U-bolt (NumPy 1.26) kernel and restart it.'
        )


def modify_jet_to_gray(jet_colors=None):
    """Return a ListedColormap with the first 85 jet colors replaced by gray.

    jet_colors optionally supplies an RGBA array of shape (256,4). Pass the
    returned colormap directly to plotting functions; no global registration
    of the name gray_jet is required."""
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    colors = (mpl.colormaps['jet'](np.linspace(0, 1, 256))
              if jet_colors is None else np.array(jet_colors, copy=True))
    if colors.shape != (256, 4):
        raise ValueError('jet_colors deve avere shape (256,4)')
    colors[:85, :3] = np.linspace(.3, .7, 85)[:, None]
    return ListedColormap(colors, name='gray_jet')


def build_damage_evaluator(mesh, Vh, x0=(0.5, 3.0), device='cpu'):
    """Build a PyTorch FEM evaluation operator for one or more spatial points.

    x0 is a point (2,) or an array (n_points,2). The returned operator maps
    fields of shape (n_times,n_dofs) to (n_times,n_points). Input tensors must
    already be on the requested device; this function does not convert them."""
    from dlroms.minns import Evaluate
    evaluator = Evaluate(mesh, Vh, np.asarray(x0, dtype=float).reshape(-1, 2))
    evaluator.to(device)
    return evaluator


def _fields(fields):
    """Convert fields to a finite float64 array with at least one axis."""
    x = np.asarray(fields, dtype=np.float64)
    if x.ndim < 1 or not np.isfinite(x).all():
        raise ValueError('Campi finiti con almeno un asse richiesti')
    return x


def _mass_apply(M, fields):
    """Apply M along the last field axis without converting a sparse M to dense."""
    x = _fields(fields)
    if M.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError('M e numero DOF incompatibili')
    return np.asarray(M @ x.reshape(-1, x.shape[-1]).T).T.reshape(x.shape)


def compute_l2_pod(snapshots, M, max_modes=100, rtol=1e-12):
    """Compute an uncentered mass-weighted POD using C = S @ M @ S.T.

    snapshots is a training-only array S of shape (n_snapshots,n_dofs); M is a
    dense or sparse symmetric positive definite mass matrix. max_modes limits
    the number of modes; rtol discards eigenvalues below rtol times the largest.
    Return PODResult with V.T @ M @ V = I and total_energy = trace(C), including
    energy outside the extracted modes. No mean subtraction or time weighting
    is applied. The correlation matrix requires O(n_snapshots**2) memory, and
    very small modes may lose numerical accuracy. Compute POD once, before tuning."""
    S = _fields(snapshots)
    if S.ndim != 2 or not S.size or max_modes < 1 or not 0 < rtol < 1:
        raise ValueError('S 2D non vuota, max_modes>=1, 0<rtol<1 richiesti')
    C = S @ _mass_apply(M, S).T
    C = (C+C.T)/2
    energy = float(np.trace(C))
    if energy <= 0:
        raise ValueError('Energia training nulla o M non positiva')
    k = min(int(max_modes), *S.shape)
    values, vectors = eigh(C, subset_by_index=(len(S)-k, len(S)-1))
    values, vectors = values[::-1], vectors[:, ::-1]
    keep = values > rtol*values[0]
    values, vectors = values[keep], vectors[:, keep]
    singular = np.sqrt(values)
    V = (S.T @ vectors)/singular
    return PODResult(V, singular, np.cumsum(values)/energy, energy)


def project_l2(snapshots, basis, M):
    """Project fields onto an M-orthonormal basis using A = S @ M @ V.

    Map (...,n_dofs) to (...,n_modes). Reuse the training basis for validation
    and test data rather than fitting a new POD."""
    return _mass_apply(M, snapshots) @ np.asarray(basis)


def reconstruct_pod(coefficients, basis):
    """Reconstruct fields as A @ V.T, mapping (...,n_modes) to (...,n_dofs)."""
    return np.asarray(coefficients) @ np.asarray(basis).T


def make_nn_inputs(mu, time_grid):
    """Return trajectory-major network inputs with normalized time t/T.

    mu has shape (n_trajectories,n_parameters). time_grid is strictly increasing,
    starts at zero, and contains at least two times. Return an array of shape
    (n_trajectories*n_times,n_parameters+1), consistent with u.reshape(-1,n_dofs).
    Physical parameters are not rescaled; the U-bolt parameters are already in [0,1]."""
    mu, t = _fields(mu), _fields(time_grid)
    if mu.ndim != 2 or t.ndim != 1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0):
        raise ValueError('mu 2D, t crescente da zero con almeno due punti')
    return np.column_stack([np.repeat(mu, len(t), axis=0), np.tile(t/t[-1], len(mu))])


def fit_coefficient_scaler(A_train):
    """Fit per-mode min-max scaling on training coefficients only.

    A_train has shape (n_snapshots,n_modes). Return CoefficientScaler, replacing
    exactly zero ranges with one. Validation/test values may lie outside [0,1]
    and are not clipped."""
    A = _fields(A_train)
    if A.ndim != 2 or not A.size:
        raise ValueError('A_train 2D non vuoto richiesto')
    minimum, scale = A.min(axis=0), np.ptp(A, axis=0)
    return CoefficientScaler(minimum, np.where(scale==0, 1., scale))


def _scaler_arrays(A, scaler):
    """Return scaling arrays compatible with the NumPy input or PyTorch dtype/device."""
    if hasattr(A, 'detach') and hasattr(A, 'device'):
        import torch
        return (torch.as_tensor(scaler.minimum, dtype=A.dtype, device=A.device),
                torch.as_tensor(scaler.scale, dtype=A.dtype, device=A.device))
    return scaler.minimum, scaler.scale


def scale_coefficients(A, scaler):
    """Apply (A-minimum)/scale, preserving PyTorch device and gradients."""
    minimum, scale = _scaler_arrays(A, scaler)
    return (A-minimum)/scale


def unscale_coefficients(A_scaled, scaler):
    """Apply A_scaled*scale+minimum, preserving PyTorch device and gradients."""
    minimum, scale = _scaler_arrays(A_scaled, scaler)
    return A_scaled*scale+minimum


def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and CUDA without guaranteeing bitwise determinism."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(input_dim, output_dim, hidden_layers=(128,)*5,
                activation='silu', device='cpu'):
    """Build a PyTorch MLP with a linear, unclipped coefficient output.

    input_dim is usually n_parameters+1 and output_dim is the POD dimension.
    hidden_layers gives positive layer widths. activation is leaky_relu (slope
    0.1), tanh, silu, or relu. Return the network on the requested device."""
    import torch.nn as nn
    choices = {'leaky_relu': lambda: nn.LeakyReLU(.1), 'tanh': nn.Tanh,
               'silu': nn.SiLU, 'relu': nn.ReLU}
    if activation not in choices or min(input_dim, output_dim, *hidden_layers)<1:
        raise ValueError('Attivazione o dimensioni non valide')
    layers, width_in = [], input_dim
    for width in hidden_layers:
        layers.extend([nn.Linear(width_in, width), choices[activation]()])
        width_in = width
    layers.append(nn.Linear(width_in, output_dim))
    return nn.Sequential(*layers).to(device)


def count_parameters(model):
    """Return the number of PyTorch parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def coefficient_loss(pred_scaled, target, scaler):
    """Return MSE between unscaled predictions and unscaled target tensors.

    For an M-orthonormal POD basis, this is the squared spatial L2 regression
    error averaged over samples and modes. It is not mean relative L2 error."""
    return ((unscale_coefficients(pred_scaled, scaler)-target)**2).mean()


def train_model(X_train, A_train, X_val, A_val, scaler, *,
                hidden_layers=(128,)*5, activation='silu', lr=1e-3,
                weight_decay=1e-4, max_epochs=1500, batch_size=64,
                device='cpu', seed=42, scheduler_patience=100,
                scheduler_factor=0.5, log_every=250):
    """Train a new POD-NN using 2D NumPy inputs and unscaled coefficients.

    Create float32 tensors, a shuffled training DataLoader, Adam, and a
    ReduceLROnPlateau scheduler driven by validation coefficient_loss. Run all
    max_epochs without early stopping, then restore the lowest-validation-loss
    weights. Seed initialization precedes network and loader creation. The scaler
    must already be fitted on training data; test data and FOM fields are unused.

    Return a dictionary containing model, train_history, val_history,
    best_val_loss, best_epoch (one-based), and config. Training losses are
    averaged during updates; validation loss is measured at each epoch end."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    arrays = [np.asarray(x) for x in (X_train, A_train, X_val, A_val)]
    X_train, A_train, X_val, A_val = arrays
    if any(x.ndim!=2 or not len(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Input e target devono essere matrici finite non vuote')
    if (len(X_train)!=len(A_train) or len(X_val)!=len(A_val)
        or X_train.shape[1]!=X_val.shape[1] or A_train.shape[1]!=A_val.shape[1]
        or len(scaler.minimum)!=A_train.shape[1]):
        raise ValueError('Dimensioni dati/scaler incompatibili')
    if max_epochs<1 or batch_size<1 or lr<=0 or weight_decay<0:
        raise ValueError('Iperparametri non validi')
    set_seed(seed)
    model = build_model(X_train.shape[1], A_train.shape[1], hidden_layers, activation, device)
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in arrays]
    loader = DataLoader(TensorDataset(tensors[0], tensors[1]), batch_size=batch_size, shuffle=True)
    xv, av = tensors[2].to(device), tensors[3].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience)
    train_history, val_history = [], []
    best_loss, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, max_epochs+1):
        model.train()
        total = 0.
        for xb, ab in loader:
            xb, ab = xb.to(device), ab.to(device)
            optimizer.zero_grad()
            loss = coefficient_loss(model(xb), ab, scaler)
            if not torch.isfinite(loss):
                raise FloatingPointError('Loss training non finita')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = coefficient_loss(model(xv), av, scaler).item()
        if not np.isfinite(val_loss):
            raise FloatingPointError('Loss validation non finita')
        scheduler.step(val_loss)
        train_history.append(total/len(X_train))
        val_history.append(val_loss)
        if val_loss<best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if log_every and epoch%log_every==0:
            print(f'Epoch {epoch}: train={train_history[-1]:.6g}, val={val_loss:.6g}')
    model.load_state_dict(best_state)
    model.eval()
    return dict(model=model, train_history=train_history, val_history=val_history,
                best_val_loss=best_loss, best_epoch=best_epoch,
                config=dict(hidden_layers=list(hidden_layers), activation=activation,
                            lr=lr, weight_decay=weight_decay, seed=seed))


def predict_coefficients(model, X, scaler, batch_size=4096):
    """Predict unscaled coefficients in batches without tracking gradients.

    X is a NumPy array or tensor of shape (n_samples,input_dim). Infer device and
    dtype from the model and return a NumPy array (n_samples,n_modes). Restore
    the model's previous training/evaluation state after inference."""
    import torch
    if batch_size<1 or len(X)==0:
        raise ValueError('batch_size positivo e X non vuoto richiesti')
    parameter = next(model.parameters())
    x = torch.as_tensor(X, dtype=parameter.dtype, device=parameter.device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return np.concatenate([unscale_coefficients(model(xb), scaler).cpu().numpy()
                                   for xb in x.split(batch_size)], axis=0)
    finally:
        model.train(was_training)


def predict_solution(model, X, scaler, basis, batch_size=4096):
    """Return reconstructed NumPy fields (n_samples,n_dofs) for flattened inputs X."""
    return reconstruct_pod(predict_coefficients(model, X, scaler, batch_size), basis)


def l2_norm(fields, M):
    """Compute sqrt(u.T @ M @ u) along the last axis: (...,n_dofs) -> (...).

    Clip small negative roundoff values to zero. Substantially negative squared
    norms raise ValueError because they indicate invalid fields or a mass matrix."""
    x = _fields(fields)
    squares = np.sum(x*_mass_apply(M, x), axis=-1)
    if np.any(squares < -1e-12*max(1., float(np.max(np.abs(squares))))):
        raise ValueError('Norma quadratica negativa: verificare M')
    return np.sqrt(np.maximum(squares, 0.))


def relative_l2_error(u_true, u_pred, M, zero_policy='raise', zero_tol=0.):
    """Return ||u_true-u_pred||_M / ||u_true||_M as fractions, not percentages.

    Fields must have identical shape (...,n_dofs). For reference norms at or below
    zero_tol, zero_policy='raise' fails and 'nan' returns NaN for exclusion by
    summarize_errors. The denominator is not replaced with an epsilon."""
    true, pred = _fields(u_true), _fields(u_pred)
    if true.shape!=pred.shape or zero_policy not in ('raise', 'nan') or zero_tol<0:
        raise ValueError('Shape, zero_policy o zero_tol non validi')
    denominator = l2_norm(true, M)
    zero = denominator<=zero_tol
    if np.any(zero) and zero_policy=='raise':
        raise ValueError('Norma FOM nulla: usare zero_policy="nan" o errore assoluto')
    return np.divide(l2_norm(true-pred, M), denominator,
                     out=np.full(denominator.shape, np.nan), where=~zero)


def summarize_errors(errors):
    """Summarize fractional errors of shape (n_trajectories,n_times).

    Average over valid times within each trajectory, then over trajectories.
    Ignore NaN snapshots but fail if an entire trajectory is undefined. This is
    an arithmetic time average, not a temporal L2 norm. Return mean_error,
    median_error, worst_trajectory_error, max_snapshot_error, trajectory_errors,
    best_index, representative_index (closest to the mean), worst_index, and
    excluded_snapshots."""
    e = np.asarray(errors)
    if e.ndim!=2 or not e.size or np.isinf(e).any() or np.any(e<0) or np.any(np.all(np.isnan(e), axis=1)):
        raise ValueError('Errori 2D validi con almeno un valore per traiettoria')
    trajectory = np.nanmean(e, axis=1)
    mean = float(trajectory.mean())
    median = float(np.median(trajectory))
    return dict(mean_error=mean, median_error=median, worst_trajectory_error=float(trajectory.max()),
                max_snapshot_error=float(np.nanmax(e)), trajectory_errors=trajectory,
                best_index=int(trajectory.argmin()), worst_index=int(trajectory.argmax()),
                representative_index=int(np.argmin(abs(trajectory-mean))),
                excluded_snapshots=int(np.isnan(e).sum()))


def evaluate_model_L2(model, X, u_true, basis, M, scaler, *, zero_policy='raise'):
    """Predict fields and summarize their relative spatial L2 errors.

    u_true has shape (n_trajectories,n_times,n_dofs); X must follow
    make_nn_inputs ordering. Return summarize_errors entries plus errors of
    shape (n_trajectories,n_times) and prediction with the same shape as u_true."""
    u = _fields(u_true)
    if u.ndim!=3 or len(X)!=u.shape[0]*u.shape[1]:
        raise ValueError('u_true 3D e numero input coerente richiesti')
    prediction = predict_solution(model, X, scaler, basis).reshape(u.shape)
    errors = relative_l2_error(u, prediction, M, zero_policy)
    return dict(**summarize_errors(errors), errors=errors, prediction=prediction)


def run_experiments(configurations, X_train, A_train, X_val, A_val,
                    scaler, u_val, basis, M, *, device='cpu', zero_policy='raise'):
    """Train named configurations and evaluate them on validation data.

    configurations maps names to train_model keyword arguments. Return a mapping
    from each name to its training result plus evaluation and parameter count.
    No winner is selected automatically. Results retain models and reconstructed
    fields, so use small configuration batches to control memory usage."""
    results = {}
    for name, config in configurations.items():
        print(f'Experiment: {name}')
        options = dict(config)
        options.setdefault('device', device)
        trained = train_model(X_train, A_train, X_val, A_val, scaler, **options)
        trained['evaluation'] = evaluate_model_L2(
            trained['model'], X_val, u_val, basis, M, scaler, zero_policy=zero_policy)
        trained['parameters'] = count_parameters(trained['model'])
        results[name] = trained
    return results


def plot_training_history(train_history, val_history, ax=None):
    """Plot unscaled coefficient MSE on a logarithmic y-axis; return ax without showing."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1,len(train_history)+1), train_history, label='Train')
    ax.semilogy(np.arange(1,len(val_history)+1), val_history, label='Validation')
    ax.set(xlabel='Epoch', ylabel='MSE on unscaled POD coefficients')
    ax.grid(alpha=.3)
    ax.legend()
    return ax


def plot_error_evolution(time_grid, errors, ax=None):
    """Plot relative spatial errors against physical time and return ax.

    errors maps labels to fractional arrays of shape (n_times,). Values are
    converted to percentages for display. This function does not call show()."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, values in errors.items():
        ax.plot(time_grid, 100*np.asarray(values), label=name)
    ax.set(xlabel='Time', ylabel='Relative spatial L2 error [%]')
    ax.grid(alpha=.3)
    ax.legend()
    return ax


def gif_compare(name, u_fom, u_rom, Vh, time_grid, *, figsize=(15,4),
                titles=('FOM','POD-NN','Absolute error'),
                error_scale='fixed', max_frames=None):
    """Save a GIF of FOM/ROM fields and their pointwise absolute difference.

    Both fields have shape (n_times,n_dofs); time_grid contains physical times.
    FOM and ROM share fixed color limits. Error limits are fixed by default,
    or vary per frame with error_scale='dynamic'. max_frames optionally
    subsamples frames; linspace includes the final time when at least two
    frames are selected. Playback timing follows the course GIF writer, not
    physical simulation time. Return name after saving through fe.gifs.save."""
    import matplotlib.pyplot as plt
    import dlroms.fespaces as fe
    from dlroms.fespaces import gifs
    true, pred = _fields(u_fom), _fields(u_rom)
    t = np.asarray(time_grid)
    if true.ndim!=2 or true.shape!=pred.shape or len(t)!=len(true) or len(t)<1:
        raise ValueError('Campi (Nt,Nh) e tempi coerenti richiesti')
    if len(titles)!=3 or error_scale not in ('fixed','dynamic'):
        raise ValueError('Tre titoli e scala fixed/dynamic richiesti')
    if max_frames is not None and max_frames<1:
        raise ValueError('max_frames deve essere positivo')
    n = len(t) if max_frames is None else min(len(t), int(max_frames))
    indices = np.unique(np.linspace(0,len(t)-1,n,dtype=int))
    error = abs(true-pred)
    lower, upper = min(true.min(),pred.min()), max(true.max(),pred.max())
    upper = max(upper, lower+1e-12)
    cmap = modify_jet_to_gray()
    def drawframe(i):
        """Draw one selected FOM/ROM/error frame for the GIF writer."""
        j = indices[i]
        plt.figure(figsize=figsize)
        for panel, field in enumerate((true, pred, error)):
            plt.subplot(1,3,panel+1)
            if panel<2:
                fe.plot(field[j], Vh, vmin=lower, vmax=upper, cmap=cmap)
            else:
                maximum = error.max() if error_scale=='fixed' else error[j].max()
                fe.plot(field[j], Vh, vmin=0., vmax=max(float(maximum),1e-12), cmap='magma')
            plt.title(f'{titles[panel]}\nt = {t[j]:.5f}')
        plt.tight_layout()
    gifs.save(drawframe, len(indices), str(name))
    return name


def benchmark_rom(model, X_trajectory, scaler, basis, *, n_repeats=100,
                  warmup=20, fom_time=None):
    """Measure prepared-input inference time in seconds, including reconstruction.

    Include network evaluation, coefficient unscaling, CPU transfer, and full-field
    reconstruction; exclude input preparation, loading, and training. Synchronize
    CUDA around each measurement. Return times, mean, median, std, and optional
    speedup=fom_time/mean. The reference fom_time must cover the same trajectory
    and time interval on comparable hardware."""
    import torch
    if n_repeats<1 or warmup<0 or (fom_time is not None and fom_time<=0):
        raise ValueError('Parametri benchmark non validi')
    parameter = next(model.parameters())
    X = torch.as_tensor(X_trajectory, dtype=parameter.dtype, device=parameter.device)
    def synchronize():
        """Wait for pending CUDA operations when the model uses a CUDA device."""
        if parameter.device.type=='cuda':
            torch.cuda.synchronize(parameter.device)
    for _ in range(warmup):
        predict_solution(model, X, scaler, basis)
    elapsed = []
    for _ in range(n_repeats):
        synchronize()
        start = time.perf_counter()
        predict_solution(model, X, scaler, basis)
        synchronize()
        elapsed.append(time.perf_counter()-start)
    elapsed = np.asarray(elapsed)
    result = dict(times=elapsed, mean=float(elapsed.mean()), median=float(np.median(elapsed)),
                  std=float(elapsed.std()))
    if fom_time is not None:
        result['speedup'] = fom_time/result['mean']
    return result


def plot_parameter_splits(mu_train, mu_val, mu_test):
    """Plot all parameter pairs for training, validation, and test trajectories.

    Each input has shape (n_trajectories,n_parameters) with matching column order.
    Axis limits follow the data. Return (fig, axes) without calling show()."""
    from itertools import combinations
    import numpy as np
    import matplotlib.pyplot as plt

    splits = {
        "Train": np.asarray(mu_train),
        "Validation": np.asarray(mu_val),
        "Test": np.asarray(mu_test),
    }

    arrays = list(splits.values())
    if any(a.ndim != 2 or len(a) == 0 for a in arrays):
        raise ValueError("Ogni split deve essere una matrice 2D non vuota.")

    n_parameters = arrays[0].shape[1]
    if any(a.shape[1] != n_parameters for a in arrays):
        raise ValueError("Gli split devono avere lo stesso numero di parametri.")
    if n_parameters < 2:
        raise ValueError("Servono almeno due parametri.")

    pairs = list(combinations(range(n_parameters), 2))
    fig, axes = plt.subplots(
        1, len(pairs),
        figsize=(5 * len(pairs), 4),
        squeeze=False,
    )
    axes = axes.ravel()

    styles = {
        "Train": dict(c="tab:blue", marker="o", s=30, alpha=0.7),
        "Validation": dict(c="tab:orange", marker="s", s=40, alpha=0.7),
        "Test": dict(c="tab:green", marker="x", s=60, alpha=0.9),
    }

    for ax, (i, j) in zip(axes, pairs):
        for name, mu in splits.items():
            ax.scatter(mu[:, i], mu[:, j], label=name, **styles[name])

        ax.set_xlabel(rf"$\mu_{i + 1}$")
        ax.set_ylabel(rf"$\mu_{j + 1}$")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.suptitle("Train/Validation/Test distribution in parameter space")
    fig.tight_layout()
    return fig, axes


def summarize_parameter_splits(mu_train, mu_val, mu_test):
    """Return a pandas table of min, max, mean, and std per split and parameter.

    Inputs are arrays (n_trajectories,n_parameters) with matching columns.
    Standard deviations use ddof=0. No input modification or printing is performed."""
    import numpy as np
    import pandas as pd

    splits = {
        "Train": np.asarray(mu_train),
        "Validation": np.asarray(mu_val),
        "Test": np.asarray(mu_test),
    }

    arrays = list(splits.values())
    if any(a.ndim != 2 or len(a) == 0 for a in arrays):
        raise ValueError("Ogni split deve essere una matrice 2D non vuota.")
    if any(a.shape[1] != arrays[0].shape[1] for a in arrays):
        raise ValueError("Gli split devono avere lo stesso numero di parametri.")

    rows = []
    for name, mu in splits.items():
        for i in range(mu.shape[1]):
            values = mu[:, i]
            rows.append({
                "split": name,
                "parameter": f"mu_{i + 1}",
                "n_trajectories": len(mu),
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(ddof=0),
            })

    return pd.DataFrame(rows).set_index(["parameter", "split"])


def diagnose_parameter_splits(mu_train, mu_val, mu_test):

    """Measure marginal balance and joint parameter coverage across splits.

    Inputs are finite arrays (n_trajectories,n_parameters) with matching columns.
    Parameters must use comparable scales for distances (U-bolt uses [0,1]).
    Return (balance, coverage), two pandas DataFrames. balance contains min, max,
    mean, std, absolute mean shift divided by training std, and std/training std.
    coverage contains trajectory counts, median/90th-percentile/maximum nearest
    training distances, and percentages inside the training convex hull. Training
    distances exclude each point itself. Failed hull construction gives NaN
    coverage and a hull_status explanation. No thresholds or split changes are
    applied automatically."""
    import numpy as np
    import pandas as pd
    from scipy.spatial import Delaunay, QhullError
    from scipy.spatial.distance import cdist

    splits = {
        "Train": np.asarray(mu_train, dtype=float),
        "Validation": np.asarray(mu_val, dtype=float),
        "Test": np.asarray(mu_test, dtype=float),
    }

    arrays = list(splits.values())

    if any(
        a.ndim != 2 or len(a) == 0 or not np.isfinite(a).all()
        for a in arrays
    ):
        raise ValueError("Gli split devono essere matrici finite non vuote.")

    n_parameters = arrays[0].shape[1]

    if n_parameters == 0 or any(
        a.shape[1] != n_parameters for a in arrays
    ):
        raise ValueError("Numero di parametri non valido o diverso tra split.")

    train = splits["Train"]

    if len(train) < 2:
        raise ValueError("Servono almeno due traiettorie di training.")

    # ------------------------------------------------------------
    # 1. Bilanciamento marginale
    # ------------------------------------------------------------
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0, ddof=0)

    rows = []

    for name, parameters in splits.items():
        means = parameters.mean(axis=0)
        stds = parameters.std(axis=0, ddof=0)

        for j in range(n_parameters):
            reference_std = train_std[j]

            rows.append({
                "parameter": f"mu_{j + 1}",
                "split": name,
                "min": parameters[:, j].min(),
                "max": parameters[:, j].max(),
                "mean": means[j],
                "std": stds[j],
                "mean_shift_std": (
                    abs(means[j] - train_mean[j]) / reference_std
                    if reference_std > 0 else np.nan
                ),
                "std_ratio": (
                    stds[j] / reference_std
                    if reference_std > 0 else np.nan
                ),
            })

    balance = pd.DataFrame(rows).set_index(["parameter", "split"])

    # ------------------------------------------------------------
    # 2. Copertura: nearest neighbour
    # ------------------------------------------------------------
    train_distances = cdist(train, train)
    np.fill_diagonal(train_distances, np.inf)

    nearest_distances = {
        "Train": train_distances.min(axis=1),
        "Validation": cdist(splits["Validation"], train).min(axis=1),
        "Test": cdist(splits["Test"], train).min(axis=1),
    }

    # ------------------------------------------------------------
    # 3. Inviluppo convesso del training
    # ------------------------------------------------------------
    hull = None
    hull_status = "ok"

    if n_parameters > 1:
        try:
            hull = Delaunay(train)
        except QhullError:
            hull_status = "non costruibile: training insufficiente o degenere"

    coverage_rows = []

    for name, parameters in splits.items():
        distances = nearest_distances[name]

        if n_parameters == 1:
            # In una dimensione l'inviluppo è semplicemente un intervallo.
            inside = (
                (parameters[:, 0] >= train[:, 0].min())
                & (parameters[:, 0] <= train[:, 0].max())
            )
            inside_percent = 100 * inside.mean()

        elif hull is not None:
            inside = hull.find_simplex(parameters, tol=1e-10) >= 0
            inside_percent = 100 * inside.mean()

        else:
            inside_percent = np.nan

        coverage_rows.append({
            "split": name,
            "n_trajectories": len(parameters),
            "nearest_train_median": np.median(distances),
            "nearest_train_p90": np.quantile(distances, 0.90),
            "nearest_train_max": distances.max(),
            "inside_train_hull_percent": inside_percent,
            "hull_status": hull_status,
        })

    coverage = pd.DataFrame(coverage_rows).set_index("split")

    return balance, coverage


def plot_pod_analysis(
    s,
    total_energy,
    thresholds=(0.90, 0.95, 0.99, 0.999, 0.9999),
):
    """Plot singular-value decay and cumulative energy; print threshold mode counts.

    s contains retained singular values in descending order. total_energy is
    the full snapshot energy, including omitted modes. Return (cumulative_energy,
    threshold_modes), where the dictionary contains only thresholds reached by
    the supplied modes. This function calls plt.show()."""
    import numpy as np
    import matplotlib.pyplot as plt

    s = np.asarray(s)

    if not np.isfinite(total_energy) or total_energy <= 0:
        raise ValueError("total_energy deve essere finita e positiva")

    modes = np.arange(1, len(s) + 1)
    cumulative_energy = np.cumsum(s**2) / total_energy


    # Find number of modes required for each threshold
    threshold_modes = {}

    print("POD cumulative energy thresholds:")
    for threshold in thresholds:
        if cumulative_energy[-1] >= threshold:
            N = np.searchsorted(cumulative_energy, threshold) + 1
            threshold_modes[threshold] = N
            print(f"  {threshold*100:6.2f}% -> {N:3d} modes")
        else:
            print(f"  {threshold*100:6.2f}% -> not reached")

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Singular values decay ---
    axes[0].semilogy(
        modes,
        s / s[0],
        'o-',
        markersize=3
    )

    for (threshold, N), color in zip(threshold_modes.items(), colors):
        axes[0].axvline(
            N,
            color=color,
            linestyle='--',
            alpha=0.8,
            label=f"{threshold*100:.2f}% → N={N}"
        )

    axes[0].set_xlabel("POD mode")
    axes[0].set_ylabel(r"$\sigma_i / \sigma_1$")
    axes[0].set_title(r"$L^2$ POD singular value decay")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # --- Cumulative energy ---
    axes[1].plot(
        modes,
        100 * cumulative_energy,
        'o-',
        markersize=3
    )

    for (threshold, N), color in zip(threshold_modes.items(), colors):
        axes[1].axvline(
            N,
            color=color,
            linestyle='--',
            alpha=0.8,
            label=f"{threshold*100:.2f}% → N={N}"
        )

        axes[1].axhline(
            100 * threshold,
            color=color,
            linestyle=':',
            alpha=0.5
        )

    axes[1].set_xlabel("Number of POD modes")
    axes[1].set_ylabel("Cumulative energy [%]")
    axes[1].set_title(r"$L^2$ POD cumulative energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    return cumulative_energy, threshold_modes

def run_cached_experiment(config, X_train, A_train, X_val, A_val, scaler,
                          u_val, basis, M, *, model_dir='models', device='cpu',
                          force=False, cache_version='v1', zero_policy='raise'):
    """Train or reload a POD-NN experiment, with validation L2 metrics.

    Cache key includes complete training options, data, basis, mass matrix,
    scaler and explicit cache_version. Only use checkpoints you trust.
    Returns train_model-style dict plus model, evaluation, metrics, parameters,
    experiment_id, checkpoint_path, cached. Validation only; never test.
    """
    import hashlib
    import json
    from pathlib import Path
    import torch
    from scipy import sparse

    def fingerprint(value):
        """Update the experiment hash with array contents, sparse structure, or scalar metadata."""
        h = hashlib.sha256()
        if sparse.issparse(value):
            a = value.tocsr(copy=True)
            a.sum_duplicates()
            a.sort_indices()
            parts = (np.asarray(a.shape, dtype=np.int64), a.indptr, a.indices, a.data)
        else:
            parts = (np.asarray(value),)
        for part in parts:
            x = np.ascontiguousarray(part)
            h.update(str(x.shape).encode())
            h.update(x.dtype.str.encode())
            h.update(x.tobytes())
        return h.hexdigest()

    defaults = dict(hidden_layers=(128,)*5, activation='silu', lr=1e-3,
                    weight_decay=1e-4, max_epochs=1500, batch_size=64,
                    seed=42, scheduler_patience=100, scheduler_factor=0.5,
                    log_every=250)
    options = dict(defaults)
    options.update(config)
    options.pop('device', None)
    options['hidden_layers'] = tuple(options['hidden_layers'])
    if set(options) != set(defaults):
        raise ValueError(f'Unsupported configuration keys: {set(options)-set(defaults)}')
    if len(basis.shape) != 2 or basis.shape[1] != A_train.shape[1]:
        raise ValueError('POD basis and coefficient dimensions are inconsistent')

    identity = dict(version=cache_version, options=options,
                    hashes={name: fingerprint(value) for name, value in dict(
                        X_train=X_train, A_train=A_train, X_val=X_val, A_val=A_val,
                        u_val=u_val, basis=basis, M=M,
                        scaler_min=scaler.minimum, scaler_scale=scaler.scale).items()},
                    zero_policy=zero_policy)
    signature = json.dumps(identity, sort_keys=True)
    experiment_id = hashlib.sha256(signature.encode()).hexdigest()[:16]
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    layers = options['hidden_layers']

    architecture = (
        f"W{layers[0]}_D{len(layers)}"
        if len(set(layers)) == 1
        else "H" + "-".join(map(str, layers))
    )

    filename = (
        f"podnn_N{basis.shape[1]}_"
        f"{architecture}_"
        f"{options['activation']}_"
        f"lr{options['lr']:.0e}_"
        f"wd{options['weight_decay']:.0e}_"
        f"bs{options['batch_size']}_"
        f"ep{options['max_epochs']}_"
        f"s{options['seed']}_"
        f"{experiment_id}.pt"
    )

    path = folder / filename
    cached = path.exists() and not force

    if cached:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != json.loads(signature):
            raise ValueError('Checkpoint identity mismatch')
        model = build_model(X_train.shape[1], A_train.shape[1],
                            options['hidden_layers'], options['activation'], device)
        model.load_state_dict(saved['state_dict'])
        model.eval()
        result = saved['training']
        metrics = saved['metrics']
        print('Training already exists — loading checkpoint')
    else:
        print(f'Training new experiment: {experiment_id}')
        trained = train_model(X_train, A_train, X_val, A_val, scaler,
                              device=device, **options)
        model = trained.pop('model')
        result = trained
        evaluation = evaluate_model_L2(model, X_val, u_val, basis, M, scaler,
                                       zero_policy=zero_policy)
        metrics = {k: float(evaluation[k]) for k in
                   ('mean_error', 'worst_trajectory_error', 'max_snapshot_error')}
        torch.save(dict(identity=json.loads(signature),
                        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        training=result, metrics=metrics), path)
        print(f'Checkpoint saved: {path}')

    print(f"Parameters: {count_parameters(model):,} | Best epoch: {result['best_epoch']}")
    print(f"Validation coefficient MSE: {result['best_val_loss']:.6e}")
    print(f"Mean relative L2: {100*metrics['mean_error']:.3f}% | "
          f"Worst trajectory: {100*metrics['worst_trajectory_error']:.3f}% | "
          f"Max snapshot: {100*metrics['max_snapshot_error']:.3f}%")
    return dict(**result, model=model, metrics=metrics,
                parameters=count_parameters(model), experiment_id=experiment_id,
                checkpoint_path=str(path), cached=cached)

