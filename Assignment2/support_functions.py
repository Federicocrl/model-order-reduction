"""Supporto POD-NN per u-bolt: POD e metriche nella norma L2 FEM.

Convenzioni: u=(n_trajectories,n_times,n_dofs), S=(n_snapshots,n_dofs),
V=(n_dofs,n_modes), A=(n_snapshots,n_modes). Ordine trajectory-major.
M deve essere simmetrica definita positiva e avere lo stesso ordinamento DOF
 dei dati. Nessuna operazione FEM, download o training viene eseguita all'import.
NumPy/SciPy sono obbligatori; torch, matplotlib e dlroms sono importati solo
nelle funzioni che li utilizzano. Vedere support_functions_guide.md.
"""
from dataclasses import dataclass
import copy
import random
import time
import numpy as np
from scipy.linalg import eigh


@dataclass
class PODResult:
    """Risultato POD: basis, singular_values, cumulative_energy, total_energy."""
    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    total_energy: float


@dataclass
class CoefficientScaler:
    """minimum e scale per modo; scale=1 per coefficienti costanti."""
    minimum: np.ndarray
    scale: np.ndarray


def build_ubolt_mesh(stepsize=0.05, knots=30):
    """Geometria dal notebook del corso, incapsulata senza variabili globali.

    Input: stepsize positivo, knots>=3 punti per arco. Per il dataset fornito
    mantenere stepsize=0.05 e knots=30. Output: mesh, Vh (CG1 scalare).
    Serve PRIMA della POD. Stessa dimensione DOF non garantisce stesso ordine:
    utilizzare la stessa pipeline di mesh del dataset e verificarne i plot.
    """
    import dlroms.fespaces as fe
    if stepsize <= 0 or knots < 3:
        raise ValueError('stepsize>0 e knots>=3 richiesti')
    points = [np.array([0., 0.])]
    def step(z):
        points.append(points[-1] + [z.real, z.imag])
    def arch(r, thetas):
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
    """Restituisce M_ij=integrale phi_i phi_j tramite API fe.assemble del corso.

    Vh: spazio FEM scalare. Mantiene la rappresentazione restituita da dlroms;
    nelle routine numeriche M deve essere array NumPy o matrice sparse SciPy.
    """
    import dlroms.fespaces as fe
    return fe.assemble(lambda u, v: fe.inner(u, v)*fe.dx, Vh)


def modify_jet_to_gray(jet_colors=None):
    """Colormap del corso: i primi 85 colori di jet diventano grigi.

    jet_colors opzionale: array (256,4). Restituisce ListedColormap; passare
    l'oggetto ai plot, senza dipendere da registrazioni globali di 'gray_jet'.
    """
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    colors = (mpl.colormaps['jet'](np.linspace(0, 1, 256))
              if jet_colors is None else np.array(jet_colors, copy=True))
    if colors.shape != (256, 4):
        raise ValueError('jet_colors deve avere shape (256,4)')
    colors[:85, :3] = np.linspace(.3, .7, 85)[:, None]
    return ListedColormap(colors, name='gray_jet')


def build_damage_evaluator(mesh, Vh, x0=(0.5, 3.0), device='cpu'):
    """Evaluate del corso nel punto x0 o nei punti (n_points,2).

    Restituisce un operatore torch: evaluator(tensor(Nt,Nh)) -> (Nt,n_points).
    Serve per misure puntuali/task inverso; i dati passati devono trovarsi sullo
    stesso device dell'operatore. Non converte automaticamente gli input.
    """
    from dlroms.minns import Evaluate
    evaluator = Evaluate(mesh, Vh, np.asarray(x0, dtype=float).reshape(-1, 2))
    evaluator.to(device)
    return evaluator


def _fields(fields):
    x = np.asarray(fields, dtype=np.float64)
    if x.ndim < 1 or not np.isfinite(x).all():
        raise ValueError('Campi finiti con almeno un asse richiesti')
    return x


def _mass_apply(M, fields):
    """Applicazione sparse-safe lungo l'ultimo asse, senza densificare M."""
    x = _fields(fields)
    if M.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError('M e numero DOF incompatibili')
    return np.asarray(M @ x.reshape(-1, x.shape[-1]).T).T.reshape(x.shape)


def compute_l2_pod(snapshots, M, max_modes=100, rtol=1e-12):
    """POD non centrata tramite metodo degli snapshot C=S M S.T.

    S: (Ns,Nh) SOLO training; M: (Nh,Nh) SPD, densa o sparse; max_modes:
    limite positivo ai modi; rtol: soglia relativa per autovalori numericamente
    nulli. Restituisce PODResult. Non sottrae la media e non pesa i tempi.
    lambda_i,q_i di C danno V_i=S.T q_i/sqrt(lambda_i); V.T M V=I.
    Energia totale=trace(C), NON somma dei soli autovalori estratti.
    Memoria O(Ns^2), costo di costruzione C rilevante: non chiamare nel tuning.
    Metodi basati su C perdono precisione sui modi molto piccoli.
    """
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
    """A=S M V per base M-ortonormale. Supporta (...,Nh) -> (...,N).

    Usare per train/validation/test, senza ricostruire la POD sui nuovi dati.
    """
    return _mass_apply(M, snapshots) @ np.asarray(basis)


def reconstruct_pod(coefficients, basis):
    """A V.T: (...,N) -> (...,Nh), indipendente dalla norma della POD."""
    return np.asarray(coefficients) @ np.asarray(basis).T


def make_nn_inputs(mu, time_grid):
    """mu=(ntraj,nparam), tempi=(Nt,) crescenti, iniziano da zero e T>0.

    Output (ntraj*Nt,nparam+1), con tempo t/T. Ogni traiettoria occupa un
    blocco di Nt righe, coerente con u.reshape(-1,Nh). Non scala i parametri:
    per u-bolt sono gia in [0,1].
    """
    mu, t = _fields(mu), _fields(time_grid)
    if mu.ndim != 2 or t.ndim != 1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0):
        raise ValueError('mu 2D, t crescente da zero con almeno due punti')
    return np.column_stack([np.repeat(mu, len(t), axis=0), np.tile(t/t[-1], len(mu))])


def fit_coefficient_scaler(A_train):
    """Fit min-max per modo SOLO su A_train=(Ns,N). Output CoefficientScaler.

    Range esattamente nullo sostituito da 1 per evitare divisione per zero.
    Validation/test possono uscire da [0,1]: non vengono tagliati.
    """
    A = _fields(A_train)
    if A.ndim != 2 or not A.size:
        raise ValueError('A_train 2D non vuoto richiesto')
    minimum, scale = A.min(axis=0), np.ptp(A, axis=0)
    return CoefficientScaler(minimum, np.where(scale==0, 1., scale))


def _scaler_arrays(A, scaler):
    if hasattr(A, 'detach') and hasattr(A, 'device'):
        import torch
        return (torch.as_tensor(scaler.minimum, dtype=A.dtype, device=A.device),
                torch.as_tensor(scaler.scale, dtype=A.dtype, device=A.device))
    return scaler.minimum, scaler.scale


def scale_coefficients(A, scaler):
    """(A-minimum)/scale; NumPy o torch, preservando device e gradienti."""
    minimum, scale = _scaler_arrays(A, scaler)
    return (A-minimum)/scale


def unscale_coefficients(A_scaled, scaler):
    """A_scaled*scale+minimum; NumPy o torch, preservando gradienti."""
    minimum, scale = _scaler_arrays(A_scaled, scaler)
    return A_scaled*scale+minimum


def set_seed(seed):
    """Seed Python/NumPy/torch/CUDA. Non garantisce determinismo bit per bit."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(input_dim, output_dim, hidden_layers=(128,)*5,
                activation='silu', device='cpu'):
    """MLP torch con uscita LINEARE: nessun clipping/sigmoid dei coefficienti.

    input_dim=nparam+1; output_dim=N POD; hidden_layers: larghezze positive.
    activation: leaky_relu (slope .1), tanh, silu o relu; device cpu/cuda.
    """
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
    """Numero di parametri torch con requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def coefficient_loss(pred_scaled, target, scaler):
    """MSE tra coefficienti denormalizzati e target NON scalati (tensor).

    Con V.T M V=I equivale a errore L2 spaziale quadratico di regressione,
    diviso per N modi. Non equivale alla media dell'errore RELATIVO L2.
    """
    return ((unscale_coefficients(pred_scaled, scaler)-target)**2).mean()


def train_model(X_train, A_train, X_val, A_val, scaler, *,
                hidden_layers=(128,)*5, activation='silu', lr=1e-3,
                weight_decay=1e-4, max_epochs=1500, batch_size=64,
                device='cpu', seed=42, scheduler_patience=100,
                scheduler_factor=0.5, log_every=250):
    """Training da zero; input e target NumPy 2D, coefficienti NON scalati.

    Crea internamente tensor float32, DataLoader shuffle, Adam e scheduler
    ReduceLROnPlateau sulla coefficient_loss di validation. Tutte le epoche
    vengono eseguite (nessun early stopping). Ripristina il minimo di validation.
    Restituisce dict: model, train_history, val_history, best_val_loss,
    best_epoch (1-based), config. I valori train sono medie durante gli update;
    val si calcola a fine epoca. seed impostato PRIMA di creare rete e loader.
    Non usa test, M o campi FOM. Lo scaler deve essere gia fit sul training.
    """
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
    """X NumPy/tensor (Ns,input_dim) -> NumPy (Ns,N), in batch, senza gradienti.

    Inferisce device/dtype dalla rete; ripristina il precedente stato train/eval.
    """
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
    """Input appiattiti -> snapshot ricostruiti (Ns,Nh), NumPy."""
    return reconstruct_pod(predict_coefficients(model, X, scaler, batch_size), basis)


def l2_norm(fields, M):
    """sqrt(u.T M u) su ultimo asse: (...,Nh) -> (...).

    Piccoli negativi di roundoff sono tagliati a zero; negativi significativi
    segnalano una matrice di massa/input non validi.
    """
    x = _fields(fields)
    squares = np.sum(x*_mass_apply(M, x), axis=-1)
    if np.any(squares < -1e-12*max(1., float(np.max(np.abs(squares))))):
        raise ValueError('Norma quadratica negativa: verificare M')
    return np.sqrt(np.maximum(squares, 0.))


def relative_l2_error(u_true, u_pred, M, zero_policy='raise', zero_tol=0.):
    """||u_true-u_pred||L2 / ||u_true||L2; output in FRAZIONE, non percentuale.

    Campi con shape identica (...,Nh). zero_policy='raise' (default) oppure
    'nan': snapshot con norma<=zero_tol esclusi esplicitamente dai riepiloghi.
    Non sostituisce il denominatore con epsilon: altererebbe la metrica.
    """
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
    """errors=(ntraj,Nt), frazioni. Media temporale aritmetica per traiettoria.

    Con NaN esclude snapshot non definiti; restituisce anche excluded_snapshots.
    Fallisce se una traiettoria non ha snapshot validi. Non e norma L2 nel tempo.
    Output dict: mean_error, worst_trajectory_error, max_snapshot_error,
    trajectory_errors, best_index, representative_index, worst_index.
    """
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
    """Valuta train/validation/test con stessa API; u_true=(ntraj,Nt,Nh).

    X deve seguire make_nn_inputs. Restituisce dict di summarize_errors piu
    errors=(ntraj,Nt), prediction con shape di u_true. Nessun uso di mu globale.
    """
    u = _fields(u_true)
    if u.ndim!=3 or len(X)!=u.shape[0]*u.shape[1]:
        raise ValueError('u_true 3D e numero input coerente richiesti')
    prediction = predict_solution(model, X, scaler, basis).reshape(u.shape)
    errors = relative_l2_error(u, prediction, M, zero_policy)
    return dict(**summarize_errors(errors), errors=errors, prediction=prediction)


def run_experiments(configurations, X_train, A_train, X_val, A_val,
                    scaler, u_val, basis, M, *, device='cpu', zero_policy='raise'):
    """configurations: dict nome -> kwargs di train_model (architettura,seed,...).

    Restituisce dict nome -> risultato training piu evaluation e parameters.
    Usa SOLO validation; non sceglie automaticamente il vincitore. Trattiene
    modelli e campi ricostruiti: usare poche configurazioni per limitare memoria.
    """
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
    """Grafico semilog MSE denormalizzata; restituisce ax, non chiama show."""
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
    """errors: dict etichetta -> frazioni (Nt,), convertite in % nel grafico."""
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
    """GIF di due campi (Nt,Nh) e loro errore assoluto puntuale, non norma L2.

    Scale FOM/ROM comuni e fisse; errore fisso di default per confrontare tempi,
    oppure error_scale='dynamic'. time_grid: tempi fisici dei frame originali.
    max_frames opzionale sottocampiona includendo il frame finale. Il ritmo
    della GIF e quello del writer del corso, non una simulazione in tempo reale.
    Restituisce name. Utilizza fe.gifs.save come nel notebook originale.
    """
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
    """Tempi in secondi, input preparati: NN+unscaling+CPU transfer+ricostruzione.

    Esclude preparazione input, caricamento e training. Sincronizza CUDA prima
    e dopo ogni misura. Include overhead della funzione di inferenza. fom_time
    opzionale deve riferirsi alla stessa traiettoria/intervallo e hardware
    comparabile. Output times, mean, median, std e speedup opzionale.
    """
    import torch
    if n_repeats<1 or warmup<0 or (fom_time is not None and fom_time<=0):
        raise ValueError('Parametri benchmark non validi')
    parameter = next(model.parameters())
    X = torch.as_tensor(X_trajectory, dtype=parameter.dtype, device=parameter.device)
    def synchronize():
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
    """
    Visualizza tutte le coppie di parametri per train/validation/test.

    Input
    -----
    mu_train, mu_val, mu_test : array (n_trajectories, n_parameters)
        Parametri di ogni split, con lo stesso numero e ordine di colonne.

    Output
    ------
    fig, axes
        Figura Matplotlib e array degli assi.

    Serve dopo lo split, per individuare zone poco campionate o differenze
    di copertura. I limiti degli assi sono ricavati dai dati.
    Non chiama plt.show().
    """
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
    """
    Restituisce min, max, media e deviazione standard per split e parametro.

    Input: tre array (n_trajectories, n_parameters).
    Output: DataFrame pandas, con una riga per coppia split/parametro.

    La deviazione standard usa ddof=0, come np.std nel notebook originale.
    Serve per confrontare quantitativamente copertura e distribuzione.
    Non stampa né modifica i dati.
    """
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

    """
    Diagnostica quantitativa degli split nello spazio dei parametri.

    Input
    -----
    mu_train, mu_val, mu_test : array (n_trajectories, n_parameters)
        Matrici finite, con lo stesso ordine delle colonne.
        Per le distanze, i parametri devono avere scale confrontabili:
        nell'u-bolt sono già normalizzati in [0, 1].

    Output
    ------
    balance : pandas.DataFrame
        Per ogni parametro e split:
        - min, max, mean, std;
        - mean_shift_std: differenza assoluta dalla media del training,
          divisa per la deviazione standard del training;
        - std_ratio: deviazione standard dello split / quella del training.

    coverage : pandas.DataFrame
        Per ogni split:
        - numero di traiettorie;
        - mediana, percentile 90 e massimo della distanza dal training;
        - percentuale di punti dentro l'inviluppo convesso del training.

        Per il training, le distanze escludono il punto stesso.
        Se l'inviluppo non è costruibile, la percentuale è NaN e il motivo
        viene riportato nella colonna hull_status.

    Interpretazione
    ---------------
    mean_shift_std vicino a 0 e std_ratio vicino a 1 indicano distribuzioni
    marginali simili. Le distanze descrivono la copertura congiunta.
    Non vengono applicate soglie automatiche né modificato lo split.
    """
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
    