# Support functions per POD-NN u-bolt

Il modulo contiene codice riutilizzabile; il notebook conserva scelta degli iperparametri, motivazioni, formule centrali e interpretazione dei risultati. Non modifica automaticamente il notebook originale.

## Ambiente e convenzioni

Salva `support_functions.py` nella stessa cartella del notebook e del dataset `u-bolt.npz`, e seleziona il kernel WSL/venv in cui hai già installato dlroms/FEniCS. Dipendenze: NumPy, SciPy, PyTorch, Matplotlib; dlroms/FEniCS e il backend mesh/GIF del corso per le sole funzioni FEM. Non occorre eseguire un'installazione all'import del modulo.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import support_functions as sf

# Durante lo sviluppo, per ricaricare il modulo dopo averlo modificato:
# import importlib
# importlib.reload(sf)
```

Tutte le funzioni hanno docstring italiane consultabili con `help(sf.train_model)` o `sf.train_model?` in Jupyter.

| Simbolo | Shape | Significato |
|---|---|---|
| `mu` | `(n_trajectories, n_parameters)` | Parametri fisici, già normalizzati nel dataset del corso |
| `u` | `(n_trajectories, n_times, n_dofs)` | Traiettorie FOM |
| `snapshots` | `(n_trajectories*n_times, n_dofs)` | Una riga per snapshot, tutti i tempi della prima traiettoria prima della seconda |
| `M` | `(n_dofs, n_dofs)` | Matrice di massa FEM, preferibilmente sparse SciPy |
| `Vn` | `(n_dofs, n_modes)` | Base POD L2 |
| `A` | `(n_snapshots, n_modes)` | Coefficienti POD NON scalati |
| `X` | `(n_snapshots, n_parameters+1)` | Parametri e tempo normalizzato |

La matrice di massa deve corrispondere esattamente alla mesh e all'ordinamento DOF dei dati. L'uguaglianza delle dimensioni è necessaria ma non sufficiente. Il modulo ricostruisce la geometria con lo stesso codice del notebook del corso; verificare nell'ambiente locale che i campi del dataset si visualizzino correttamente.

## 1. Dati e split: mantenere nel notebook

Corrisponde alle celle iniziali di caricamento e split. Lo split deve essere per TRAIETTORIA, prima di creare gli snapshot. POD e scaling usano esclusivamente training.

```python
with np.load('u-bolt.npz') as data:
    mu = data['mu']
    u = data['u']

indices = np.random.default_rng(115).permutation(len(mu))
train_idx, val_idx, test_idx = indices[:60], indices[60:75], indices[75:]
mu_train, u_train = mu[train_idx], u[train_idx]
mu_val, u_val = mu[val_idx], u[val_idx]
mu_test, u_test = mu[test_idx], u[test_idx]

Nt, Nh = u.shape[1:]
dt = 5e-4  # passo temporale del dataset u-bolt del corso
t = np.arange(Nt)*dt
T = t[-1]
```

Non serve una funzione per nascondere questo blocco: è breve e documenta le scelte sperimentali.

## 2. Mesh e massa: spostare PRIMA della POD

Sostituisce il lungo blocco geometrico attualmente vicino alla valutazione FEM e la successiva assemblea della massa.

```python
mesh, Vh = sf.build_ubolt_mesh(stepsize=0.05, knots=30)
M = sf.assemble_mass_matrix(Vh)
assert M.shape == (Nh, Nh)
```

`build_ubolt_mesh` incapsula `step` e `arch` del corso, senza modificare la geometria. Non cambiare risoluzione o numero di punti degli archi per un dataset già generato. `assemble_mass_matrix` usa la stessa chiamata `fe.assemble` del notebook originale. M rappresenta gli integrali tra le funzioni di forma: `u.T @ M @ u` è il quadrato della norma spaziale L2 del campo FEM.

## 3. POD L2: sostituire randomized_svd euclidea

```python
snapshots_train = u_train.reshape(-1, Nh)
pod = sf.compute_l2_pod(snapshots_train, M, max_modes=100)
V = pod.basis
s = pod.singular_values
cumulative_energy = pod.cumulative_energy

plt.semilogy(np.arange(1, len(s)+1), s/s[0], 'o-')
plt.xlabel('POD mode')
plt.ylabel(r'$\sigma_i/\sigma_1$')
plt.grid(alpha=0.3)
plt.show()

for threshold in [0.90, 0.95, 0.99, 0.999]:
    if cumulative_energy[-1] >= threshold:
        print(threshold, np.searchsorted(cumulative_energy, threshold)+1)
    else:
        print(threshold, 'non raggiunta nei modi estratti')
```

La funzione usa il metodo degli snapshot: `C=S @ M @ S.T`. Estrae gli autovalori maggiori con `scipy.linalg.eigh`, scarta quelli numericamente nulli, e costruisce `v_i=S.T @ q_i/sqrt(lambda_i)`. Non centra i dati: preserva la scelta del notebook iniziale di usare una POD lineare, non affine. La matrice di massa resta sparse; C è densa. Per 2460 snapshot, C da sola occupa circa 48 MB in float64, ma matrici temporanee, snapshot e solver richiedono ulteriore memoria. Il calcolo non è una randomized SVD e può essere più lento.

L'energia totale è `trace(C)=sum_j ||S_j||L2²`, non la somma dei soli primi 100 autovalori. La POD attribuisce uguale peso a ogni snapshot; non implementa una quadratura temporale trapezoidale. La precisione dei modi a energia molto bassa risente del metodo di correlazione; `rtol=1e-12` evita divisioni per autovalori quasi nulli. Il numero di modi restituito può essere inferiore a `max_modes`.

```python
n_modes = 30  # da confermare con il NUOVO studio di proiezione
if n_modes > V.shape[1]:
    raise ValueError('Modi richiesti superiori ai modi disponibili')
Vn = V[:, :n_modes]
G = Vn.T @ (M @ Vn)
print('Errore ortonormalità M:', np.linalg.norm(G-np.eye(n_modes)))
```

Ora `G` deve essere vicino all'identità. I vecchi studi di energia, dimensione POD e tuning non descrivono questa nuova base e devono essere aggiornati.

## 4. Proiezione e scelta della dimensione POD

```python
A_train = sf.project_l2(snapshots_train, Vn, M)
A_val = sf.project_l2(u_val.reshape(-1, Nh), Vn, M)
u_val_pod = sf.reconstruct_pod(A_val, Vn).reshape(u_val.shape)
projection_errors = sf.relative_l2_error(u_val, u_val_pod, M)
projection_summary = sf.summarize_errors(projection_errors)
print('POD validation [%]:', 100*projection_summary['mean_error'])
```

`project_l2` applica `A=S M V`, non `S V`. `reconstruct_pod` mantiene `A V.T`. Entrambe supportano anche input con più assi iniziali; ad esempio `project_l2(u_val, Vn, M)` restituisce `(n_val,Nt,N)`.

Per ripetere lo studio del numero di modi, usare la stessa base completa calcolata SOLO sul training:

```python
pod_errors = {}
for n in [3, 5, 10, 20, 30, 50, 75, 100]:
    if n > V.shape[1]:
        continue
    basis = V[:, :n]
    a = sf.project_l2(u_val, basis, M)
    reconstruction = sf.reconstruct_pod(a, basis)
    err = sf.relative_l2_error(u_val, reconstruction, M)
    pod_errors[n] = sf.summarize_errors(err)['mean_error']
```

## 5. Input rete e scaling

```python
X_train = sf.make_nn_inputs(mu_train, t)
X_val = sf.make_nn_inputs(mu_val, t)
scaler = sf.fit_coefficient_scaler(A_train)
# Per analisi/visualizzazione, opzionali:
A_train_scaled = sf.scale_coefficients(A_train, scaler)
A_val_scaled = sf.scale_coefficients(A_val, scaler)
```

`make_nn_inputs` combina parametri ripetuti e `t/T`; evita duplicazioni tra train, validation e test. Lo scaling usa minimo/range per modo, fit solo su training. Valori di validation fuori da [0,1] non vengono tagliati. Un coefficiente costante ha range sostituito da 1, evitando divisioni per zero.

Il training riceve `A_train` e `A_val` NON scalati. La rete produce coefficienti nella scala normalizzata; `coefficient_loss` li denormalizza prima di confrontarli con i target. Non occorre costruire manualmente tensor, DataLoader o scaler tensor. `scale_coefficients` e `unscale_coefficients` funzionano con NumPy e torch, conservando gradienti e device per torch.

## 6. Training singolo

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trained = sf.train_model(
    X_train, A_train, X_val, A_val, scaler,
    hidden_layers=[128]*5, activation='silu',
    lr=1e-3, weight_decay=1e-4,
    max_epochs=1500, batch_size=64,
    device=device, seed=42,
)
model = trained['model']
print('Best epoch:', trained['best_epoch'])
print('Best validation loss:', trained['best_val_loss'])
sf.plot_training_history(trained['train_history'], trained['val_history'])
plt.show()
```

Sostituisce `train_architecture`, `train_model`, i cicli manuali iniziali e il secondo blocco di training aggiuntivo per un'esecuzione pulita da zero. La funzione crea una MLP con uscita lineare, Adam e `ReduceLROnPlateau` (factor 0.5, patience 100 di default). Ripristina la rete con loss di validation minima. Non interrompe prima delle epoche richieste e non riprende uno stato precedente dell'optimizer.

Il seed viene impostato prima di creare modello e DataLoader. Il ritorno è un dizionario: NON usare il vecchio unpacking in cinque variabili. `build_model` è disponibile anche separatamente se vuoi una rete senza addestrarla; `count_parameters` ne conta i parametri trainabili. `set_seed` non garantisce uguaglianza bit per bit su tutti gli hardware.

Perché la loss è coerente con L2? Con `V.T M V=I`, l'errore spaziale quadratico di regressione è la somma dei quadrati degli errori dei coefficienti. La MSE divide anche per N modi. Questo fattore costante non cambia il minimo per N fissato, ma rende non direttamente confrontabili le loss di esperimenti con dimensioni POD diverse. Inoltre la loss è ASSOLUTA quadratica, mentre la metrica dell'assignment è RELATIVA e mediata nel tempo.

Per una base M-ortonormale vale anche la decomposizione ortogonale per snapshot:

`||u - V a_pred||L2² = ||u - V a_exact||L2² + ||a_exact - a_pred||2²`.

Il training minimizza la seconda parte; la prima è il limite imposto dalla proiezione.

## 7. Valutazione e metriche

```python
validation = sf.evaluate_model_L2(model, X_val, u_val, Vn, M, scaler)
print('Mean validation [%]:', 100*validation['mean_error'])
print('Worst trajectory [%]:', 100*validation['worst_trajectory_error'])
print('Max snapshot [%]:', 100*validation['max_snapshot_error'])
```

`predict_coefficients` inferisce device e dtype dalla rete, esegue inferenza a batch e restituisce NumPy denormalizzato. `predict_solution` aggiunge ricostruzione FOM. Non impongono clipping a [0,1] del campo di danno: questo cambierebbe il modello.

`l2_norm(fields,M)` restituisce una norma per ogni snapshot sull'ultimo asse. `relative_l2_error` conserva gli assi traiettoria/tempo e restituisce FRAZIONI. Moltiplicare per 100 solo quando stampi o costruisci tabelle. `summarize_errors` usa la media aritmetica nel tempo per ogni traiettoria, poi la media tra traiettorie; distingue peggiore media di traiettoria da massimo istantaneo.

Se la norma FOM è nulla, l'errore relativo non è definito. Il default solleva un errore. Puoi scegliere `zero_policy='nan'` in `relative_l2_error`/`evaluate_model_L2`: gli snapshot indefiniti sono esclusi dalla media e conteggiati in `excluded_snapshots`. Questa scelta modifica la metrica e va documentata; una traiettoria interamente indefinita resta un errore. Per i tempi iniziali nulli, mostrare separatamente l'errore assoluto può essere più informativo. Nessun epsilon viene aggiunto silenziosamente al denominatore.

## 8. Tuning e seed

```python
configurations = {
    '128x4_silu': dict(hidden_layers=[128]*4, activation='silu', seed=42),
    '128x5_silu': dict(hidden_layers=[128]*5, activation='silu', seed=42),
}
experiments = sf.run_experiments(
    configurations, X_train, A_train, X_val, A_val,
    scaler, u_val, Vn, M, device=device,
)

# Tabella di confronto: pandas resta nel notebook
import pandas as pd
rows = {}
for name, result in experiments.items():
    evaluation = result['evaluation']
    rows[name] = {
        **result['config'],
        'parameters': result['parameters'],
        'best_epoch': result['best_epoch'],
        'best_val_loss': result['best_val_loss'],
        'mean_L2_percent': 100*evaluation['mean_error'],
        'worst_trajectory_L2_percent': 100*evaluation['worst_trajectory_error'],
    }
results_df = pd.DataFrame.from_dict(rows, orient='index')
display(results_df)
```

Ogni configurazione può specificare gli argomenti keyword di `train_model`: lr, weight_decay, activation, max_epochs, batch_size, seed, ecc. Per i seed creare cinque configurazioni con identici iperparametri e seed diversi. Non selezionare il seed usando il test.

`run_experiments` esegue le configurazioni ma non decide un vincitore. Conserva modelli e ricostruzioni, quindi richiede memoria. Liste di architetture candidate, tabelle, export Excel e motivazioni restano nel notebook. Dopo aver scelto un altro N POD, ricalcolare coefficienti e scaler prima di ripetere gli esperimenti.

Il confronto del notebook tra checkpoint scelto tramite coefficient loss e tramite errore relativo L2 resta uno studio diagnostico distinto. La funzione standard conserva il checkpoint per coefficient loss: non promette di scegliere quello di minimo errore relativo fisico. Puoi confrontare le due metriche in un blocco dedicato, senza cambiare implicitamente la strategia tra esperimenti.

## 9. Test finale e grafici

```python
# Eseguire solo dopo aver fissato modello e iperparametri dalla validation
X_test = sf.make_nn_inputs(mu_test, t)
test = sf.evaluate_model_L2(model, X_test, u_test, Vn, M, scaler)
idx = test['representative_index']
u_pod = sf.reconstruct_pod(sf.project_l2(u_test[idx], Vn, M), Vn)
pod_error = sf.relative_l2_error(u_test[idx], u_pod, M)
sf.plot_error_evolution(t, {
    'POD projection': pod_error,
    'POD-NN': test['errors'][idx],
})
plt.show()

# Separare norme assolute da errori relativi quando si interpretano picchi:
true_norm = sf.l2_norm(u_test[idx], M)
absolute_error_norm = sf.l2_norm(u_test[idx]-test['prediction'][idx], M)
```

## 10. GIF e strumenti del corso

```python
sf.gif_compare(
    'PODNN_test_comparison.gif',
    u_test[idx], test['prediction'][idx], Vh, t,
    error_scale='fixed', max_frames=Nt,
)
from IPython.display import Image, display
display(Image('PODNN_test_comparison.gif'))
```

La GIF mostra FOM e ROM con la stessa scala cromatica fissa e l'errore assoluto puntuale con scala fissa nel tempo. `error_scale='dynamic'` evidenzia dettagli di ogni frame ma rende meno immediato confrontare intensità tra tempi diversi. I tempi nei titoli sono quelli di `time_grid`, anche con sottocampionamento. La colormap è un oggetto, senza registrazione globale.

`modify_jet_to_gray` è la funzione del corso; il contorno e l'operatore Evaluate sono anch'essi adattati dalle celle del docente. `gif_compare` è una rielaborazione della funzione del nostro notebook, non una funzione fornita dal docente.

Per il task delle misure puntuali:

```python
evaluator = sf.build_damage_evaluator(mesh, Vh, x0=(0.5,3.0), device=device)
trajectory_tensor = torch.as_tensor(test['prediction'][idx], dtype=torch.float32, device=device)
with torch.no_grad():
    damage_at_x0 = evaluator(trajectory_tensor).cpu().numpy()
```

Questo costruisce l'operatore di osservazione; NON implementa ancora l'ottimizzazione del problema inverso. I download gdown e gli strumenti di acquiferi/sidewalk non sono inclusi nel modulo u-bolt.

## 11. Benchmark

```python
X_trajectory = X_test[idx*Nt:(idx+1)*Nt]
timing = sf.benchmark_rom(
    model, X_trajectory, scaler, Vn,
    n_repeats=100, warmup=20, fom_time=19.33,
)
print('ROM mean [ms]:', 1000*timing['mean'])
print('ROM median [ms]:', 1000*timing['median'])
print('Speedup:', timing['speedup'])
```

`19.33` è il riferimento presente nel notebook originale, non un valore misurato nuovamente. Usarlo solo se confrontabile con il benchmark ROM. La misura include inferenza, denormalizzazione, trasferimento alla CPU e ricostruzione, con sincronizzazione CUDA; esclude preparazione input, caricamento e training. Include overhead delle chiamate Python: misurare la pipeline effettivamente usata. Lo speedup dipende anche da BLAS, numero di thread, hardware e dimensione del batch.

## Cambi necessari rispetto al notebook storico

1. Geometria e M prima della POD.
2. `randomized_svd` sostituita da `compute_l2_pod`.
3. Ogni proiezione `S @ V30` sostituita da `project_l2(S,Vn,M)`.
4. Energia euclidea/Frobenius sostituita da energia L2 restituita dalla POD.
5. Scaling ricostruito sulla nuova A_train.
6. Nessuna cella manuale per tensor/DataLoader/loss globale.
7. Training e valutazione letti dai nuovi dizionari restituiti.
8. Errori percentuali solo in output, non dentro le routine numeriche.
9. Duplicati e tentativi intermedi eliminati dalla versione pulita.
10. Riaddestrare: i vecchi pesi NN non sono compatibili con la nuova base/scaling.

## Validazione disponibile

Verifiche sintetiche su matrice di massa sparse: ortonormalità M, ricostruzione a rango completo, energia totale/cumulativa, errore di troncamento, identità tra norma dei coefficienti e norma L2 dei campi ridotti, scaling con modo costante, ordine input, metriche relative e gestione di rango carente/denominatore nullo. PyTorch e dlroms/FEniCS non sono disponibili nell’ambiente di generazione: training, inferenza torch, mesh, Evaluate e GIF non sono stati eseguiti qui. Non è stato eseguito il dataset completo né riaddestrato il modello u-bolt.
