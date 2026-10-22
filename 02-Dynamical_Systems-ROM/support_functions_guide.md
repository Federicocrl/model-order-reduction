# U-bolt POD-NN Support Utilities

`support_functions.py` contains reusable numerical, training, evaluation, and plotting utilities. The notebooks explain the physical problem, select hyperparameters, and interpret results. All function docstrings are in English and can be inspected with `help(sf.train_model)` or `sf.train_model?` in Jupyter.

## Environment and array conventions

Keep the module beside the notebooks. Use the legacy FEniCS/dlroms Linux or WSL environment for FEM operations. NumPy and SciPy are imported at module load; PyTorch, Matplotlib, pandas, and dlroms are imported by the routines that require them. Importing the module does not construct a mesh or train a network.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `mu` | `(n_trajectories,n_parameters)` | Physical parameters, already normalized in the course dataset |
| `u` | `(n_trajectories,n_times,n_dofs)` | Full-order solution trajectories |
| `snapshots` | `(n_trajectories*n_times,n_dofs)` | One snapshot per row, with all times of each trajectory grouped together |
| `M` | `(n_dofs,n_dofs)` | FEM mass matrix, dense NumPy or sparse SciPy |
| `Vn` | `(n_dofs,n_modes)` | Mass-orthonormal POD basis |
| `A` | `(n_snapshots,n_modes)` | Unscaled POD coefficients |
| `X` | `(n_snapshots,n_parameters+1)` | Physical parameters followed by normalized time |

The mass matrix, basis, and fields must share the exact same mesh and DOF ordering. Matching dimensions alone is insufficient.

## 1. Load data and split trajectories

Split entire trajectories before flattening snapshots. Fit POD and coefficient scaling on training data only. Keep the test set separate from hyperparameter selection.

```python
import numpy as np
import support_functions as sf

with np.load("u-bolt.npz") as data:
    mu, u = data["mu"], data["u"]

indices = np.random.default_rng(115).permutation(len(mu))
train_idx, val_idx, test_idx = indices[:60], indices[60:75], indices[75:]
n_times, n_dofs = u.shape[1:]
time_grid = np.arange(n_times) * 5e-4
```

`plot_parameter_splits`, `summarize_parameter_splits`, and `diagnose_parameter_splits` compare marginal distributions and joint coverage. They do not change the split automatically. Nearest-neighbor distances assume comparable parameter scales; convex-hull coverage is reported as undefined when hull construction fails.

## 2. Construct the FEM space and POD

```python
mesh, Vh = sf.build_ubolt_mesh(stepsize=0.05, knots=30)
sf.check_fem_conversion(u[0, 0], Vh)
M = sf.assemble_mass_matrix(Vh)

training_snapshots = u[train_idx].reshape(-1, n_dofs)
pod = sf.compute_l2_pod(training_snapshots, M, max_modes=100)
cumulative_energy, threshold_modes = sf.plot_pod_analysis(
    pod.singular_values, pod.total_energy
)
```

`compute_l2_pod` uses the uncentered correlation matrix `C=S @ M @ S.T`. The total energy is `trace(C)`, including omitted modes. No temporal quadrature weights or mean subtraction are applied. Correlation-matrix storage grows quadratically with the number of training snapshots, so compute the basis once before tuning.

Choose `n_modes` in the notebook using training/validation analysis, then reuse that basis for every split:

```python
n_modes = 50  # Selected in the forward-model notebook.
Vn = pod.basis[:, :n_modes]
A_train = sf.project_l2(u[train_idx], Vn, M).reshape(-1, n_modes)
A_val = sf.project_l2(u[val_idx], Vn, M).reshape(-1, n_modes)
scaler = sf.fit_coefficient_scaler(A_train)

X_train = sf.make_nn_inputs(mu[train_idx], time_grid)
X_val = sf.make_nn_inputs(mu[val_idx], time_grid)
```

`project_l2` assumes `Vn.T @ M @ Vn = I`. `reconstruct_pod` maps coefficients back to fields using `A @ Vn.T`. The scaler uses a separate min-max range per mode; constant modes use range one. Values outside the training range are not clipped.

## 3. Train and compare neural surrogates

```python
result = sf.train_model(
    X_train, A_train, X_val, A_val, scaler,
    hidden_layers=(128,) * 5,
    activation="silu",
    lr=3e-3,
    weight_decay=0.0,
    seed=42,
    device="cpu",
)
evaluation = sf.evaluate_model_L2(
    result["model"], X_val, u[val_idx], Vn, M, scaler
)
```

The MLP predicts scaled coefficients through a linear output layer. `coefficient_loss` unscales predictions before computing coefficient MSE. For an M-orthonormal basis this measures spatial regression error, but it is not mean relative spatial error.

`train_model` uses Adam and a validation-driven learning-rate scheduler. It runs all requested epochs and restores the lowest-validation-loss weights. The returned dictionary includes the model, training/validation histories, best validation loss, one-based best epoch, and configuration. Seeding controls the random generators without guaranteeing bitwise reproducibility across hardware.

`run_experiments` accepts a mapping from experiment names to training options and returns training results plus validation evaluations. It retains models and reconstructed fields in memory. `run_cached_experiment` instead trains or reloads checkpoints keyed by the configuration, data, basis, mass matrix, scaler, and explicit cache version. Neither utility selects a winning model automatically or requires test data.

## 4. Interpret errors consistently

- `l2_norm(fields, M)` computes a spatial norm along the last axis.
- `relative_l2_error` returns fractions, not percentages. A zero reference norm raises by default; `zero_policy="nan"` marks it undefined without altering the denominator.
- `summarize_errors` first averages valid snapshots within each trajectory, then averages trajectories. It reports median, worst trajectory, largest snapshot error, representative indices, and excluded snapshots. An entirely undefined trajectory raises an error.
- `evaluate_model_L2` adds the predicted fields and snapshot errors to this summary.

Arithmetic averages of relative spatial errors, RMS relative errors, and global space-time error ratios are different metrics. Use the exact definitions in the notebook when comparing reported results. Separate POD projection error from additional neural-regression error.

## 5. Visualize and measure inference

`plot_training_history` returns an axis with coefficient MSE on a logarithmic scale. `plot_error_evolution` accepts fractional error curves and displays percentages. Neither calls `show()`.

`gif_compare` saves FOM/ROM fields and pointwise absolute differences. The latter is not an L2 norm. Field color limits are shared across times; error limits can be fixed or dynamic. GIF playback timing follows the writer rather than physical simulation time.

`benchmark_rom` measures network prediction, coefficient unscaling, CPU transfer, and field reconstruction on prepared inputs. It excludes loading, input preparation, offline POD, and training. CUDA operations are synchronized around timed calls. A supplied FOM timing must cover the same trajectory and time interval on comparable hardware.

## 6. Connect the forward, inverse, and extrapolation notebooks

The inverse notebook loads `artifacts/pod_basis.npy`, `artifacts/scaler.npz`, and `artifacts/pod_nn_model.pt`. It saves identified parameters in `artifacts/inverse_results.npz`. The extrapolation notebook loads those results and additional selected dynamics checkpoints.

These files must be generated/exported or supplied separately; importing this module does not create them. Keep basis columns, scaling arrays, network architecture, parameter conventions, and DOF ordering consistent across stages. The repository currently excludes datasets, model caches, and the artifacts directory from Git; see the root README for execution limitations.
