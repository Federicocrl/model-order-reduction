# Key Results and Code References

This summary covers the four published results notebooks. Numbers come from their saved outputs and supporting discussion; the experiments have not been rerun for this document. Where timing prose and saved outputs differ, the saved outputs take precedence. Runtime measurements depend on the execution environment.

**Navigation:** cell numbers below count every cell from one, including Markdown cells; they are not Jupyter execution counts. Section titles and function names provide additional search targets. Source links point to the implementations used by the notebooks.

## 1. Chemical mixing: reduced modeling and optimization

Notebook: [Chemical mixing results](01-FEM-Surrogate-Modelling/Results.ipynb).

### Compression and independent prediction

- The stationary concentration model has **2,763 FEM degrees of freedom**. POD uses **50 training solutions**, with a separate set of **50 test configurations**.
- **34 modes** retain at least **99.99%** of the lifted training snapshot energy. This is Euclidean snapshot energy, not physical energy or a guarantee of pointwise accuracy.
- The selected POD-Galerkin model achieves **0.660% mean**, **0.583% median**, and **1.916% maximum** relative full-field test error.
- The saved test benchmark reports **0.132 ms per online query** and approximately **699× speedup**. The notebook discussion still cites an earlier timing of about 0.12 ms and 770×; those are not the current saved benchmark values.

**Find it in the notebook:** Sections 4.5, 5.2–5.3, and 6.3; cells 36 (dimension selection), 41/43 (affine operators and `build_ROM`), 51 (timing helper), and 55 (final test assessment).

**Implementation:** [FOMsolver](01-FEM-Surrogate-Modelling/MixerFunctions.py#L258) generates reference concentrations; [assemble_affine_operators](01-FEM-Surrogate-Modelling/MixerFunctions.py#L300) provides the parameter-independent operators. POD SVD, lifting, reduced projection, `build_ROM(N)`, and `benchmark_rom(...)` are implemented directly in the notebook. Online queries include reduced assembly, solution, and field reconstruction, excluding offline projection.

### Optimization and the limits of a global error metric

At the target point `(1.5, 0.25)`, the 34-mode multistart optimization returns cylinder velocities approximately **(-1.22749, -2.68747, -5.00000)**.

| Quantity | Saved result |
| --- | ---: |
| ROM-predicted concentration at the candidate | 0.403723 |
| FOM concentration at the same candidate | 0.429193 |
| FOM-validated reduction from zero rotation | 16.92% |
| Relative pointwise prediction error | 5.934% |

The dimension study is nonmonotonic: 20 modes give a **20.00%** FOM-validated reduction, while 40 modes give **19.93%** with approximately **0.90%** pointwise prediction error. More modes do not necessarily produce a better optimized candidate.

At the 34-mode optimized candidate, the field diagnostic separates **0.733% POD projection error**, **1.754% Galerkin discrepancy**, and **1.901% total Galerkin field error**. These are norm magnitudes, not additive percentages. Their orthogonal squared-error identity is verified to numerical precision.

**Find it in the notebook:** Sections 7.2–7.4 and 8; cells 62 (multistart `minimize`), 64 (FOM verification), 66 (dimension sensitivity), and 70 (error decomposition). `build_ROM` supplies the coefficient, field, and pointwise-objective callables; `mixer.FOMsolver(mu_opt)` independently checks the chosen candidate.

**Interpretation:** the surrogate supports efficient parameter exploration and improves the FOM objective, but neither a small global error nor multistart convergence establishes a global optimum or equally small pointwise error.

## 2. U-bolt: POD-NN forward surrogate

Notebook: [Forward prediction results](02-Dynamical_Systems-ROM/I%26II_Results.ipynb).

### Representation and network selection

- The dataset contains **100 trajectories**, each with **41 snapshots** and **6,636 FEM degrees of freedom**. The split is **60 training / 15 validation / 25 test trajectories**.
- The energy criterion initially suggests 26 modes; validation analysis retains **50 mass-weighted POD modes**, with a **1.770% mean validation projection error** and **1.779% mean test projection error**.
- The selected network has **five hidden layers of 128 neurons**, **SiLU** activation, learning rate **0.003**, zero weight decay, batch size 64, and a 1,500-epoch budget. The reference seed is 42; the network has **73,138 trainable parameters**.
- The explored architecture, activation, learning-rate, and regularization comparisons favor this configuration among the tested candidates. This is an experimental selection, not proof of a universally optimal architecture.
- Across five seeds, mean validation errors range from **2.845% to 3.347%**, with a reported mean and standard deviation of **3.075% ± 0.212%**.

**Find it in the notebook:** Sections 2.3 and 4; cells 16–22 (POD and validation dimension study), 41–53 (tuning utilities, experiments, and seed robustness), and 55 (reference checkpoint).

**Implementation:** [compute_l2_pod](02-Dynamical_Systems-ROM/support_functions.py#L146), [project_l2](02-Dynamical_Systems-ROM/support_functions.py#L174), [train_model](02-Dynamical_Systems-ROM/support_functions.py#L277), and [run_cached_experiment](02-Dynamical_Systems-ROM/support_functions.py#L897). The notebook's `run_tuning` and `save_experiment_table` record validation comparisons in [tuning_results.xlsx](02-Dynamical_Systems-ROM/tuning_results.xlsx).

### Generalization and error decomposition

| Metric | Validation | Independent test |
| --- | ---: | ---: |
| Mean relative spatial L2 error | 2.845% | 3.242% |
| Worst time-averaged trajectory error | 4.625% | 6.396% |
| Largest individual snapshot error | 36.673% | 51.057% |
| POD projection mean error | 1.770% | 1.779% |
| RMS relative spatial error | 5.422% | 6.597% |
| Global weighted error ratio | 2.845% | 3.359% |

The low average test error meets the study's 10% mean-error target, but individual transient snapshots can be much less accurate. Orthogonal error analysis assigns approximately **69.96% of the aggregate squared relative test error to NN regression**, and **30.04% to POD truncation**. These shares are not obtained by subtracting the mean-error percentages.

**Find it in the notebook:** Section 5; cells 57–58 (split metrics), 61–63 (error decomposition), 64–68 (component comparison and metric definitions), and 70–72 (spatial visualizations).

**Implementation:** [evaluate_model_L2](02-Dynamical_Systems-ROM/support_functions.py#L425), plus `relative_l2_error`, `summarize_errors`, and `l2_norm` in the same module. The notebook defines `error_metrics(errors, reference_norms)` for mean, RMS, and global metrics.

### Runtime and offline investment

The saved online benchmark reports **41.596 ms median latency** and **313.8× mean-based speedup** relative to a supplied FOM reference of **19.33 s per complete trajectory**. The timing-distribution cell gives a mean/median ratio of **1.48**, showing substantial variability.

**Reporting discrepancy:** cell 75 labels the median as both “ROM mean” and “ROM median” because `t_rom = benchmark["median"]`. The actual speedup comes from `benchmark["speedup"]`, which uses the mean; it implies a mean latency of approximately **61.6 ms**. The prose still quotes an earlier 34.90 ms / 554× benchmark. These different timing summaries must not be combined as one measurement.

Using median online latency, the saved cost scenarios give a first profitable evaluation of **7** for measured training only (**2.11 min**), **187** for an illustrative one-hour offline budget, and **287** when that budget also includes generating 100 FOM trajectories. The latter two budgets are hypothetical; the training-only estimate excludes data generation, POD construction, and tuning.

**Find it in the notebook:** Sections 6.1–6.3; cells 75–76 (online benchmark/distribution), 79 (training costs), and 82 (cost scenarios). [benchmark_rom](02-Dynamical_Systems-ROM/support_functions.py#L534) times inference, unscaling, CPU transfer, and field reconstruction on prepared inputs. Hardware equivalence with the supplied FOM reference is not established.

## 3. U-bolt: single-sensor parameter identification

Notebook: [Parameter identification results](02-Dynamical_Systems-ROM/III_Results.ipynb).

The frozen POD-NN fits **41 observations at `(0.5, 3)`**. A 27-point coarse grid supplies five starts, supplemented by ten random starts for **15 L-BFGS-B optimization runs**. No extra FOM solves are used during optimization.

| Quantity | Best identified result |
| --- | ---: |
| Normalized parameters | (0.401006, 0.896568, 0.898164) |
| Diffusion coefficient D | 2.51771 |
| Reaction coefficient r | 788.07508 |
| Initial-location parameter delta | 1.89816 |
| Sensor MSE | 6.5049e-6 |
| Sensor RMSE | 0.0025505 |
| Relative temporal signal error | 0.8783% |
| Peak-normalized RMSE | 0.2973% |
| Maximum absolute sensor discrepancy | 0.00634 |

**Find it in the notebook:** Sections 1–2; cell 6 precomputes the sensor observation of the POD basis, cell 9 defines `predict_sensor`, cell 12 defines `objective` and runs multistart optimization, and cell 14 reports fit metrics. Cell 21 exports the inverse results.

**Implementation:** the notebook combines `sf.build_damage_evaluator`, `sf.make_nn_inputs`, and `sf.predict_coefficients` with a sensor-mode matrix-vector product. The optimizer changes the three physical inputs, not the network weights.

### Spatial ambiguity despite a good sensor fit

A distinct successfully terminated candidate, **(0.176431, 0.906241, 0.510301)**, lies at normalized parameter distance **0.4483** from the best fit. Its sensor MSE is **3.1603e-5**, approximately 4.9 times larger, yet still small in absolute terms. The two surrogate fields differ by up to **0.9179** at the final observed time.

**Find it in the notebook:** Section 3.2, cell 19. The code filters successful runs by a minimum parameter distance of 0.10, selects the lowest-MSE eligible alternative, and reconstructs both fields using `sf.reconstruct_pod`.

**Interpretation:** accurate reproduction of one sensor signal does not establish unique recovery of the true parameters or the spatial field. These results show practical ambiguity in the surrogate-based inversion; the unknown true parameters and corresponding FOM solution are unavailable for independent validation.

## 4. U-bolt: temporal dynamics and extrapolation

Notebook: [Temporal extrapolation results](02-Dynamical_Systems-ROM/IV_Results.ipynb).

### Initial identification and diagnostic experiments

The initial temporal study identifies dynamics on **[0, T/2]** and forecasts to **T**, where **T=0.02**. The fixed POD basis was constructed from complete training trajectories over [0,T], so the temporal holdout applies to dynamics identification, not the spatial representation. High reaction in the main comparisons means **mu2 >= 0.65**.

| Experiment | Main finding | Notebook location and implementation |
| --- | --- | --- |
| DMD rank selection | Rank 18 gives test median coefficient error **5.67%**, but mean **57.93%** and maximum **418.62%**. Error has Spearman correlation **0.954** with mu2. | Sections 2.2–2.3, cells 10/12; [fit_dmd](02-Dynamical_Systems-ROM/extrapolation_functions.py#L423), [predict_dmd](02-Dynamical_Systems-ROM/extrapolation_functions.py#L499), and `dmd_forecast`. |
| DMD transient/window sensitivity | Excluding three snapshots with rank 16 yields **6.36%** test median, versus baseline **5.67%**. A separately tuned high-reaction subset improves **64.44% to 54.55%**, but uses a different threshold, mu2 >= 0.7642. | Section 2.4 and its saved diagnostics; do not compare the subset statistics directly with the main benchmark's high-reaction group. |
| Continuous SINDy identification | Baseline derivative residual is **4.95%**, yet **4,592/4,600 coefficients** remain active. Tuned alpha=0.01, threshold=10 completes **11/15** validation cases and only **2/6** high-reaction cases. | Sections 3.4–3.7, cells 19–25; `build_sindy_library`, `fit_sindy`, [simulate_sindy](02-Dynamical_Systems-ROM/extrapolation_functions.py#L848), and [tune_sindy](02-Dynamical_Systems-ROM/extrapolation_functions.py#L908). |
| Galerkin dimension study | On the 12 cases completed at every dimension, median final FEM error improves **7.97% to 2.97%** from 10 to 50 modes. The 50-mode model completes **14/15** cases overall. | Section 4.2, cells 30–33; [assemble_galerkin_operators](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1120), [simulate_galerkin](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1220), and `benchmark_galerkin`. |
| Hybrid correction | Training derivative discrepancy falls **176.80% to 12.65%**, but completion remains **14/15**. At strength 0.3, high-reaction median improves **13.11% to 7.93%** while the maximum worsens **90.22% to 395.00%**. | Section 4.3, cells 35–37; [fit_hybrid_correction](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1397) and `simulate_hybrid`. |
| POD-LSTM | Selected 10-mode, history-5, hidden-128 model completes **15/15** validation rollouts; median FEM error **10.98%**, high-reaction median **18.67%**. | Section 4.4, cell 39; [train_pod_lstm](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1636) and [rollout_pod_lstm](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1605). |
| Discrete SINDy | Selected 5-mode map completes **15/15** validation rollouts with **28.19%** median FEM error. Increasing modes improves projection but not these tested evolution maps. | Section 4.5, cells 42–44; [fit_discrete_sindy](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1857) and [rollout_discrete_sindy](02-Dynamical_Systems-ROM/extrapolation_functions.py#L1934). |

Low derivative residuals, low projection errors, and numerical completion are distinct from accurate autonomous forecasts. Window adjustments do not resolve the difficult high-reaction regime in these experiments.

### Independent 25-trajectory comparison

Final-time errors below are **FEM spatial L2 percentages at T**, not the earlier coefficient-only errors. Medians and P90 exclude failed forecasts; completion counts must accompany them.

| Model | Completed | High-reaction completed | Median | P90 | High-reaction median |
| --- | --- | --- | ---: | ---: | ---: |
| DMD | 25/25 | 8/8 | 5.70% | 146.81% | 87.19% |
| Continuous SINDy | 17/25 | 1/8 | 7.50% | 60.39% | 121.55% |
| POD-Galerkin | 23/25 | 6/8 | 3.99% | 17.13% | 14.35% |
| POD-LSTM | 25/25 | 8/8 | 10.77% | 43.54% | 30.03% |
| Discrete SINDy | 25/25 | 8/8 | 29.90% | 34.65% | 30.45% |

Galerkin has the smallest conditional median, but fails on two high-reaction trajectories. LSTM completes every forecast with a lower overall median than discrete SINDy. DMD's favorable overall median conceals large high-reaction errors. Restricting comparison to the 17 cases completed by all models changes the population and cannot establish robustness on the omitted cases.

**Find it in the notebook:** Sections 5.1–5.4; cell 46 defines the five forecast wrappers and configurations, cell 47 runs [benchmark_temporal_models](02-Dynamical_Systems-ROM/extrapolation_functions.py#L2016), and cells 48/51 visualize aggregate and pointwise behavior. The pointwise illustrations use a diagnostic location distinct from the final target.

### Final forecast beyond the available time interval

Full-window models are selected using complete training trajectories to T, with configurations distinct from the initial benchmark. The POD-NN reconstructs a history at the inferred parameters; temporal models propagate it **40 additional steps to 2T=0.04**. There is **no beyond-T FOM reference in these reported comparisons**.

At the requested point **(3.5, 2)**, the 50-mode POD-NN seed is approximately **-0.011762 at T**. Predictions at 2T are:

| Model | Predicted target damage | Outcome |
| --- | ---: | --- |
| DMD | 0.04233 | Completed; spatial field violates physical bounds |
| Continuous SINDy | — | Integration did not complete |
| POD-Galerkin | — | Integration did not complete |
| POD-LSTM | 1.16560 | Completed; exceeds the physical upper bound of 1 |
| Discrete SINDy | 1.27613 | Completed; exceeds the physical upper bound of 1 |

At 2T, the DMD field ranges approximately **-2.05 to 1.22**. Approximately **59.57%** of LSTM nodal values and **50.00%** of discrete-SINDy nodal values exceed 1; these are DOF fractions, not spatial area fractions.

**Find it in the notebook:** Sections 6.1–6.5; cell 53 calls [run_final_full_window_tuning](02-Dynamical_Systems-ROM/extrapolation_functions.py#L2366), cell 58 restores the history, cell 61 defines `store_forecast`/`attempt` and generates forecasts, and cells 64–66 diagnose spatial admissibility and animate the fields.

**Interpretation:** LSTM and discrete SINDy suggest a possible strong damage increase, but overshoots, model disagreement, and incomplete integrations prevent assigning a validated intensity or arrival time. Clipping predictions to 1 would not establish physical saturation.

### Reaction-regime sensitivity

Changing only the normalized reaction parameter to **mu2=0.2** before reconstructing a new seed history allows **all five forecasts to complete**. Continuous SINDy predicts **0.04133** at the target with nodal extrema inside [0,1]; Galerkin gives approximately **-0.00068**, and LSTM **0.00860**, with much smaller bound violations than at the inferred high-reaction parameters. DMD and discrete SINDy still contain negative spatial values.

**Find it in the notebook:** Section 6.6, cells 69–71. `run_low` repeats the model calls on the new POD-NN history and refits DMD to that history. This changes initialization as well as reaction, so it does not isolate the dynamics alone; plausible bounds also do not verify accuracy without a reference.

## Scope of the evidence

The mixing and forward-surrogate notebooks demonstrate accurate global predictions and substantial indicative acceleration. The inverse notebook demonstrates a strong sensor fit with practical spatial ambiguity. The temporal notebook identifies a clear reliability limitation in the high-reaction extrapolation regime. These conclusions concern different protocols and metrics and should not be reduced to one accuracy or speedup figure.

The historical `fit_dmd`/`fit_sindy` definitions at the beginning of `extrapolation_functions.py` are superseded later in that module. The references above describe the later implementations used by the published temporal notebook. Environment specification, access to numerical artifacts, and the runtime-reporting discrepancies remain reproducibility/maintenance work; see the [README](README.md) for required local files.
