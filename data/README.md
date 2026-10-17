# Data release

Derived data behind the BPTT vs RFLO comparison on MotorNet center-out reaching. The raw
run artifacts (~130 GB, most with a full per-step history of the recurrent weights) are
not in the repository. This directory holds what is needed to recompute every reported
number and its paired test, and to redo the analyses in other ways.

Regenerate everything here with `python export_data.py --out data` (needs the artifacts).
Check it against the reported numbers with `python validate_data.py` (needs only this
directory). Rebuild the figures with `python make_readme_figure.py`.

## Layout

| path | what | tier |
|---|---|---|
| `tables/runs.csv` | one row per run: provenance, final loss, convergence, per-direction error | per run |
| `tables/loss_curves.csv` | training loss per run, mean of each 1000-step window | per run |
| `tables/subspace_windows.csv` | one row per learning-subspace window of a run | per run |
| `arrays/runs_<group>.npz` | eval-rollout trajectories, hidden states, weights, B | per run |
| `arrays/subspace_<group>.npz` | learning-subspace bases (k = 10) and weight change per window | per run |
| `pairwise/subspace_k3.csv` | Grassmann distances (k = 3) between windows over the same steps | pairwise |
| `pairwise/activation_dsa.csv` | activation-DSA distances between runs | pairwise |
| `summary/*.csv` | per-claim tables, built from the files above | summary |
| `provenance.json` | git commit, export parameters, source of every derived column | |
| `MANIFEST.csv` | every file with its size, SHA-256 and shape or row count | |

Tables are CSV. Arrays are compressed NumPy `.npz` files: `np.load(path)` (or
`data_release.load_arrays`). Strings are stored as unicode arrays, so no `allow_pickle` is
needed. Missing values are `NaN`.

A `<group>` is `<effector>_<condition>_<budget>`, e.g. `arm_fixB_40k`, and a `run_id` is
`<group>_s<seed>`, e.g. `arm_fixB_40k_s3`.

## Conditions

| condition | rule | feedback / gradient | effector, budgets |
|---|---|---|---|
| `BPTT` | BPTT | exact gradient | arm 40k, 100k; point mass 70k |
| `projB0` | BPTT | `W_rec`, `W_in` gradients projected onto span(B₀); `W_out` free | arm 40k, 100k |
| `fixB` | RFLO | one shared random B (B₀, `b_seed` 0) for every seed | arm 40k, 100k, 300k; point mass 70k |
| `varyB` | RFLO | a different random B per seed (`b_seed` = seed) | arm 40k |
| `b101`, `b102`, `b103` | RFLO | one shared B per condition, drawn from `b_seed` 101-103 | arm 40k |

Arm: `RigidTendonArm26` (2 joints, 6 muscles), lr 0.025. Point mass: `ReluPointMass24`
(4 actuators), lr 0.01. All runs: 64 recurrent units, plain SGD, batch 32, L1 trajectory
loss, no sensory noise.

`projB0` at 40k for seeds 1 3 4 5 8 9 10 11 is the first 40000 steps of their 100k runs
(bit-identical to a 40k run up to that step). Seeds 1 3 4 5 did have 40k runs, but their
100k runs were saved to the same files and replaced them; seeds 8-11 were only run to 100k. Those rows have a loss curve and a learning
subspace but no eval rollout (`has_eval` false), since the saved rollout was taken at 100k.

## Units and conventions

- **Error** is the terminal error as a fraction of reach distance:
  ‖fingertip at the last step − target‖ / reach. Reach is 0.1 m on the arm and 0.5 on the
  point mass (dimensionless box), so both effectors share one scale.
- **Directions.** Direction index k points at 45·k degrees, counter-clockwise from +x,
  measured from the start position: 0 is right (0°), 2 is up (90°), 4 is left (180°), 6 is
  down (270°). This is MotorNet's `CenterOutReach` convention, checked against the saved
  targets (`goal − FT[0]`).
- **Eval rollout.** `FT` and `H` come from one deterministic rollout per run at the end of
  training, one trial per direction, 100 time steps of 10 ms, no noise.
- **Weights.** `W_rec` is (64, 64) with `W_rec[i, j]` the weight from unit j to unit i.
  Flattened vectors (the subspace bases) are row-major: element `64 * i + j` is `W_rec[i, j]`.
- **Initial weights** are not in the artifacts (the weight history starts after the first
  update). They are replayed from the seed exactly as training did and checked against the
  first snapshot (`init_rel_diff`, one SGD step: 1e-7 to 7e-7 for arm BPTT, about 3e-5 for
  arm RFLO, up to 5e-3 for point-mass RFLO, whose first steps are larger).

## Roster, convergence and pairing

- **Convergence gate** (`converged`): over 2000-step windows of the training loss, the last
  window dropped by less than 2e-3 from the one before, and the last window is below 0.10.
  It flags runs stuck in a high-loss basin. It does not mean learning has finished (below).
  The 2000-step windows are the means of adjacent pairs of the 1000-step windows in
  `loss_curves.csv`.
- **COMMON17** (`in_common17`): arm seeds 1 3 4 5 8 9 10 11 12 14 18 19 20 21 22 23 24, the
  seeds that converged in every 40k arm condition. The 40k conditions hold all 26 seeds so
  the roster choice is visible; the reported 40k arm numbers use COMMON17.
- **multiB grid:** nets 1 3 4 5 8 × B ∈ {B₀ (`fixB`), 101, 102, 103, own (`varyB`)}. Run
  `arm_b101_40k_s4` fails the gate; the crossed, all-converged block is nets 1 3 5 8.
- **Pairing.** Runs with the same seed share their initial network and their stream of
  training targets across every condition (and BPTT/RFLO). Conditions must be compared
  with paired tests over seeds (e.g. flipping condition labels within a seed), not by
  treating runs as independent. Pairwise distances between runs are not independent
  samples either; test them by permutation.
- **Budgets.** 40k steps is a shelf on the arm, not an asymptote: RFLO keeps improving to
  100k and 300k and no rule's loss has flattened at 100k. Compare conditions only at equal
  budget or along the learning curves, never across budgets.

## File schemas

### `tables/runs.csv`

| column | type | meaning |
|---|---|---|
| `run_id` | str | `<group>_s<seed>` |
| `group` | str | `<effector>_<condition>_<budget>`; names the array files |
| `effector` | str | `RigidTendonArm26` or `ReluPointMass24` |
| `condition` | str | see Conditions |
| `rule` | str | `BPTT` or `RFLO` |
| `budget` | int | training steps this row stands for |
| `seed` | int | network seed (init and target stream) |
| `in_common17` | bool | arm seed in COMMON17 (always false on the point mass) |
| `source_path` | str | artifact the row was exported from |
| `step_stop` | int | steps of the artifact used (less than `artifact_num_steps` when truncated) |
| `artifact_num_steps` | int | steps the artifact was trained for |
| `b_seed` | int | seed of RFLO's feedback matrix B; empty for BPTT |
| `grad_basis_seed` | int | seed of the gradient-projection basis (`projB0`); empty otherwise |
| `lr` | float | SGD learning rate |
| `batch_size` | int | training batch size |
| `n_rec` | int | recurrent units |
| `n_out` | int | outputs (muscles or actuators) |
| `proprioception_noise` | float | training-time sensory noise std (0 throughout) |
| `vision_noise` | float | training-time sensory noise std (0 throughout) |
| `loss_type` | str | `trajectory`: L1 distance to target averaged over time |
| `reaching_distance` | float | reach in the effector's units |
| `has_eval` | bool | an end-of-training eval rollout exists (FT, H, error columns) |
| `has_weight_history` | bool | per-step `W_rec` was saved, so subspace windows exist |
| `B_source` | str | where B in the arrays file came from: `artifact`, `artifact (gradient basis)`, `regenerated from b_seed`, `none` |
| `init_rel_diff` | float | ‖replayed W_rec(0) − first snapshot‖ / ‖first snapshot‖ |
| `final_loss` | float | mean training loss over the last 2000 steps used |
| `converged` | bool | convergence gate on the steps used |
| `note` | str | e.g. "first 40k steps of the 100k run" |
| `mean_error` | float | mean of `err_d0` … `err_d7` |
| `err_d0` | float | terminal error toward direction 0, fraction of reach |
| `err_d1` | float | direction 1 |
| `err_d2` | float | direction 2 |
| `err_d3` | float | direction 3 |
| `err_d4` | float | direction 4 |
| `err_d5` | float | direction 5 |
| `err_d6` | float | direction 6 |
| `err_d7` | float | direction 7 |
| `worst_dir` | int | direction with the largest error; −1 without an eval rollout |
| `spread` | float | max − min of the eight errors |

### `tables/loss_curves.csv`

| column | type | meaning |
|---|---|---|
| `run_id` | str | run |
| `step_end` | int | last step of the 1000-step window |
| `loss` | float | mean training loss over the window |

### `tables/subspace_windows.csv`

A window is a step range `[step_start, step_stop)` of one run's `W_rec` history. Every run
with a history has the window covering all its steps; the 100k runs with a history also
have 40k-100k. (Their 0-40k window is the 40k run's, so it is not repeated.)

| column | type | meaning |
|---|---|---|
| `window_id` | str | `<run_id>_w<start>-<stop>k` |
| `run_id` | str | run |
| `step_start` | int | first snapshot (0-based index into the history) |
| `step_stop` | int | one past the last snapshot |
| `bin_size` | int | snapshots averaged per bin before the SVD (50) |
| `var_k3` | float | fraction of the binned trajectory's variance in the top 3 directions |
| `var_k10` | float | same for the top 10 |
| `dW_rec_norm` | float | ‖W_rec(last) − W_rec(first)‖_F over the window |
| `share_dW_rec_in_span_B0` | float | ‖QᵀΔW‖²_F / ‖ΔW‖²_F, Q an orthonormal basis of span(B₀); chance is n_out / 64 |
| `share_dW_rec_in_span_own_B` | float | same with the run's own B (or projection basis); NaN for BPTT |

### `pairwise/subspace_k3.csv`

All pairs of windows of one effector that cover the same steps (e.g. every arm 0-40k
window, across conditions), using the first 3 basis columns.

| column | type | meaning |
|---|---|---|
| `window_i` | str | window |
| `window_j` | str | window |
| `steps` | str | the shared step range, `<start>-<stop>` |
| `geodesic_k3` | float | ‖θ‖ over the 3 principal angles θ (radians) |
| `chordal_k3` | float | ‖sin θ‖ |
| `angle_deg_k3` | float | geodesic / √3 in degrees: the RMS principal angle |

### `pairwise/activation_dsa.csv`

Converted from DSA distance matrices saved by the analysis scripts (angular score, delay
embedding 10, rank 20); DSA was not refit for the release. The fits did not record a random
seed, and refits agree to about 2e-4. The point-mass matrix had no saved row labels; its
order (BPTT seeds 0-14, then RFLO seeds 0-14) is the order the script writes and reproduces
the logged within- and between-rule means. The multiB matrix is a pre-screen. Rows for the
aligned-B condition, which is not in this release, are dropped.

| column | type | meaning |
|---|---|---|
| `run_i` | str | run |
| `run_j` | str | run |
| `distance` | float | DSA distance (angular, in [0, π]) |
| `source_matrix` | str | the saved matrix this pair came from |

### `arrays/runs_<group>.npz`

One row per run with an eval rollout; `n` runs, in `run_id` order.

| variable | shape | dtype | meaning |
|---|---|---|---|
| `run_id` | (n,) | str | run |
| `source_path` | (n,) | str | artifact |
| `FT` | (n, 100, 8, 2) | float32 | fingertip position (time, direction, xy), effector units |
| `H` | (n, 100, 8, 64) | float32 | hidden state (time, direction, unit) |
| `goal` | (n, 8, 2) | float32 | target position per direction |
| `W_in_init` | (n, 64, n_in) | float32 | input weights at step 0 (replayed) |
| `W_rec_init` | (n, 64, 64) | float32 | recurrent weights at step 0 (replayed) |
| `W_out_init` | (n, n_out, 64) | float32 | readout weights at step 0 (replayed) |
| `W_in_final` | (n, 64, n_in) | float32 | after training; NaN on the point mass (not saved) |
| `W_rec_final` | (n, 64, 64) | float32 | after training |
| `W_out_final` | (n, n_out, 64) | float32 | after training; NaN on the point mass (not saved) |
| `dW_in` | (n, 64, n_in) | float32 | `W_in_final − W_in_init` |
| `B` | (n, 64, n_out) | float32 | RFLO's B, or projected BPTT's basis; NaN for BPTT |
| `git_commit` | () | str | commit the export ran at |

Arm: n_in = 16, n_out = 6. Point mass: n_out = 4.

### `arrays/subspace_<group>.npz`

One row per window; `m` windows, in `window_id` order.

| variable | shape | dtype | meaning |
|---|---|---|---|
| `window_id` | (m,) | str | window |
| `run_id` | (m,) | str | run |
| `step_start` | (m,) | int64 | as in `subspace_windows.csv` |
| `step_stop` | (m,) | int64 | as in `subspace_windows.csv` |
| `basis` | (m, 4096, 10) | float32 | orthonormal columns: top 10 right singular vectors of the window's `W_rec` history, centered on its first snapshot and averaged in bins of 50 steps |
| `cumvar` | (m, 10) | float32 | cumulative fraction of variance in the top 1 … 10 directions |
| `W_rec_start` | (m, 64, 64) | float32 | `W_rec` at the first snapshot of the window |
| `dW_rec` | (m, 64, 64) | float32 | last snapshot − first snapshot |
| `git_commit` | () | str | commit the export ran at |

The reported subspace angles use the first 3 columns. Smaller k means truncating `basis`.

### `summary/`

Built by `data_release.build_summaries` from the per-run and pairwise files, so they
cannot disagree with them. `behavior.csv`: error, pole counts and loss per condition,
budget and roster. `corridor.csv`: within-condition subspace distance and span(B₀) share per
condition and window, plus the random-subspace null. `activation_dsa.csv`: within- and
between-condition DSA means per source matrix. `multib_grid.csv`: worst direction per net × B.

## Where numbers come from

- Error, convergence and loss use the repository's `analysis` functions
  (`reach_relative_terminal_error`, `converged`, `loss_windows`); subspaces use
  `analysis.learning_subspace` and the Grassmann distances in `analysis/weights.py`. See
  `provenance.json` → `derived_columns`.
- The span(B₀) share follows `delta_w_rank.py`.
- Some reported numbers were first computed by one-off read-outs rather than repository
  scripts: the 100k and 300k behavior, the projected-BPTT corridor, the multiB grid and its
  DSA matrix. The release recomputes the first three with the definitions above;
  `validate_data.py` compares them with the reported values.

## Loading

With plain NumPy:

```python
import numpy as np
with np.load("data/arrays/runs_arm_fixB_40k.npz") as f:
    H, run_id = f["H"], f["run_id"]               # (26, 100, 8, 64), (26,)
```

With the helper module:

```python
import data_release as dr
runs = dr.load_runs()
fixB = dr.select(runs, "RigidTendonArm26", "fixB", 40000, dr.COMMON17)
arrays = dr.load_arrays("arm_fixB_40k")          # arrays["H"].shape == (26, 100, 8, 64)
bases = dr.load_subspaces("arm_fixB_40k")["basis"]
```

