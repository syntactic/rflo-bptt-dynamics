# Comparing BPTT and RFLO Learning Dynamics in Motor Control

Two recurrent networks learn the same center-out reaching task, one with
**backpropagation through time** and one with **RFLO** (Random Feedback Local Online
learning; [Murray, _eLife_ 2019](https://elifesciences.org/articles/43299)), which replaces
the transported gradient with a fixed random feedback matrix `B` and local eligibility
traces. Both drive [MotorNet](https://github.com/OlivierCodol/MotorNet) effectors: a planar
point mass with four Cartesian actuators, and a two-joint arm with six Hill-type muscles.
RFLO learns the point mass as well as BPTT and plateaus on the arm at more than twice
BPTT's error. We want to know what that failure looks like and what causes it.

This project started as a [Neuromatch Academy](https://academy.neuromatch.io/) project.

## Authors

- **Timothy Ho**
- **Victoria Pierce**
- **Chirag Vaswani**

---

## Preliminary results

![Summary figure](figures/canalization_summary.png)

Arm, 17 seeds that converged in every condition, lr 0.025, 40k steps, 64 units. "Shared `B`"
means every RFLO seed uses the same feedback matrix. "Per-seed `B`" gives each seed its own,
which is how RFLO is usually run.

**RFLO plateaus on the arm.** RFLO's reaches end 0.27 of the reach distance from the target on
average, and BPTT's end 0.11 away. The gap holds at every learning rate we tried (0.01 to
0.05). Wider networks (128 units for 100k steps, 256 for 120k) stay at the same plateau, and a
loss weighted toward the end of the reach lowers the error without changing its pattern.

**The errors lie along one axis.** Panel C: RFLO's errors lie along the axis from the rightward
target (direction 0) to the leftward one (direction 4). Each seed is worst at one of the two
poles, and its error falls off with distance from that pole. With a shared `B`, all 17 seeds
are worst at direction 0 or 4 (6 and 11 seeds). BPTT is flat at 0.11 in every direction. Which
pole a seed fails at depends on both its initialization and its `B`: swapping `B` at a fixed
initialization flips 7 of 17 networks. The axis is not the arm's inertia. At the start posture
the hand is hardest to accelerate along the forearm, 36° away from it.

**RFLO's weight changes stay in a few directions.** Williams, Payeur & Lajoie
([arXiv:2606.00243](https://arxiv.org/abs/2606.00243)) show that in linear RNNs, RFLO's
solutions are low-rank perturbations of the initial weights. We see the same constraint in a
nonlinear network controlling a closed-loop arm. Each seed's recurrent weight trajectory has a
learning subspace: the top-3 directions its weights moved along. RFLO seeds that share `B` have
subspaces 70.3° apart. BPTT seeds are 84.1° apart, per-seed-`B` RFLO seeds 88.6°, and two
random 3-dimensional subspaces of the 4096-dimensional weight space 88.7° (paired permutation
over seeds, p = 1.5e-5 for both contrasts). The cosine between seeds' weight changes agrees:
0.40 with a shared `B`, spread over an effective 4.8 of 17 dimensions, against 0.016 by chance.
When `B` is the network's own readout (the exact credit direction, rescaled to a random `B`'s
norm), seeds sit 85.0° apart, close to BPTT, and the arm error only drops from 0.27 to 0.25.

**The shared weight directions don't show up in dynamics or behavior.** Activation-space DSA
distance between seeds is 0.047 with a shared `B` and 0.069 with per-seed `B` (paired p =
0.073). The across-seed spread of the reach profile is 0.784 and 0.824 (p = 0.249). BPTT
seeds are closer to each other in activation space than RFLO seeds are: 0.029 against 0.047
(p = 1.5e-3). BPTT also reaches much better on the arm, so part of that gap may come from
performance.

**The point mass as a negative control.** At a shared learning rate of 0.01 the two rules
land within 1.3% and 2.0% of reach distance of the target (15 seeds each), a gap inside the 5%
equivalence margin, and every run finishes within 10% of reach on every direction. There the
weight geometry shows no shared directions: RFLO seeds sit 86.5° apart against BPTT's 88.5°
and a null of 88.7°, with a participation ratio of 14.4 out of 15 across-seed directions.

**Training stability.** RFLO converged in 24 of 26 arm seeds with a shared `B`, 20 of 26 with
per-seed `B`, and BPTT in 26 of 26.

### Open questions

- **Why does RFLO fail the arm?** Either its fixed update directions are enough to cause the
  failure, or its approximate credit is the problem. RFLO's error signal here is
  backpropagated through the arm's dynamics, but it drops credit through the network's own
  recurrence and through the sensory loop. On a first look at 5 seeds, RFLO's total change in
  recurrent weights has an effective rank of about 1.8 against BPTT's 4.7, and 59% of it lies
  along `B` (9% by chance). Next we'll train BPTT with its gradients projected onto the span of
  `B`. If it fails along the same axis, the directions are enough. If it learns the arm, the
  credit is what's missing.
- **What sets the axis?** Inertia doesn't. The muscles' force capacity varies with direction
  and is the next candidate.
- **How often does a network change poles?** Each network has been trained with only one
  alternate `B`, so 7 of 17 is a lower bound. Training a few networks with many `B`s each
  would measure it.

### Notes

Hidden units have no canonical ordering, so cross-seed weight comparisons mix real geometry
with relabeling. Every weight claim here is made in the paired frame, where the two runs of
a seed share an initialization. MotorNet is closed-loop, so hidden-state trajectories are
partly driven by sensory input and not purely intrinsic dynamics. Raw DSA distances are
small fractions of the metric's [0, π] range and are read against anchors (a time-shuffled
surrogate sits at 0.245, networks trained on the other effector at 0.052), not in absolute
terms.

---

## Setup

```bash
conda create -n neuromatch python=3.11
conda activate neuromatch
pip install torch numpy scipy matplotlib seaborn scikit-learn
pip install dsa-metric
```

The `CenterOutReach` environment lives on a fork of MotorNet (branch
[`center-out-reach`](https://github.com/syntactic/MotorNet/tree/center-out-reach)), to be
submitted upstream:

```bash
git clone -b center-out-reach https://github.com/syntactic/MotorNet.git
pip install -e ./MotorNet
```

Run `pytest` from the repository root to check the trainers and the analysis math.

---

## Running experiments

`run_experiment.py` trains one or more seeds and writes a self-describing `.pt` artifact per
(effector, seed, rule), recording its own learning rate, width, feedback-matrix setting and
noise levels:

```bash
python run_experiment.py RigidTendonArm26 --seeds 0 1 2 --lr 0.025 \
    --num-steps 40000 --rule rflo --b-seed 0 --output-dir results/evaryB_fixB
```

The scripts behind the results above are in `drivers/`. Each runs one seed per process,
throttles to `MAX` concurrent jobs, skips artifacts already on disk (so an interrupted
campaign resumes by relaunching), and takes its parameters from environment variables:

| driver                 | what it runs                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `evaryB.sh`            | RFLO on the arm twice, once with a shared feedback matrix and once with a per-seed one.                                              |
| `arm_bptt.sh`          | BPTT on the arm at the matched rate and budget. Seed _s_ starts from the same initialization as RFLO seed _s_.                       |
| `lr_sweep.sh`          | Learning-rate sweep for either effector, both rules. The point mass matches at 0.01 and the arm does not match at any rate.          |
| `pointmass_matched.sh` | Point mass, 15 seeds at lr 0.01 with weight history: the matched-behavior negative control.                                          |
| `arm_lift.sh`          | Attempts to lift RFLO's arm plateau: wider networks, a terminal-weighted loss, and a sweep of `B` at one fixed network.              |
| `alignedB.sh`          | RFLO with the random feedback matrix replaced by the live readout weights, testing whether the corridor needs feedback to be random. |
| `fixnet.sh`            | One fixed network, many feedback matrices. Separates `B`'s effect on the reach profile from the initialization's.                    |
| `noisy.sh`             | Proprioceptive and visual feedback noise during training, scaled to each effector's reach distance.                                  |

```bash
# 15 seeds, both feedback conditions, with weight history (~0.66 GB per seed)
WEIGHT_HISTORY=1 SEEDS="0 1 2 3 4" nohup bash drivers/evaryB.sh &>logs/evaryB/driver.log &

These are the parameters that worked for me on my Macbook Air M3. A better computer would probably be able to run more concurrent processes. I have tested this on Linux and also with GPUs but the RNGs are hardware-specific so comparisons are best done between runs generated by the same machine.
```

---

## Analysis

Getting all runs to the same behavioral performance (and ideally solving the task) is a prereq for the analysis.
The below script checks that training loss has both
flattened and landed below an absolute bound, since a run can stall in a failed basin and
still look converged by a slope test:

```bash
python check_convergence.py --dir results/evaryB_fixB_wh --rule RFLO
```

Then, by channel:

```bash
# behavior: per-direction terminal error, as a fraction of reach distance
python per_direction_error.py --dir results/evaryB_fixB_wh --rules RFLO --seeds 1 3 4 --per-seed
python summarize_behavior.py --dir results/pm_lr0.01_N15 --n-seeds 15

# activation space: DSA over hidden-state trajectories
python run_activation_dsa.py RigidTendonArm26 --results-dir results/clean \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

# weight space: learning-subspace angles and weight-displacement geometry
python run_subspace_geometry.py RigidTendonArm26 \
    --condition fixB=results/evaryB_fixB_wh=RFLO \
    --condition varyB=results/evaryB_varyB_wh=RFLO \
    --condition BPTT=results/arm_bptt_40k_wh \
    --contrast varyB-vs-fixB --contrast BPTT-vs-fixB \
    --seeds 1 3 4 5 8 9 10 11 12 14 18 19 20 21 22 23 24
python delta_w_geometry.py --dir results/evaryB_fixB_wh --effector RigidTendonArm26 \
    --rules RFLO --seeds 1 3 4 5 8 9 10 11 12 14 18 19 20 21 22 23 24

# significance for a family of contrasts, Holm-corrected
python run_paired_permutation.py --cell RigidTendonArm26 \
    results/clean/RigidTendonArm26_activation_dsa_matrix.npy \
    BPTT=results/clean RFLO=results/clean

# the README figure
python make_readme_figure.py
```

Two runs of the same seed share an initialization and a target stream, so they are not
freely exchangeable. Every test permutes the condition label within a seed rather than
shuffling labels across seeds, and a family of tests is Holm-corrected.

---

## Repository layout

```text
├── rnn.py                    # LeakyRNN, with per-step outputs for RFLO's traces
├── trainers.py               # BaseTrainer, BPTTTrainer, RFLOTrainer
├── analysis/                 # library, split by measurement channel
│   ├── io.py                 #   artifact loading, per-direction reshaping
│   ├── behavior.py           #   reach metrics
│   ├── activation.py         #   hidden-state PCA and DSA
│   ├── weights.py            #   weight-displacement and subspace geometry
│   └── stats.py              #   grouped dispersion, permutation tests, Holm
├── plot.py                   # trajectory and workspace plotting
├── drivers/                  # training campaigns (see the table above)
├── run_experiment.py         # train one or more seeds, write artifacts
├── check_convergence.py      # the convergence gate
├── per_direction_error.py    # per-direction reach error
├── summarize_behavior.py     # reach-relative error and scaled success
├── run_activation_dsa.py# activation-space DSA at N=15
├── run_subspace_geometry.py  # learning-subspace (Grassmann) geometry
├── delta_w_geometry.py       # weight-displacement cosine and participation ratio
├── run_paired_permutation.py # paired permutation tests with Holm correction
├── make_readme_figure.py     # the figure above
└── tests/                    # pytest suite
```

---

## Conventions

- Conditions are separated by output directory (`results/clean`, `results/evaryB_fixB_wh`). Filenames stay keyed on effector, seed and rule and the rest lives in the artifact's metadata.
- `--b-seed 0` gives every RFLO run the same feedback matrix; `--b-seed match` draws a new
  one per network seed. BPTT has no feedback matrix, so every `B` experiment is RFLO-only
  and doubles half the runs.
- Weight history is a full recurrent weight matrix per step, about 0.66 GB per arm seed at
  40k steps. Use `--no-weight-history` for anything that only needs behavior or hidden states.
- BPTT and RFLO share an initialization within a seed, by deep copy. Both use plain SGD;
  swapping in Adam for BPTT would break the comparison.
- Noise levels are relative to each effector's reach distance (0.1 m for the arm, 0.5 in the
  point mass's dimensionless box). A shared absolute value would be a different perturbation
  for each.
- Evaluation rollouts are deterministic even when training is noisy: MotorNet's
  `deterministic` flag does not gate the proprioceptive and visual channels, so the trainer
  zeroes those on its own evaluation environment.

---

## References

- Murray, J. M. (2019). Local online learning in recurrent networks with random feedback. _eLife_, 8, e43299.
- Codol, O., Michaels, J. A., Kashefi, M., Pruszynski, J. A., & Gribble, P. L. (2024). MotorNet: a Python toolbox for controlling biomechanical models with artificial neural networks. _eLife_, 13, RP98044.
- Ostrow, M., Eisen, A., Kozachkov, L., & Fiete, I. (2023). Beyond geometry: comparing the temporal structure of computation in neural circuits with dynamical similarity analysis. _NeurIPS_.
- Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. _Nature Communications_, 7, 13276.
