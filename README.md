# Comparing BPTT and RFLO Learning Dynamics in Motor Control

This repository compares two RNN credit assignment rules on center-out reaching tasks:
- **BPTT** (Backpropagation Through Time)
- **RFLO** (Random Feedback Local Online learning; [Murray, *eLife* 2019](https://elifesciences.org/articles/43299))

Using [MotorNet](https://github.com/OlivierCodol/MotorNet) for biomechanics and [DSA](https://github.com/adnguyen/DSA) (Dynamical Similarity Analysis), we compare:
1. **Activation dynamics ($H$)**: How recurrent state trajectories differ during reaching.
2. **Weight trajectories ($W_{\text{rec}}$)**: How parameter updates diverge over training.

This project originated as a [Neuromatch Academy](https://academy.neuromatch.io/) project.

---

## Authors
- **Timothy Ho**
- **Victoria Pierce**
- **Chirag Vaswani**

---

## Repository Structure

```text
├── rnn.py                # LeakyRNN implementation with pre-activation hooks
├── trainers.py           # BaseTrainer, BPTTTrainer, and RFLOTrainer
├── analysis.py           # Reach metrics and DSA helper functions
├── plot.py               # Trajectory and workspace visualization
├── run_experiment.py     # Multi-seed training script across effectors
├── run_dsa.py            # Activation-space DSA and permutation testing
├── run_weight_dsa.py     # Weight trajectory PCA and weight-space DSA
└── tests/                # Pytest suite for training and environment isolation
```

---

## Setup

### 1. Dependencies

```bash
conda create -n neuromatch python=3.11
conda activate neuromatch
pip install torch numpy scipy matplotlib seaborn scikit-learn
pip install dsa-metric
```

### 2. MotorNet & `CenterOutReach`

The center-out reaching environment (`CenterOutReach`) is currently maintained on a fork of MotorNet (branch [`center-out-reach`](https://github.com/syntactic/MotorNet/tree/center-out-reach)) and will be submitted upstream:

```bash
git clone -b center-out-reach https://github.com/syntactic/MotorNet.git
pip install -e ./MotorNet
```

---

## Usage

### Training Models

Train 5 seeds of BPTT and RFLO for a given effector (`ReluPointMass24` or `RigidTendonArm26`):

```bash
# Point-mass reaching
python run_experiment.py ReluPointMass24

# 2-DOF biomechanical arm
python run_experiment.py RigidTendonArm26
```

Runs are saved to `results/` as self-contained `.pt` artifacts.

### Dynamics Analysis (DSA)

**Activation dynamics:**
```bash
python run_dsa.py RigidTendonArm26
```
Fits HAVOK DMD models on recurrent trajectories across 8 reach directions and runs an exact label-permutation test.

**Weight-space dynamics:**
```bash
python run_weight_dsa.py
```
Centers weight histories by initialization, projects onto shared principal components, and evaluates dynamical similarity in parameter space.

---

## References

- Murray, J. M. (2019). Local online learning in recurrent networks with random feedback. *eLife*, 8, e43299.
- Codol, O., Michaels, J. A., Kashefi, M., Pruszynski, J. A., & Gribble, P. L. (2024). MotorNet: a Python toolbox for controlling biomechanical models with artificial neural networks. *eLife*, 13, RP98044.
- Nguyen, A. D., et al. (2024). Dynamical Similarity Analysis: quantifying similarities in neural dynamics across conditions, systems, and modalities.
