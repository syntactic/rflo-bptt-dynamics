"""Weight-space channel: corroboration only, and only in a gauge-safe frame.

Hidden units have no canonical labeling, so raw cross-seed weight geometry mixes
real differences with relabeling artifacts. The ΔW-corridor metrics here are
meant for the shared-init paired frame (or after permutation alignment). DSA is
not applied to weights: learning trajectories are non-autonomous optimizer
transients, not autonomous flows, so direct geometry is the right tool.
"""

from collections import defaultdict

import numpy as np
import torch
from DSA import coarse_grain, pca_reduce


def process_weight_trajectories(runs, bin_size=10, n_components=0.95):
    """Center, coarse-grain, and PCA-reduce recurrent weight trajectories across runs."""
    smoothed_trajectories = {}
    keys = list(runs.keys())

    for key in keys:
        raw_history = runs[key]["weight_history"]
        if torch.is_tensor(raw_history[0]):
            W_traj = torch.stack(raw_history).detach().cpu().numpy()
        else:
            W_traj = np.asarray(raw_history)
        centered = W_traj - W_traj[0:1, :]
        smoothed_trajectories[key] = coarse_grain(centered, bin_size=bin_size)

    all_data = np.concatenate([smoothed_trajectories[k] for k in keys], axis=0)
    _, pca = pca_reduce(
        all_data, n_components=n_components, return_pca=True, verbose=False
    )

    pc_trajectories = {}
    for key in keys:
        pc_trajectories[key] = pca.transform(smoothed_trajectories[key])

    return pc_trajectories, pca


def delta_w(run):
    wh = run["weight_history"]
    start = np.asarray(wh[0], dtype=np.float64).ravel()
    end = np.asarray(wh[-1], dtype=np.float64).ravel()
    return end - start


def _delta_ws_by_rule(runs):
    """Group raw ΔW vectors by rule: {rule: array of shape (n_seeds, d)}."""
    grouped = defaultdict(list)
    for (rule, seed), run in runs.items():
        grouped[rule].append(delta_w(run))
    return {rule: np.array(vs) for rule, vs in grouped.items()}


def build_gram_matrices(units_by_rule):
    """Cosine (Gram) matrices for every unordered rule pair, keyed by sorted
    pair (any number of rules): (rule, rule) is the within-seed Gram,
    (rule_a, rule_b) the cross-rule one. Inputs are unit vectors, so each entry
    is a cosine.
    """
    rules = sorted(units_by_rule)
    grams = {}
    for i, rule_a in enumerate(rules):
        Xa = units_by_rule[rule_a]
        for rule_b in rules[i:]:
            Xb = units_by_rule[rule_b]
            grams[(rule_a, rule_b)] = Xa @ Xb.T
    return grams


def gram_cosines(gram_matrices):
    """Mean pairwise cosine per rule pair. Within-rule reads the strict upper
    triangle (drops the self-cosine diagonal); between-rule averages all entries.
    Keyed like summarize_within_between, so the permutation test consumes both.
    """
    cosines = {}
    for (rule_a, rule_b), G in gram_matrices.items():
        if rule_a == rule_b:
            iu = np.triu_indices(len(G), k=1)
            cosines[(rule_a, rule_b)] = (
                float(G[iu].mean()) if iu[0].size else float("nan")
            )
        else:
            cosines[(rule_a, rule_b)] = float(G.mean())
    return cosines


def participation_ratios(gram_matrices):
    """Participation ratio PR = (sum lambda)^2 / sum(lambda^2) of each within-rule
    Gram's eigenvalues: PR ~ 1 is one shared direction (corridor), PR ~ n_seeds
    is an isotropic starburst. Drops roundoff eigenvalues; between-rule pairs have
    no PR.
    """
    prs = {}
    for (rule_a, rule_b), G in gram_matrices.items():
        if rule_a == rule_b:
            lam = np.linalg.eigvalsh(G)
            lam = lam[lam > 1e-12]
            prs[rule_a] = float(lam.sum() ** 2 / np.sum(lam**2))
    return prs


def delta_w_geometry(runs):
    """Across-seed geometry of ΔW = W_final - W_init, per rule. Splits the
    corridor question into length (‖ΔW‖ mean/std) and direction (mean pairwise
    cosine + PR of the unit ΔW set). Raw grouped values; the asymmetry and its
    permutation test come downstream.
    """
    raw = _delta_ws_by_rule(runs)
    norms, units = {}, {}
    for rule, X in raw.items():
        rule_norms = np.linalg.norm(X, axis=1)
        norms[rule] = {"mean": float(rule_norms.mean()), "std": float(rule_norms.std())}
        units[rule] = X / rule_norms[:, None]
    grams = build_gram_matrices(units)
    return {
        "norms": norms,
        "cosine": gram_cosines(grams),
        "pr": participation_ratios(grams),
    }


def delta_w_cosine_matrix(runs):
    """Pairwise cosine matrix across all runs' unit ΔW vectors with RULE-SEED labels."""
    labels = []
    U = []
    for (rule, seed), run in runs.items():
        vec = delta_w(run)
        unit_vec = vec / np.linalg.norm(vec)
        U.append(unit_vec)
        labels.append(f"{rule}-{seed}")
    U = np.array(U)
    M = U @ U.T
    return M, labels


def learning_subspace(run, k, bin_size=10, return_spectrum=False):
    """Orthonormal basis (d, k) for the subspace one seed's weight trajectory explored.

    Centers on W(0), coarse-grains, keeps the top-k right singular vectors. Per-seed, not
    the shared frame of process_weight_trajectories: principal angles need each seed's own
    subspace. Caps k at the trajectory's rank.

    return_spectrum also returns the cumulative variance curve, so a caller can report how
    much of the trajectory the k it chose covers without redoing the SVD.
    """
    raw_history = run["weight_history"]
    if torch.is_tensor(raw_history[0]):
        W_traj = torch.stack(raw_history).detach().cpu().numpy()
    else:
        W_traj = np.asarray(raw_history)
    centered = W_traj - W_traj[0:1, :]
    smoothed_trajectory = coarse_grain(centered, bin_size=bin_size)
    _, s, Vt = np.linalg.svd(smoothed_trajectory, full_matrices=False)
    k = min(k, Vt.shape[0])
    if return_spectrum:
        return Vt[:k, :].T, np.cumsum(s**2) / np.sum(s**2)
    return Vt[:k, :].T


def principal_angles(Q_a, Q_b):
    """Principal angles (radians, smallest first) between two subspaces.

    Singular values of Q_a.T @ Q_b are their cosines; clip guards arccos at 1+eps. Angles
    lie in [0, pi/2] since subspaces are undirected: ~0 shared, ~pi/2 orthogonal.
    """
    _, sigma, _ = np.linalg.svd(Q_a.T @ Q_b)
    theta = np.arccos(np.clip(sigma, 0, 1))
    return theta


def grassmann_distance(theta, metric="geodesic"):
    """One scalar distance on the Grassmannian from a principal-angle spectrum.

    geodesic = ||theta|| (arc length); chordal = ||sin theta|| (= ||P_a - P_b||_F / sqrt(2)).
    Both are proper metrics, unlike the mean angle, so they behave in the permutation test.
    """
    if metric == "geodesic":
        return np.linalg.norm(theta)
    elif metric == "chordal":
        return np.linalg.norm(np.sin(theta))
    else:
        raise ValueError(f"Unrecognized distance metric {metric}")


def subspace_distance_matrix(runs, k, metric="geodesic", bin_size=10):
    """Pairwise Grassmann-distance matrix across runs, labeled "rule-seed".

    Feeds summarize_within_between like delta_w_cosine_matrix, but as a distance, so
    canalization shows as small within-group values (the cosine version had them large).
    Gauge caveat: cross-seed subspaces sit in different init gauges, clean only in a
    shared gauge or after permutation alignment.
    """
    keys = list(runs.keys())
    Qs = [learning_subspace(runs[key], k, bin_size) for key in keys]
    labels = [f"{rule}-{seed}" for (rule, seed) in keys]  # keys are (rule, seed) tuples

    n = len(keys)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = grassmann_distance(principal_angles(Qs[i], Qs[j]), metric)
            M[i, j] = M[j, i] = d
    return M, labels


def random_subspace_distance(d, k, metric="geodesic", n_pairs=200, seed=0):
    """Mean/std Grassmann distance between random k-subspaces of R^d: the null.

    Independent subspaces are near-orthogonal in high d, so this is the "no corridor"
    baseline: a within-group distance only reads as canalized if it sits well below it.
    """
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_pairs):
        Qa = np.linalg.qr(rng.standard_normal((d, k)))[0]
        Qb = np.linalg.qr(rng.standard_normal((d, k)))[0]
        dists.append(grassmann_distance(principal_angles(Qa, Qb), metric))
    return float(np.mean(dists)), float(np.std(dists))
