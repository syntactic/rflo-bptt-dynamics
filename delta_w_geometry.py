"""Across-seed geometry of the recurrent weight displacement dW = W_final - W_init.

Do independently seeded runs of a rule move along a shared direction in weight space,
or in mutually orthogonal ones? Reports ||dW||, the mean pairwise cosine (chance in d
dimensions is 0 with std 1/sqrt(d)), and the participation ratio of the dW set
(PR ~ 1 is a single shared direction, PR ~ n_seeds is isotropic).

DSA is deliberately not used here: a weight trajectory over training is a
non-autonomous optimizer transient, not an autonomous flow. Independently initialized
seeds also label their hidden units differently, so these numbers are read as a
same-rule contrast, not as absolute cross-seed geometry.
"""

import argparse

import numpy as np

from analysis import load_experiment_run


def delta_w(run):
    wh = run["weight_history"]
    w0 = np.asarray(wh[0], dtype=np.float64).ravel()
    wl = np.asarray(wh[-1], dtype=np.float64).ravel()
    return wl - w0


def participation_ratio(vectors, normalize=True):
    """PR of the un-centered Gram of a stack of vectors (rows = seeds).

    normalize=True unit-scales each vector first, so PR reflects direction spread
    rather than length spread (a corridor is about shared direction)."""
    X = np.asarray(vectors, dtype=np.float64)
    if normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T  # (n, n) Gram; eigenvalues are the squared singular values of X
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-12]
    return float(lam.sum() ** 2 / np.sum(lam**2))


def mean_pairwise_cosine(vectors):
    X = np.asarray(vectors, dtype=np.float64)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = U @ U.T
    iu = np.triu_indices(len(X), k=1)
    return float(C[iu].mean()), float(C[iu].std())


def between_rule_cosine(a, b):
    Ua = a / np.linalg.norm(a, axis=1, keepdims=True)
    Ub = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((Ua @ Ub.T).mean())


def main():
    p = argparse.ArgumentParser(description="Across-seed Delta W corridor geometry.")
    p.add_argument("--dir", required=True)
    p.add_argument("--effector", default="ReluPointMass24")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(15)))
    p.add_argument("--rules", nargs="+", default=["BPTT", "RFLO"])
    args = p.parse_args()

    print(f"Delta W across-seed geometry: {args.dir} ({args.effector})")
    dw = {}
    for rule in args.rules:
        vecs = []
        for s in args.seeds:
            try:
                vecs.append(
                    delta_w(load_experiment_run(args.dir, args.effector, s, rule))
                )
            except FileNotFoundError:
                continue
        vecs = np.stack(vecs)
        dw[rule] = vecs
        norms = np.linalg.norm(vecs, axis=1)
        cos_m, cos_s = mean_pairwise_cosine(vecs)
        pr = participation_ratio(vecs, normalize=True)
        dim = vecs.shape[1]
        angle = np.degrees(np.arccos(np.clip(cos_m, -1, 1)))
        print(
            f"\n{rule}  (n={len(vecs)}, dim={dim}, chance cosine std=1/sqrt(d)={1 / np.sqrt(dim):.4f})"
        )
        print(f"  ||dW||          mean {norms.mean():.4f}  std {norms.std():.4f}")
        print(
            f"  mean pairwise cosine  {cos_m:+.4f} +- {cos_s:.4f}   (angle {angle:.1f} deg)"
        )
        print(
            f"  participation ratio   {pr:.2f}  of {len(vecs)}  (PR~1 corridor, PR~n isotropic)"
        )
    if len(args.rules) == 2 and all(r in dw for r in args.rules):
        bc = between_rule_cosine(dw[args.rules[0]], dw[args.rules[1]])
        print(
            f"\nBetween {args.rules[0]}-{args.rules[1]} mean cosine  {bc:+.4f}"
            f"   (angle {np.degrees(np.arccos(np.clip(bc, -1, 1))):.1f} deg)"
        )


if __name__ == "__main__":
    main()
