"""Read the data release in data/ and build its summary tables.

Needs only numpy and pandas, so it runs from a clone without the run artifacts.
export_data.py calls build_summaries() after writing the per-run and pairwise files, so
the summary tables are always derived from them.

    import data_release as dr
    runs = dr.load_runs()                       # one row per run
    arrays = dr.load_arrays("arm_fixB_40k")     # FT, H, weights, B for one group
    bases = dr.load_subspaces("arm_fixB_40k")   # learning-subspace bases per window
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
ARM, PM = "RigidTendonArm26", "ReluPointMass24"

# Arm seeds converged in every 40k condition (the paired roster).
COMMON17 = [1, 3, 4, 5, 8, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 24]
# multiB nets. b101 net 4 fails the convergence gate, so GRID4 is the converged block.
GRID_NETS = [1, 3, 4, 5, 8]
GRID4 = [1, 3, 5, 8]
LONG8 = [1, 3, 4, 5, 8, 9, 10, 11]  # BPTT, projB0 and fixB at 100k
LONG4 = [1, 3, 9, 10]  # fixB at 300k

N_DIRECTIONS = 8
ERROR_COLUMNS = [f"err_d{d}" for d in range(N_DIRECTIONS)]
# Direction k points at 45k degrees, counter-clockwise from +x (MotorNet CenterOutReach).
DIRECTION_DEG = [int(360 * d / N_DIRECTIONS) for d in range(N_DIRECTIONS)]

K_PUBLISHED = 3


def direction_label(d):
    """'0 (0°)', '1 (45°)', ..."""
    return f"{d} ({DIRECTION_DEG[d]}°)"


# --- readers ----------------------------------------------------------------------------


def load_runs(data_dir=DATA):
    return pd.read_csv(Path(data_dir) / "tables" / "runs.csv")


def load_loss_curves(data_dir=DATA):
    return pd.read_csv(Path(data_dir) / "tables" / "loss_curves.csv")


def load_windows(data_dir=DATA):
    return pd.read_csv(Path(data_dir) / "tables" / "subspace_windows.csv")


def load_pairwise_subspace(data_dir=DATA):
    return pd.read_csv(Path(data_dir) / "pairwise" / "subspace_k3.csv")


def load_pairwise_dsa(data_dir=DATA):
    return pd.read_csv(Path(data_dir) / "pairwise" / "activation_dsa.csv")


def load_csv_summary(name, data_dir=DATA):
    """One of the summary/*.csv tables, e.g. 'multib_grid.csv'."""
    return pd.read_csv(Path(data_dir) / "summary" / name)


def _load_npz(path):
    """All arrays in one file, with string arrays as Python strings."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    for key, value in d.items():
        if value.dtype.kind == "U":
            d[key] = str(value) if value.ndim == 0 else [str(x) for x in value]
    return d


def load_arrays(group, data_dir=DATA):
    """Per-run arrays for one group, e.g. 'arm_fixB_40k'; rows follow d['run_id']."""
    return _load_npz(Path(data_dir) / "arrays" / f"runs_{group}.npz")


def load_subspaces(group, data_dir=DATA):
    """Learning-subspace bases for one group; rows follow d['window_id']."""
    return _load_npz(Path(data_dir) / "arrays" / f"subspace_{group}.npz")


def array_groups(kind, data_dir=DATA):
    """Groups that have a `kind` ('runs' or 'subspace') array file."""
    files = sorted((Path(data_dir) / "arrays").glob(f"{kind}_*.npz"))
    return [f.stem[len(kind) + 1 :] for f in files]


def all_bases(data_dir=DATA):
    """{window_id: (4096, k_stored) basis} across every subspace file."""
    out = {}
    for group in array_groups("subspace", data_dir):
        d = load_subspaces(group, data_dir)
        out.update(zip(d["window_id"], d["basis"]))
    return out


# --- math -------------------------------------------------------------------------------


def principal_angles(Qa, Qb):
    """Principal angles (radians) between the column spans of two orthonormal bases."""
    sigma = np.linalg.svd(Qa.T.astype(float) @ Qb.astype(float), compute_uv=False)
    return np.arccos(np.clip(sigma, 0, 1))


def geodesic(theta):
    return float(np.linalg.norm(theta))


def chordal(theta):
    return float(np.linalg.norm(np.sin(theta)))


def angle_deg(geodesic_distance, k=K_PUBLISHED):
    """Geodesic distance as an RMS principal angle in degrees, comparable across k."""
    return float(np.degrees(geodesic_distance / np.sqrt(k)))


def random_null_deg(d=64 * 64, k=K_PUBLISHED, n_pairs=200, seed=0):
    """Mean angle between random k-subspaces of R^d, as in analysis.random_subspace_distance."""
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_pairs):
        Qa = np.linalg.qr(rng.standard_normal((d, k)))[0]
        Qb = np.linalg.qr(rng.standard_normal((d, k)))[0]
        dists.append(geodesic(principal_angles(Qa, Qb)))
    return angle_deg(np.mean(dists), k)


def within_mean(pairs, ids, col):
    """Mean of `col` over pairs with both ends in ids, and the number of pairs."""
    ids = set(ids)
    mask = pairs.iloc[:, 0].isin(ids) & pairs.iloc[:, 1].isin(ids)
    return float(pairs.loc[mask, col].mean()), int(mask.sum())


def between_mean(pairs, ids_a, ids_b, col):
    a, b = set(ids_a), set(ids_b)
    i, j = pairs.iloc[:, 0], pairs.iloc[:, 1]
    mask = (i.isin(a) & j.isin(b)) | (i.isin(b) & j.isin(a))
    return float(pairs.loc[mask, col].mean()), int(mask.sum())


# --- summary tier -----------------------------------------------------------------------


def select(runs, effector, condition, budget, seeds=None):
    sel = runs[
        (runs.effector == effector)
        & (runs.condition == condition)
        & (runs.budget == budget)
    ]
    if seeds is not None:
        sel = sel[sel.seed.isin(seeds)]
    return sel.sort_values("seed")


def _behavior_rows(runs):
    """Mean error and pole counts per (effector, condition, budget, roster)."""
    plan = [(PM, c, 70000, "all15", list(range(15))) for c in ("BPTT", "fixB")]
    plan += [(ARM, c, 40000, "COMMON17", COMMON17) for c in ("BPTT", "fixB", "varyB", "projB0")]
    plan += [(ARM, c, 40000, "LONG8", LONG8) for c in ("BPTT", "fixB")]
    plan += [(ARM, c, 100000, "LONG8", LONG8) for c in ("BPTT", "fixB", "projB0")]
    plan += [(ARM, c, 40000, "LONG4", LONG4) for c in ("BPTT", "fixB")]
    plan += [(ARM, "fixB", 100000, "LONG4", LONG4), (ARM, "fixB", 300000, "LONG4", LONG4)]
    plan += [(ARM, c, 40000, "GRID5", GRID_NETS) for c in ("fixB", "varyB", "b101", "b102", "b103")]
    rows = []
    for effector, condition, budget, roster, seeds in plan:
        sel = select(runs, effector, condition, budget, seeds)
        ev = sel[sel.has_eval]
        if not len(sel):
            continue
        rows.append(
            dict(
                effector=effector,
                condition=condition,
                budget=budget,
                roster=roster,
                n_runs=len(sel),
                n_with_eval=len(ev),
                n_converged=int(sel.converged.sum()),
                mean_error=ev.mean_error.mean() if len(ev) else np.nan,
                sd_error=ev.mean_error.std(ddof=1) if len(ev) > 1 else np.nan,
                mean_spread=ev.spread.mean() if len(ev) else np.nan,
                n_worst_d0=int((ev.worst_dir == 0).sum()),
                n_worst_d4=int((ev.worst_dir == 4).sum()),
                mean_final_loss=sel.final_loss.mean(),
            )
        )
    return pd.DataFrame(rows)


def _corridor_rows(runs, windows, pairs):
    """Within-condition learning-subspace distance (k=3) per condition and window."""
    w = windows.merge(runs[["run_id", "effector", "condition", "budget", "seed"]], on="run_id")
    plan = [(PM, c, 0, 70000, "all15", list(range(15))) for c in ("BPTT", "fixB")]
    plan += [(ARM, c, 0, 40000, "COMMON17", COMMON17) for c in ("BPTT", "fixB", "varyB", "projB0")]
    plan += [
        (ARM, c, 0, 40000, "GRID4", GRID4)
        for c in ("fixB", "varyB", "BPTT", "b101", "b102", "b103")
    ]
    plan += [
        (ARM, c, start, 100000, "LONG8_wh", None)
        for c in ("fixB", "projB0")
        for start in (0, 40000)
    ]
    rows = []
    for effector, condition, start, stop, roster, seeds in plan:
        sel = w[
            (w.effector == effector)
            & (w.condition == condition)
            & (w.step_start == start)
            & (w.step_stop == stop)
        ]
        if seeds is not None:
            sel = sel[sel.seed.isin(seeds)]
        if len(sel) < 2:
            continue
        geo, n_pairs = within_mean(pairs, sel.window_id, "geodesic_k3")
        cho, _ = within_mean(pairs, sel.window_id, "chordal_k3")
        rows.append(
            dict(
                effector=effector,
                condition=condition,
                steps=f"{start}-{stop}",
                roster=roster,
                n_runs=len(sel),
                n_pairs=n_pairs,
                within_geodesic_k3=geo,
                within_chordal_k3=cho,
                within_angle_deg=angle_deg(geo),
                mean_share_dW_rec_in_span_B0=sel.share_dW_rec_in_span_B0.mean(),
                min_share_dW_rec_in_span_B0=sel.share_dW_rec_in_span_B0.min(),
                max_share_dW_rec_in_span_B0=sel.share_dW_rec_in_span_B0.max(),
            )
        )
    rows.append(
        dict(
            effector="any (d = 4096)",
            condition="random 3-subspaces",
            steps="",
            roster="null",
            n_runs=0,
            n_pairs=200,
            within_angle_deg=random_null_deg(),
        )
    )
    return pd.DataFrame(rows)


def _dsa_rows(runs, dsa):
    """Within/between activation-DSA means per source matrix and condition pair."""
    cond = runs.set_index("run_id").condition
    rows = []
    for source, block in dsa.groupby("source_matrix", sort=False):
        pair = [tuple(sorted(p)) for p in zip(block.run_i.map(cond), block.run_j.map(cond))]
        block = block.assign(pair=pair)
        for (ca, cb), sub in block.groupby("pair", sort=False):
            rows.append(
                dict(
                    source_matrix=source,
                    condition_a=ca,
                    condition_b=cb,
                    n_pairs=len(sub),
                    mean_distance=sub.distance.mean(),
                )
            )
    return pd.DataFrame(rows)


def _multib_grid(runs):
    """Worst direction per net x B on the multiB grid (B0 = fixB, own B = varyB)."""
    labels = {"fixB": "B0", "b101": "B101", "b102": "B102", "b103": "B103", "varyB": "own"}
    rows = []
    for condition, label in labels.items():
        for _, r in select(runs, ARM, condition, 40000, GRID_NETS).iterrows():
            rows.append(
                dict(
                    net=int(r.seed),
                    B=label,
                    run_id=r.run_id,
                    converged=bool(r.converged),
                    worst_dir=int(r.worst_dir),
                    mean_error=r.mean_error,
                    spread=r.spread,
                )
            )
    return pd.DataFrame(rows)


def build_summaries(data_dir=DATA):
    """Write summary/*.csv from the per-run and pairwise tiers already on disk."""
    data_dir = Path(data_dir)
    runs = load_runs(data_dir)
    out = data_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "behavior.csv": _behavior_rows(runs),
        "corridor.csv": _corridor_rows(
            runs, load_windows(data_dir), load_pairwise_subspace(data_dir)
        ),
        "activation_dsa.csv": _dsa_rows(runs, load_pairwise_dsa(data_dir)),
        "multib_grid.csv": _multib_grid(runs),
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format="%.6g")
    return tables
