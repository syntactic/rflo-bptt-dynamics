"""Analysis library, organized by measurement channel.

The math is split across four channels plus shared plumbing, mirroring the
paper's structure:

- ``io``         -- artifact loading and per-direction H reshaping.
- ``behavior``   -- reach-outcome metrics (concrete channel).
- ``activation`` -- hidden-state PCA + DSA (primary, gauge-invariant channel).
- ``weights``    -- ΔW corridor geometry (corroboration, gauge-safe frame only).
- ``stats``      -- grouped dispersion summaries and permutation tests.

Public names are re-exported here so ``from analysis import <name>`` and
``analysis.<name>`` keep working; the channel modules are the place to add new
functions. Import a submodule directly (``from analysis.behavior import ...``)
when you want only one channel and its dependencies.
"""

from .activation import (
    plot_dsa_heatmap,
    process_activation_trajectories,
    run_dsa,
)
from .behavior import (
    per_direction_metrics,
    perp_dist,
    reach_distance,
    reach_metrics,
    reach_relative_terminal_error,
    terminal_error,
)
from .io import (
    extract_per_direction_trajectories,
    load_all_runs,
    load_experiment_run,
)
from .stats import (
    _seed_to_pair,
    _within_mean,
    calculate_p_value_of_dsa_distance,
    calculate_stats_foreach_grouping,
    holm_bonferroni,
    paired_permutation_test,
    summarize_within_between,
)
from .weights import (
    build_gram_matrices,
    delta_w,
    delta_w_cosine_matrix,
    delta_w_geometry,
    gram_cosines,
    grassmann_distance,
    learning_subspace,
    participation_ratios,
    principal_angles,
    process_weight_trajectories,
    random_subspace_distance,
    subspace_distance_matrix,
)

__all__ = [
    # io
    "load_experiment_run",
    "load_all_runs",
    "extract_per_direction_trajectories",
    # behavior
    "perp_dist",
    "reach_distance",
    "terminal_error",
    "reach_relative_terminal_error",
    "reach_metrics",
    "per_direction_metrics",
    # activation
    "process_activation_trajectories",
    "run_dsa",
    "plot_dsa_heatmap",
    # weights
    "delta_w",
    "build_gram_matrices",
    "gram_cosines",
    "participation_ratios",
    "delta_w_geometry",
    "delta_w_cosine_matrix",
    "learning_subspace",
    "principal_angles",
    "grassmann_distance",
    "subspace_distance_matrix",
    "random_subspace_distance",
    "process_weight_trajectories",
    # stats
    "summarize_within_between",
    "calculate_stats_foreach_grouping",
    "calculate_p_value_of_dsa_distance",
    "paired_permutation_test",
    "holm_bonferroni",
]
