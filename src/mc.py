import numpy as np


def build_doublet_arrays_from_cluster_pairs(cluster_pairs, min_dist=10.0, sort_by_z=True):
    """
    Build per-candidate arrays from event-level two-cluster pairs.

    Important: this stores `event_key`, which is the join key used by the
    public blinding mask for observed doublets.
    """
    event_key = []
    dx, dy, dz = [], [], []
    theta_z, theta_zx, theta_zy = [], [], []
    e0, e1, esum, eabsdiff = [], [], [], []
    npts0, npts1 = [], []

    for c0, c1 in cluster_pairs:
        p0 = np.array([c0["x"], c0["y"], c0["z"]], dtype=float)
        p1 = np.array([c1["x"], c1["y"], c1["z"]], dtype=float)

        if sort_by_z and (p1[2] < p0[2]):
            p0, p1 = p1, p0
            c0, c1 = c1, c0

        d = p1 - p0
        dist = np.linalg.norm(d)
        if dist < float(min_dist):
            continue

        dx.append(d[0])
        dy.append(d[1])
        dz.append(d[2])

        theta_z.append(np.arccos(np.clip(d[2] / dist, -1.0, 1.0)))
        theta_zx.append(np.arctan2(d[0], d[2]))
        theta_zy.append(np.arctan2(d[1], d[2]))

        ee0 = c0.get("energy", np.nan)
        ee1 = c1.get("energy", np.nan)

        e0.append(ee0)
        e1.append(ee1)
        esum.append(ee0 + ee1 if np.isfinite(ee0) and np.isfinite(ee1) else np.nan)
        eabsdiff.append(abs(ee1 - ee0) if np.isfinite(ee0) and np.isfinite(ee1) else np.nan)

        npts0.append(c0.get("npts", np.nan))
        npts1.append(c1.get("npts", np.nan))
        event_key.append(str(c0["event_key"]))

    return {
        "event_key": np.asarray(event_key, dtype=object),
        "dx": np.asarray(dx, dtype=float),
        "dy": np.asarray(dy, dtype=float),
        "dz": np.asarray(dz, dtype=float),
        "theta_z": np.asarray(theta_z, dtype=float),
        "theta_zx": np.asarray(theta_zx, dtype=float),
        "theta_zy": np.asarray(theta_zy, dtype=float),
        "e0": np.asarray(e0, dtype=float),
        "e1": np.asarray(e1, dtype=float),
        "esum": np.asarray(esum, dtype=float),
        "eabsdiff": np.asarray(eabsdiff, dtype=float),
        "npts0": np.asarray(npts0, dtype=float),
        "npts1": np.asarray(npts1, dtype=float),
    }


def monte_carlo_doublets_from_singles(
    single_clusters,
    n_accept=100_000,
    min_dist=10.0,
    sort_by_z=True,
    seed=1234,
    allow_same_source_event=True,
    max_trials_factor=50,
    return_metadata=False,
):
    rng = np.random.default_rng(seed)
    singles = list(single_clusters)
    if len(singles) < 2:
        raise ValueError("Need at least two single clusters to build a singles-driven MC.")

    accepted_pairs = []
    trials = 0
    max_trials = int(max_trials_factor * n_accept)

    while len(accepted_pairs) < n_accept and trials < max_trials:
        trials += 1
        i0 = rng.integers(0, len(singles))
        i1 = rng.integers(0, len(singles) - 1)
        if i1 >= i0:
            i1 += 1

        c0 = singles[i0]
        c1 = singles[i1]

        if (not allow_same_source_event) and (c0["event_key"] == c1["event_key"]):
            continue

        p0 = np.array([c0["x"], c0["y"], c0["z"]], dtype=float)
        p1 = np.array([c1["x"], c1["y"], c1["z"]], dtype=float)

        if np.linalg.norm(p1 - p0) < float(min_dist):
            continue

        accepted_pairs.append((c0, c1))

    if len(accepted_pairs) < n_accept:
        print(f"Warning: requested {n_accept} accepted singles-MC doublets, got {len(accepted_pairs)}")

    arrays = build_doublet_arrays_from_cluster_pairs(
        accepted_pairs,
        min_dist=min_dist,
        sort_by_z=sort_by_z,
    )

    meta = {
        "n_requested_accept": int(n_accept),
        "n_trials": int(trials),
        "n_accepted_min_dist": int(len(accepted_pairs)),
        "min_dist_acceptance": float(len(accepted_pairs) / trials) if trials > 0 else np.nan,
        "allow_same_source_event": bool(allow_same_source_event),
        "seed": int(seed),
        "max_trials": int(max_trials),
    }

    if return_metadata:
        return arrays, meta

    return arrays


def expected_doublets_from_singles(summary, singles_mc_meta, *, roi_keep_fraction=1.0):
    """
    Estimate the expected number of accepted combinatorial doublets from the single-blip pool.

    The normalization uses only:
      - number of analyzed event windows
      - number of observed single-blip events
      - min-distance acceptance from the singles-driven MC
      - ROI keep fraction from applying the same blinding ROI to the MC

    It does NOT use observed doublet counts or sideband normalization.

    For low occupancy, lambda ~= N_single / N_events and
        N_doublet ~= N_events * lambda^2 / 2.
    """
    n_events = int(len(summary["multiplicities"]))
    n_single = int(len(summary["singles"]))

    if n_events <= 0:
        raise ValueError("No events found in summary.")

    lam_from_singles = n_single / n_events
    n_doublet_before_pair_cuts = 0.5 * n_events * lam_from_singles * lam_from_singles

    min_dist_acceptance = float(singles_mc_meta.get("min_dist_acceptance", np.nan))
    if not np.isfinite(min_dist_acceptance):
        min_dist_acceptance = 1.0

    expected_after_min_dist = n_doublet_before_pair_cuts * min_dist_acceptance
    expected_after_roi = expected_after_min_dist * float(roi_keep_fraction)

    return {
        "n_events": n_events,
        "n_single": n_single,
        "lambda_from_singles": float(lam_from_singles),
        "n_doublet_before_pair_cuts": float(n_doublet_before_pair_cuts),
        "min_dist_acceptance": float(min_dist_acceptance),
        "roi_keep_fraction": float(roi_keep_fraction),
        "expected_after_min_dist": float(expected_after_min_dist),
        "expected_after_roi": float(expected_after_roi),
    }