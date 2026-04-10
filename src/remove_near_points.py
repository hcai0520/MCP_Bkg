import numpy as np

def remove_clusters_near_track_points(clusters, R_cut=10.0, target_labels=("track", "shower")):
    if len(clusters) == 0:
        keep_clusters = []
        remove_clusters = []
        min_dists = np.array([], dtype=float)
        keep_mask = np.array([], dtype=bool)
        remove_mask = np.array([], dtype=bool)
        return keep_clusters, remove_clusters
    ref_points_list = []
    for c in clusters:
        if c["label"] in target_labels and len(c["points"]) > 0:
            ref_points_list.append(c["points"])

    if len(ref_points_list) == 0:
        keep_clusters = clusters[:]
        remove_clusters = []
        keep_mask = np.ones(len(clusters), dtype=bool)
        remove_mask = ~keep_mask
        min_dists = np.full(len(clusters), np.inf, dtype=float)
        return keep_clusters, remove_clusters

    ref_points = np.vstack(ref_points_list)   # (M,3)

    keep_mask = np.ones(len(clusters), dtype=bool)
    min_dists = np.full(len(clusters), np.inf, dtype=float)

    for i, c in enumerate(clusters):
        pts = c["points"]
        if len(pts) == 0:
            min_dists[i] = np.inf
            continue

        diff = pts[:, None, :] - ref_points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_d = dist.min()

        min_dists[i] = min_d

        if c["label"] in target_labels:
            keep_mask[i] = True
        else:
            keep_mask[i] = (min_d >= R_cut)

    remove_mask = ~keep_mask

    keep_clusters = [c for i, c in enumerate(clusters) if keep_mask[i]]
    remove_clusters = [c for i, c in enumerate(clusters) if remove_mask[i]]

    return keep_clusters, remove_clusters