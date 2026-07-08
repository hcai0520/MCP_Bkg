from sklearn.cluster import DBSCAN
import numpy as np

def compute_cluster_features(points):
    center = points.mean(axis=0)
    nhit = len(points)

    dx = points[:, 0].max() - points[:, 0].min()
    dy = points[:, 1].max() - points[:, 1].min()
    dz = points[:, 2].max() - points[:, 2].min()

    if nhit < 2:
        direction_pair = np.array([0.0, 0.0, 1.0], dtype=float)
        pair_length = 0.0
        width_rms = 0.0
        width_max = 0.0
        aspect_ratio = 0.0
        p1 = center.copy()
        p2 = center.copy()
    else:
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        p1 = points[i]
        p2 = points[j]

        vec = p2 - p1
        pair_length = np.linalg.norm(vec)
        direction_pair = vec / (pair_length + 1e-12)

        v = points - p1
        proj_len = v @ direction_pair
        proj_vec = np.outer(proj_len, direction_pair)
        perp_vec = v - proj_vec
        perp_dist = np.linalg.norm(perp_vec, axis=1)

        width_rms = np.sqrt(np.mean(perp_dist**2)) if len(perp_dist) > 0 else 0.0
        width_max = np.max(perp_dist) if len(perp_dist) > 0 else 0.0

        if nhit < 4 or width_rms < 0.05:
            aspect_ratio = 0.0
        else:
            aspect_ratio = pair_length / width_rms

    return {
        "center": center,
        "nhit": int(nhit),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "pair_length": float(pair_length),
        "direction_pair": direction_pair,
        "width_rms": float(width_rms),
        "width_max": float(width_max),
        "aspect_ratio": float(aspect_ratio),
        "p1": p1,
        "p2": p2,
    }

def classify_cluster(feat):
    L = feat["pair_length"]
    A = feat["aspect_ratio"]

    if L > 10:
        return "track"
    if L > 3 and A >= 3:
        return "track"
    if L > 3 and A < 3:
        return "shower"
    return "normal"

DBSCAN_EPS = 2.     # cm
DBSCAN_MIN_SAMPLES = 1
def run_dbscan(hits):
    coords = np.c_[hits["x"], hits["y"], hits["z"]]
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
    return labels

    
def build_clusters(hits):
    coords = np.c_[hits["x"], hits["y"], hits["z"]]
    labels = run_dbscan(hits)

    clusters = []

    for cid in np.unique(labels):
        if cid == -1:
            continue

        mask = labels == cid

        pts = coords[mask]
        hits_cluster = hits[mask]  

        feat = compute_cluster_features(pts)
        label = classify_cluster(feat)

        clusters.append({
            "cluster_id": int(cid),
            "points": pts,
            "hits": hits_cluster,
            "label": label,
            "center": feat["center"],
            "nhit": feat["nhit"],
            "dx": feat["dx"],
            "dy": feat["dy"],
            "dz": feat["dz"],
            "pair_length": feat["pair_length"],
            "direction_pair": feat["direction_pair"],
            "width_rms": feat["width_rms"],
            "aspect_ratio": feat["aspect_ratio"],
            "p1": feat["p1"],
            "p2": feat["p2"],
        })

    return clusters, labels

def assign_cluster_type(clusters):
    track_idx = 0
    shower_idx = 0
    normal_idx = 0

    for c in clusters:
        if c["label"] == "track":
            c["type"] = f"track_{track_idx}"
            track_idx += 1
        elif c["label"] == "shower":
            c["type"] = f"shower_{shower_idx}"
            shower_idx += 1
        else:
            c["type"] = f"normal_{normal_idx}"
            normal_idx += 1

    return clusters


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