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
