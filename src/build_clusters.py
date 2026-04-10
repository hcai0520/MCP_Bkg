from dbscan import *
from classify_clusters import *

def build_clusters(hits,DBSCAN_EPS,DBSCAN_MIN_SAMPLES):
    coords = np.c_[hits["x"], hits["y"], hits["z"]]
    labels = run_dbscan(hits,DBSCAN_EPS,DBSCAN_MIN_SAMPLES)

    clusters = []

    for cid in np.unique(labels):
        if cid == -1:
            continue

        mask = (labels == cid)
        pts = coords[mask]

        feat = compute_cluster_features(pts)
        label = classify_cluster(feat)

        clusters.append({
            "cluster_id": int(cid),
            "points": pts,
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
            "label": label
        })

    return clusters, labels