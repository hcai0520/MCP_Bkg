from sklearn.cluster import DBSCAN
import numpy as np
def run_dbscan(hits,DBSCAN_EPS,DBSCAN_MIN_SAMPLES):
    coords = np.c_[hits["x"], hits["y"], hits["z"]]
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
    return labels


