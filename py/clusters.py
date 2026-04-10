import numpy as np
import h5py
from src.date_reading import *
from src.build_clusters import *
from src.remove_near_points import *
from src.Find_bad_region import *
from src.plot import *

path = "data/packet-0060070-2025_10_31_15_57_27_CDT.FLOW.hdf5"
hot_regions = [
    {"z_range": [24.20, 27.30], "y_range": [-34.12, -31.02]},
    {"z_range": [18.20, 21.10], "y_range": [-43.43, -40.33]},
    {"z_range": [27.30, 30.40], "y_range": [34.45, 37.23]},

    {"z_range": [-21.10, -18.00], "y_range": [-49.63, -46.53]},
    {"z_range": [-52.11, -49.00], "y_range": [3.10, 6.20]},
    {"z_range": [-28.63, -27.74], "y_range": [-23.04, -22.16]},
]
bad_pixels = [(2, 29, 28, 18)]

DBSCAN_EPS = 2.     # cm
DBSCAN_MIN_SAMPLES = 1

all_clusters = []
bad_region_hits = []
repeat_pixels=[]

with h5py.File(path, "r") as f:
#    n_events = 400
#    for evt_idx in range(n_events):
#    print(len(f["charge/events/data"]))
    for evt_idx in range(len(f["charge/events/data"])):
        hits = get_event_hits_by_event_index(f, evt_idx)
        #fiducal cut
        hits = hits_fiducal (hits)
     #  print(len(hits))
     # hits of bad region
        if len(hits) == 0:
            continue
        hits,bad_region = hits_bad_regions(hits, bad_pixels, hot_regions)
        if len(bad_region) > 0:
            bad_region_hits.append({"evt_idx": evt_idx,"hits": bad_region})
    #print(len(hits))
    # hits of repeat pixels
        if len(hits)==0:
            continue
        hits,repeat_pixel = repeated_pixel_hits(hits,yz_radius=0.5, x_min=5.0, min_count=3)
        if len(repeat_pixel) > 0:
            repeat_pixels.append({"evt_idx": evt_idx,"hits": repeat_pixel})
    #print(len(hits))
        if len(hits)== 0:
            continue
        
            #dbscan and label track and shower
        clusters, labels = build_clusters(hits,DBSCAN_EPS,DBSCAN_MIN_SAMPLES)


        clusters_keep,clusters_remove = remove_clusters_near_track_points(clusters,R_cut=15.0, target_labels=("track", "shower"))
        if len(clusters_keep) > 10:
            for c in clusters_keep:
                c["type"] = "removed"
            for c in clusters_remove:
                c["type"] = "removed"
            clusters_final = clusters_keep + clusters_remove
        else:
            clusters_keep = assign_cluster_type(clusters_keep)
            for c in clusters_remove:
                c["type"] = "removed"
            clusters_final = clusters_keep + clusters_remove
           
        for c in clusters_final:
            c["evt_idx"] = evt_idx
            all_clusters.append(c)

from collections import defaultdict

events = defaultdict(lambda: {
    "bad": [],
    "repeat": [],
    "clusters": []
})

# --- bad region hits ---
for item in bad_region_hits:
    evt_idx = item["evt_idx"]
    events[evt_idx]["bad"].append(item["hits"])

# --- repeat pixels ---
for item in repeat_pixels:
    evt_idx = item["evt_idx"]
    events[evt_idx]["repeat"].append(item["hits"])

# --- clusters ---
for c in all_clusters:
    evt_idx = c["evt_idx"]
    events[evt_idx]["clusters"].append(c)

events = dict(events)            

plot_event(10, events)