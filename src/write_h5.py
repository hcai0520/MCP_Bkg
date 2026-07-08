from collections import defaultdict
import numpy as np
import h5py

def save_normal_hits(all_clusters, out_path, n_events):
    """
    Save normal, non-removed cluster hits into an HDF5 file with ref_region.

    Output structure:
        normal_hits/data
        normal_hits/ref_region
    """

    hit_dtype = np.dtype([
        ("evt_index",   "i4"),
        ("cluster_id",  "i4"),
        ("beam_type",   "i1"),
        ("x",           "f4"),
        ("y",           "f4"),
        ("z",           "f4"),
        ("Q",           "f4"),
        ("io_group",    "i4"),
        ("io_channel",  "i4"),
        ("chip_id",     "i4"),
        ("channel_id",  "i4"),
    ])

    ref_region_dtype = np.dtype([
        ("start", "i8"),
        ("stop",  "i8"),
    ])

    event_hits = defaultdict(list)

    for c in all_clusters:

        if c.get("label") != "normal":
            continue

        if c.get("type") == "removed":
            continue

        hits = c["hits"]

        if len(hits) == 0:
            continue

        arr = np.zeros(len(hits), dtype=hit_dtype)

        arr["evt_index"]  = c["evt_idx"]
        arr["cluster_id"] = c["cluster_id"]
        arr["beam_type"]  = c["beam_type"]

        arr["x"] = hits["x"]
        arr["y"] = hits["y"]
        arr["z"] = hits["z"]
        arr["Q"] = hits["Q"]

        arr["io_group"]   = hits["io_group"]
        arr["io_channel"] = hits["io_channel"]
        arr["chip_id"]    = hits["chip_id"]
        arr["channel_id"] = hits["channel_id"]

        event_hits[c["evt_idx"]].append(arr)

    all_hits = []
    ref_regions = []

    start = 0

    for evt_idx in range(n_events):

        if len(event_hits[evt_idx]) > 0:
            hits_evt = np.concatenate(event_hits[evt_idx])
            stop = start + len(hits_evt)
            all_hits.append(hits_evt)
        else:
            stop = start

        ref_regions.append((start, stop))
        start = stop

    if len(all_hits) > 0:
        all_hits = np.concatenate(all_hits)
    else:
        all_hits = np.zeros(0, dtype=hit_dtype)

    ref_regions = np.array(ref_regions, dtype=ref_region_dtype)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("normal_hits/data", data=all_hits)
        f.create_dataset("normal_hits/ref_region", data=ref_regions)

    print("saved:", out_path)
    print("total events:", n_events)
    print("events with normal hits:", sum(r["stop"] > r["start"] for r in ref_regions))
    print("total normal hits:", len(all_hits))