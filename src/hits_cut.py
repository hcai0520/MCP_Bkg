from collections import defaultdict
import numpy as np

def cluster_center(c):
    pts = c["points"]

    # structured array: pts["x"], pts["y"], pts["z"]
    if pts.dtype.names is not None:
        return np.array([
            np.mean(pts["x"]),
            np.mean(pts["y"]),
            np.mean(pts["z"]),
        ])

    # normal numpy array: shape = (N, 3)
    pts = np.asarray(pts)
    return np.array([
        np.mean(pts[:, 0]),
        np.mean(pts[:, 1]),
        np.mean(pts[:, 2]),
    ])


    
def device_hits_cut(occ_dict, percent=0.99):
    """
    occ_dict:
        chip_occ or pixel_occupancy

    return：
        occ_left  :occupancy (numpy array)
        keep_keys : (set)
    """

    items = sorted(occ_dict.items(), key=lambda x: x[1])  

    k = int(len(items) * percent)

    left_items = items[:k]

    occ_left = np.array([v for _, v in left_items])
    keep_keys = {key for key, _ in left_items}

    print("total:", len(items))
    print("left :", k)
    print("removed:", len(items) - k)

    total_signal = sum(v for _, v in items)
    left_signal = occ_left.sum()
    removed_signal = total_signal - left_signal

    print("total hits:", total_signal)
    print("hits left:", left_signal)
    print("hits removed:", removed_signal)
    print(f"hits lost: {removed_signal / total_signal * 100:.2f}%")

    return occ_left, keep_keys
    

def hot_region_cut(hits, chip_keys, pixel_keys):
    """
    Remove hits belonging to removed chips/pixels.

    Parameters
    ----------
    hits : structured ndarray
    chip_keys : set of (io_group, io_channel, chip_id)
    pixel_keys : set of (io_group, io_channel, chip_id, channel_id)

    Returns
    -------
    kept_hits
    removed_hits
    """

    chip_keys = set(chip_keys)
    pixel_keys = set(pixel_keys)

    io   = hits["io_group"]
    ch   = hits["io_channel"]
    chip = hits["chip_id"]
    chan = hits["channel_id"]

    chip_mask = np.fromiter(
        (
            (int(io[i]), int(ch[i]), int(chip[i])) in chip_keys
            for i in range(len(hits))
        ),
        dtype=bool,
        count=len(hits),
    )

    pixel_mask = np.fromiter(
        (
            (int(io[i]), int(ch[i]), int(chip[i]), int(chan[i])) in pixel_keys
            for i in range(len(hits))
        ),
        dtype=bool,
        count=len(hits),
    )

    keep_mask = chip_mask & pixel_mask

    return hits[keep_mask], hits[~keep_mask]





    
def repeated_pixel_cut(hits, min_count=3):
    """
    Remove hits from pixels that fire >= min_count times in the same event.

    Pixel key:
        (io_group, io_channel, chip_id, channel_id)
    """
    n = len(hits)

    if n < min_count:
        return hits, hits[:0]

    pixel_count = defaultdict(int)

    for h in hits:
        key = (int(h["io_group"]),int(h["io_channel"]),int(h["chip_id"]),int(h["channel_id"]),)
        pixel_count[key] += 1

    bad_pixels = {
        key for key, cnt in pixel_count.items()
        if cnt >= min_count
    }

    remove_mask = np.zeros(n, dtype=bool)

    for i, h in enumerate(hits):
        key = (int(h["io_group"]),int(h["io_channel"]),int(h["chip_id"]),int(h["channel_id"]),)
        if key in bad_pixels:
            remove_mask[i] = True

    return hits[~remove_mask], hits[remove_mask]


def fiducial_cut(hits):
    good = (np.isfinite(hits["x"]) & np.isfinite(hits["y"]) & np.isfinite(hits["z"]) & np.isfinite(hits["Q"]) & (hits["Q"] > 0))
    hits = hits[good]
    # fiducial cut
    y = hits["y"]
    z = hits["z"]
    geom_mask = ((y >= -51.85) & (y <= 51.85) & (((z >= 12.68) & (z <= 54.32)) |((z >= -54.32) & (z <= -12.68))))
    return hits[geom_mask]

def two_blip_min_distance_cut(clusters_final, d_min=5.0):
    """
    If an event has exactly two kept blips, require their distance >= d_min.
    Otherwise remove the two blips from this event.
    """
    blips = [
        c for c in clusters_final
        if c.get("type", "") != "removed"
        and c.get("label") == "normal"
    ]
    if len(blips) != 2:
        return clusters_final
    p0 = cluster_center(blips[0])
    p1 = cluster_center(blips[1])
    dist = np.linalg.norm(p0 - p1)
    if dist < d_min:
        for c in blips:
            c["type"] = "removed"
    return clusters_final