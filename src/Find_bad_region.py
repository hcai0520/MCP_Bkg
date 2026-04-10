from date_reading import *
import h5py
import numpy as np


def hits_fiducal (hits):
    good = (np.isfinite(hits["x"]) & np.isfinite(hits["y"]) & np.isfinite(hits["z"]) & np.isfinite(hits["Q"]) & (hits["Q"] > 0))
    hits = hits[good]
    # fiducial cut
    y = hits["y"]
    z = hits["z"]
    geom_mask = ((y >= -51.85) & (y <= 51.85) & (
                ((z >= 12.68) & (z <= 54.32)) |
                ((z >= -54.32) & (z <= -12.68))
                ))
    return hits[geom_mask]

def find_bad_pixels(
    file,
    multiplicity_threshold=4,
    seen_threshold=200,
    abnormal_threshold=100
):
    pixel_seen = {}
    pixel_abnormal = {}

    with h5py.File(file, "r") as f:
        n_events = len(f["charge/events/data"])

        for evt_idx in range(n_events):

            hits_ev = get_event_hits_by_event_index(f, evt_idx)

            if len(hits_ev) == 0:
                continue

            counts = {}

            for i in range(len(hits_ev)):
                p = (
                    int(hits_ev["io_group"][i]),
                    int(hits_ev["io_channel"][i]),
                    int(hits_ev["chip_id"][i]),
                    int(hits_ev["channel_id"][i])
                )
                counts[p] = counts.get(p, 0) + 1

            for p, n in counts.items():
                pixel_seen[p] = pixel_seen.get(p, 0) + 1

                if n > multiplicity_threshold:
                    pixel_abnormal[p] = pixel_abnormal.get(p, 0) + 1

    bad_pixels = []

    for p in pixel_seen:
        seen = pixel_seen[p]
        abnormal = pixel_abnormal.get(p, 0)

        if seen > seen_threshold and abnormal > abnormal_threshold:
            bad_pixels.append((p, seen, abnormal))

    return bad_pixels


def find_hot_regions(z, y, hist_range, bins, hot_percentile=99):
    H, z_edges, y_edges = np.histogram2d(z, y, bins=bins, range=hist_range)

    if np.any(H > 0):
        thr = np.percentile(H[H > 0], hot_percentile)
    else:
        thr = np.inf
    hot = (H > thr) 
    visited = np.zeros_like(hot, dtype=bool)
    regions = []
    nz, ny = hot.shape

    for i in range(nz):
        for j in range(ny):
            if not hot[i, j] or visited[i, j]:
                continue

            stack = [(i, j)]
            visited[i, j] = True
            cells = []

            while stack:
                x, yy = stack.pop()#x,yy is the last element of stack and get removed
                cells.append((x, yy))

                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    xn, yn = x + dx, yy + dy
                    if 0 <= xn < nz and 0 <= yn < ny:
                        if hot[xn, yn] and not visited[xn, yn]:
                            visited[xn, yn] = True
                            stack.append((xn, yn))

            iz = [c[0] for c in cells]
            iy = [c[1] for c in cells]
            
            region = {
                "z_range": (z_edges[min(iz)], z_edges[max(iz) + 1]),
                "y_range": (y_edges[min(iy)], y_edges[max(iy) + 1]),
                "n_bins": len(cells),
                "count_sum": int(sum(H[a, b] for a, b in cells))
            }
            regions.append(region)
    regions = [
        r for r in regions
        if (r["n_bins"] >= 10) or (r["count_sum"] >= 20000)
        ]

    regions.sort(key=lambda r: r["count_sum"], reverse=True)
    return H, z_edges, y_edges, hot, regions


def repeated_pixel_hits(hits,yz_radius=0.5, x_min=5.0, min_count=3):
    """
    hits when >= min_count hits appear within a small yz region
    (treated as the same pixel region) while being well separated in x.
    """
    n = len(hits)
    if n < min_count:
        return hits, hits[:0]
    remove_mask = np.zeros(n, dtype=bool)
    x = hits["x"]
    y = hits["y"]
    z = hits["z"]

    for i in range(len(hits)):
        dy = y - y[i]
        dz = z - z[i]
        dx = np.abs(x - x[i])

        yz_close = (dy**2 + dz**2) < yz_radius**2
        x_far = dx > x_min

        neighbors = yz_close & x_far
        if np.sum(neighbors) + 1 >= min_count:
            remove_mask[neighbors | (np.arange(n) == i)] = True
    return hits[~remove_mask],hits[remove_mask]

def hits_bad_regions(hits, pixels, regions):
    if len(hits) == 0:
        return hits,hits
    # bad pixel mask
    bad_mask = np.zeros(len(hits), dtype=bool)
    for io_group, io_channel, chip_id, channel_id in pixels:
        bad_mask |= (
            (hits["io_group"]   == io_group) &
            (hits["io_channel"] == io_channel) &
            (hits["chip_id"]    == chip_id) &
            (hits["channel_id"] == channel_id)
        )
    # hot region mask
    hot_mask = np.zeros(len(hits), dtype=bool)
    y = hits["y"]
    z = hits["z"]
    for r in regions:
        zmin, zmax = r["z_range"]
        ymin, ymax = r["y_range"]
        hot_mask |= (
            (z >= zmin) & (z < zmax) &
            (y >= ymin) & (y < ymax)
        )
    # combine
    remove_mask = bad_mask | hot_mask
    return hits[~remove_mask],hits[remove_mask]