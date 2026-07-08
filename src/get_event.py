from collections import defaultdict
import h5py

def slice_from_ref_region(dset, ref_region_row):
    start = int(ref_region_row["start"])
    stop  = int(ref_region_row["stop"])
    return dset[start:stop]

def get_event_hits_by_event_index(f, event_index):
    hits = f["charge/calib_prompt_hits/data"]
    rr   = f["charge/events/ref/charge/calib_prompt_hits/ref_region"]
    return slice_from_ref_region(hits, rr[event_index])

def get_beam_type(f, evt_idx):
    rr = f["charge/events/ref/light/events/ref_region"]
    light = f["light/events/data"]

    start = int(rr[evt_idx]["start"])
    stop  = int(rr[evt_idx]["stop"])

    if stop <= start:
        return -1

    return int(light[start]["trig_type"]) 


def build_occupancy(paths, level="pixel"):
    """
    paths : str or list[str]
    """

    if isinstance(paths, str):
        paths = [paths]

    occupancy = defaultdict(int)

    for path in paths:
        with h5py.File(path, "r") as f:
            n_events = len(f["charge/events/data"])

            for evt_idx in range(n_events):
                hits = get_event_hits_by_event_index(f, evt_idx)

                for h in hits:
                    if level == "pixel":
                        key = (
                            int(h["io_group"]),
                            int(h["io_channel"]),
                            int(h["chip_id"]),
                            int(h["channel_id"])
                        )
                    elif level == "chip":
                        key = (
                            int(h["io_group"]),
                            int(h["io_channel"]),
                            int(h["chip_id"])
                        )
                    else:
                        raise ValueError("level must be 'pixel' or 'chip'.")

                    occupancy[key] += 1

    return occupancy