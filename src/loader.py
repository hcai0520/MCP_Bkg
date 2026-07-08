import h5py
import numpy as np

from make_doublet_blinding_mask_axis_symmetric import (
    FiducialConfig,
    fiducial_mask,
)


FIDUCIAL = FiducialConfig()


def _stringify_event_key_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    try:
        if np.issubdtype(type(value), np.bytes_):
            return bytes(value).decode("utf-8", errors="replace")
    except Exception:
        pass

    return str(value)


def _get_optional_field(dtype_names, candidates):
    for name in candidates:
        if name in dtype_names:
            return name
    return None


def _summarize_one_event_hits(
    *,
    hits,
    event_key,
    event_index,
    trigger_type,
    use_centroid=True,
    x_field="x",
    y_field="y",
    z_field="z",
    label_field="cluster_id",
    energy_field="Q",
):
    if hits is None or len(hits) == 0:
        return []

    x = np.asarray(hits[x_field], dtype=float)
    y = np.asarray(hits[y_field], dtype=float)
    z = np.asarray(hits[z_field], dtype=float)
    labels = np.asarray(hits[label_field])

    geom_mask = fiducial_mask(x, y, z, FIDUCIAL)
    idx = np.where(geom_mask & (labels >= 0))[0]

    if idx.size == 0:
        return []

    labs = labels[idx]
    clusters = []

    for lab in np.unique(labs):
        cidx = idx[labs == lab]
        if cidx.size == 0:
            continue

        if use_centroid:
            pos = np.array(
                [x[cidx].mean(), y[cidx].mean(), z[cidx].mean()],
                dtype=float,
            )
        else:
            pos = np.array(
                [x[cidx[0]], y[cidx[0]], z[cidx[0]]],
                dtype=float,
            )

        if energy_field is not None and energy_field in hits.dtype.names:
            energy = float(np.nansum(np.asarray(hits[energy_field])[cidx]))
        else:
            energy = np.nan

        clusters.append({
            "event_key": str(event_key),
            "event_index": int(event_index),
            "trigger_type": int(trigger_type) if trigger_type is not None else -1,
            "label": int(lab),
            "npts": int(cidx.size),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "energy": float(energy) if np.isfinite(energy) else np.nan,
        })

    return clusters


def load_cluster_event_summary(
    h5_path,
    use_centroid=True,
    *,
    hits_dataset="normal_hits/data",
    ref_region_dataset="normal_hits/ref_region",
    x_field="x",
    y_field="y",
    z_field="z",
    label_field="cluster_id",
    energy_field="Q",
    event_id_field=None,
    trigger_type_field=None,
):
    multiplicities = []
    singles = []
    doublets = []
    triplets = []
    all_clusters = []

    with h5py.File(h5_path, "r") as f:
        hits_all = f[hits_dataset]
        ref = f[ref_region_dataset]

        hit_names = hits_all.dtype.names
        ref_names = ref.dtype.names

        if hit_names is None:
            raise TypeError(f"{hits_dataset} must be a structured dataset.")
        if ref_names is None:
            raise TypeError(f"{ref_region_dataset} must be a structured dataset.")

        if event_id_field is None:
            event_id_field_hits = _get_optional_field(
                hit_names,
                ["event_id", "event", "evt_id", "event_idx", "event_index", "entry", "spill_id"],
            )
            event_id_field_ref = _get_optional_field(
                ref_names,
                ["event_id", "event", "evt_id", "event_idx", "event_index", "entry", "spill_id"],
            )
        else:
            event_id_field_hits = event_id_field if event_id_field in hit_names else None
            event_id_field_ref = event_id_field if event_id_field in ref_names else None

        if trigger_type_field is None:
            trigger_field_hits = _get_optional_field(
                hit_names,
                ["beam_type", "trigger_type", "is_beam", "beam_related", "beam", "trigger"],
            )
            trigger_field_ref = _get_optional_field(
                ref_names,
                ["beam_type", "trigger_type", "is_beam", "beam_related", "beam", "trigger"],
            )
        else:
            trigger_field_hits = trigger_type_field if trigger_type_field in hit_names else None
            trigger_field_ref = trigger_type_field if trigger_type_field in ref_names else None

        for event_index in range(len(ref)):
            rr_row = ref[event_index]

            start = int(rr_row["start"])
            stop = int(rr_row["stop"])

            if stop <= start:
                multiplicities.append(0)
                continue

            hits = hits_all[start:stop]

            if event_id_field_ref is not None:
                event_key = _stringify_event_key_value(rr_row[event_id_field_ref])
            elif event_id_field_hits is not None and len(hits) > 0:
                event_key = _stringify_event_key_value(np.asarray(hits[event_id_field_hits]).reshape(-1)[0])
            else:
                event_key = str(event_index)

            if trigger_field_hits is not None and len(hits) > 0:
                trigger_type = int(np.asarray(hits[trigger_field_hits]).reshape(-1)[0])
            elif trigger_field_ref is not None:
                trigger_type = int(rr_row[trigger_field_ref])
            else:
                trigger_type = -1

            clusters = _summarize_one_event_hits(
                hits=hits,
                event_key=event_key,
                event_index=event_index,
                trigger_type=trigger_type,
                use_centroid=use_centroid,
                x_field=x_field,
                y_field=y_field,
                z_field=z_field,
                label_field=label_field,
                energy_field=energy_field,
            )

            multiplicities.append(len(clusters))
            all_clusters.extend(clusters)

            if len(clusters) == 1:
                singles.append(clusters[0])
            elif len(clusters) == 2:
                doublets.append(clusters)
            elif len(clusters) == 3:
                triplets.append(clusters)

    return {
        "multiplicities": np.asarray(multiplicities, dtype=int),
        "singles": singles,
        "doublets": doublets,
        "triplets": triplets,
        "all_clusters": all_clusters,
        "source_info": {
            "hits_dataset": hits_dataset,
            "ref_region_dataset": ref_region_dataset,
            "x_field": x_field,
            "y_field": y_field,
            "z_field": z_field,
            "label_field": label_field,
            "energy_field": energy_field,
            "event_id_field_hits": event_id_field_hits,
            "event_id_field_ref": event_id_field_ref,
            "trigger_field_hits": trigger_field_hits,
            "trigger_field_ref": trigger_field_ref,
        },
    }

    