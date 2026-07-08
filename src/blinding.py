import numpy as np

from make_doublet_blinding_mask_axis_symmetric import (
    _angle_delta,
)


def select_doublet_arrays(doublets, keep):
    keep = np.asarray(keep, dtype=bool)

    out = {}
    for key, val in doublets.items():
        arr = np.asarray(val)

        if arr.ndim > 0 and arr.shape[0] == keep.shape[0]:
            out[key] = arr[keep]
        else:
            out[key] = val

    return out


def axis_symmetric_projected_angles(dx, dy, dz, *, axis_symmetric=True):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dz = np.asarray(dz, dtype=float)

    if axis_symmetric:
        flip = dz < 0
        dx = np.where(flip, -dx, dx)
        dy = np.where(flip, -dy, dy)
        dz = np.abs(dz)

    theta_zx = np.arctan2(dx, dz)
    theta_zy = np.arctan2(dy, dz)

    return theta_zx, theta_zy


def circular_roi_mask_for_doublet_arrays(
    doublets,
    *,
    theta_radius_deg,
    center_zx_deg=0.0,
    center_zy_deg=0.0,
    axis_symmetric=True,
):
    theta_zx, theta_zy = axis_symmetric_projected_angles(
        doublets["dx"],
        doublets["dy"],
        doublets["dz"],
        axis_symmetric=axis_symmetric,
    )

    d_zx = _angle_delta(theta_zx, np.deg2rad(center_zx_deg))
    d_zy = _angle_delta(theta_zy, np.deg2rad(center_zy_deg))

    radius = np.sqrt(d_zx * d_zx + d_zy * d_zy)

    return radius <= np.deg2rad(theta_radius_deg)



def apply_roi_blinding(
    doublets,
    *,
    theta_radius_deg,
    center_zx_deg=0.0,
    center_zy_deg=0.0,
    axis_symmetric=True,
):
    in_roi = circular_roi_mask_for_doublet_arrays(
        doublets,
        theta_radius_deg=theta_radius_deg,
        center_zx_deg=center_zx_deg,
        center_zy_deg=center_zy_deg,
        axis_symmetric=axis_symmetric,
    )

    keep = ~in_roi
    blinded = select_doublet_arrays(doublets, keep)

    return blinded, keep, in_roi