import numpy as np
import scipy.stats


def chi2_ndf_pvalue_from_hists(counts_d, counts_m, *, scale=1.0, n_fit_params=0):
    """
    Chi2 GOF with data statistical error + finite data-driven MC statistical error.

    model = scale * counts_m

    variance = Var(data) + Var(scaled MC)
             = counts_d + scale^2 * counts_m
    """
    counts_d = np.asarray(counts_d, dtype=float)
    counts_m = np.asarray(counts_m, dtype=float)

    model = scale * counts_m

    var = counts_d + (scale ** 2) * counts_m

    mask = ((counts_d > 0) | (model > 0)) & (var > 0)

    chi2 = float(np.sum((counts_d[mask] - model[mask]) ** 2 / var[mask]))
    ndf = int(np.sum(mask) - n_fit_params)

    pval = float(scipy.stats.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan

    return chi2, ndf, pval


def shape_gof_from_hists(counts_d, counts_m):
    """
    Shape-only chi2.
    MC is normalized to observed data total.
    One normalization parameter is fitted, so n_fit_params=1.
    """
    counts_d = np.asarray(counts_d, dtype=float)
    counts_m = np.asarray(counts_m, dtype=float)

    if np.sum(counts_m) <= 0:
        return np.nan, 0, np.nan, np.nan

    scale = np.sum(counts_d) / np.sum(counts_m)

    chi2, ndf, pval = chi2_ndf_pvalue_from_hists(
        counts_d,
        counts_m,
        scale=scale,
        n_fit_params=1,
    )

    return chi2, ndf, pval, scale


def absolute_gof_from_hists(counts_d, counts_m, expected_total):
    """
    Absolute-rate chi2.
    MC is normalized to singles-based expected total.
    No fitted normalization parameter, so n_fit_params=0.
    """
    counts_d = np.asarray(counts_d, dtype=float)
    counts_m = np.asarray(counts_m, dtype=float)

    if np.sum(counts_m) <= 0:
        return np.nan, 0, np.nan, np.nan

    scale = float(expected_total) / np.sum(counts_m)

    chi2, ndf, pval = chi2_ndf_pvalue_from_hists(
        counts_d,
        counts_m,
        scale=scale,
        n_fit_params=0,
    )

    return chi2, ndf, pval, scale