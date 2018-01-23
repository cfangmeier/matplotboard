"""
    histogram.py
    The functions in this module use a representation of a histogram that is a
    tuple containing an arr of N bin values, an array of N bin errors(symmetric)
    and an array of N+1 bin edges(N lower edges + 1 upper edge).

    For 2d histograms, It is similar, but the arrays are two dimensional and
    there are separate arrays for x-edges and y-edges.
"""

import numpy as np
from scipy.optimize import curve_fit


def hist(th1, rescale_x=1.0, rescale_y=1.0):
    nbins = th1.GetNbinsX()

    edges = np.zeros(nbins+1, np.float32)
    values = np.zeros(nbins, np.float32)
    errors = np.zeros(nbins, np.float32)

    for i in range(nbins):
        edges[i] = th1.GetXaxis().GetBinLowEdge(i+1)
        values[i] = th1.GetBinContent(i+1)
        errors[i] = th1.GetBinError(i+1)

    edges[nbins] = th1.GetXaxis().GetBinUpEdge(nbins)
    edges *= rescale_x
    values *= rescale_y
    errors *= rescale_y
    return values, errors, edges


def hist_bin_centers(h):
    _, _, edges = h
    return (edges[:-1] + edges[1:])/2.0


def hist_slice(h, range_):
    values, errors, edges = h
    lim_low, lim_high = range_
    slice_ = np.logical_and(edges[:-1] > lim_low, edges[1:] < lim_high)
    last = len(slice_) - np.argmax(slice_[::-1])
    return (values[slice_],
            errors[slice_],
            np.concatenate([edges[:-1][slice_], [edges[last]]]))


def hist_add(*hs):
    if len(hs) == 0:
        return np.zeros(0)
    vals, errs, edges = zip(*hs)
    return np.sum(vals, axis=0), np.sqrt(np.sum([err*err for err in errs], axis=0)), edges[0]


def hist_integral(h, times_bin_width=True):
    values, errors, edges = h
    if times_bin_width:
        bin_widths = [abs(x2 - x1) for x1, x2 in zip(edges[:-1], edges[1:])]
        return sum(val*width for val, width in zip(values, bin_widths))
    else:
        return sum(values)


def hist_scale(h, scale):
    values, errors, edges = h
    return values*scale, errors*scale, edges


def hist_norm(h, norm=1):
    scale = norm/np.sum(h[0])
    return hist_scale(h, scale)


def hist_mean(h):
    xs = hist_bin_centers(h)
    ys, _, _ = h
    return sum(x*y for x, y in zip(xs, ys)) / sum(ys)


def hist_var(h):
    xs = hist_bin_centers(h)
    ys, _, _ = h
    mean = sum(x*y for x, y in zip(xs, ys)) / sum(ys)
    mean2 = sum((x**2)*y for x, y in zip(xs, ys)) / sum(ys)
    return mean2 - mean**2


def hist_std(h):
    return np.sqrt(hist_var(h))


def hist_stats(h):
    return {'int': hist_integral(h),
            'sum': hist_integral(h, False),
            'mean': hist_mean(h),
            'var': hist_var(h),
            'std': hist_std(h)}


# def hist_slice2d(h, range_):
#     values, errors, xs, ys = h

#     last = len(slice_) - np.argmax(slice_[::-1])

#     (xlim_low, xlim_high), (ylim_low, ylim_high) = range_
#     slice_ = np.logical_and(xs[:-1, :-1] > xlim_low, xs[1:, 1:] < xlim_high,
#                             ys[:-1, :-1] > ylim_low, ys[1:, 1:] < ylim_high)
#     last = len(slice_) - np.argmax(slice_[::-1])
#     return (values[slice_],
#             errors[slice_],
#             np.concatenate([edges[:-1][slice_], [edges[last]]]))


def hist_fit(h, f, p0=None):
    values, errors, edges = h
    xs = hist_bin_centers(h)
    # popt, pcov = curve_fit(f, xs, values, p0=p0, sigma=errors)
    popt, pcov = curve_fit(f, xs, values, p0=p0)
    return popt, pcov


def hist_rebin(h, range_, nbins):
    raise NotImplementedError()


def hist2d(th2, rescale_x=1.0, rescale_y=1.0, rescale_z=1.0):
    """ Converts TH2 object to something amenable to
        plotting w/ matplotlab's pcolormesh.
    """
    nbins_x = th2.GetNbinsX()
    nbins_y = th2.GetNbinsY()
    xs = np.zeros((nbins_y+1, nbins_x+1), np.float32)
    ys = np.zeros((nbins_y+1, nbins_x+1), np.float32)
    values = np.zeros((nbins_y, nbins_x), np.float32)
    errors = np.zeros((nbins_y, nbins_x), np.float32)
    for i in range(nbins_x):
        for j in range(nbins_y):
            xs[j][i] = th2.GetXaxis().GetBinLowEdge(i+1)
            ys[j][i] = th2.GetYaxis().GetBinLowEdge(j+1)
            values[j][i] = th2.GetBinContent(i+1, j+1)
            errors[j][i] = th2.GetBinError(i+1, j+1)
        xs[nbins_y][i] = th2.GetXaxis().GetBinUpEdge(i)
        ys[nbins_y][i] = th2.GetYaxis().GetBinUpEdge(nbins_y)
    for j in range(nbins_y+1):
        xs[j][nbins_x] = th2.GetXaxis().GetBinUpEdge(nbins_x)
        ys[j][nbins_x] = th2.GetYaxis().GetBinUpEdge(j)

    xs *= rescale_x
    ys *= rescale_y
    values *= rescale_z
    errors *= rescale_z

    return values, errors, xs, ys


def hist2d_norm(h, norm=1, axis=None):
    """

    :param h:
    :param norm: value to normalize the sum of axis to
    :param axis: which axis to normalize None is the sum over all bins, 0 is columns, 1 is rows.
    :return: The normalized histogram
    """
    values, errors, xs, ys = h
    with np.errstate(divide='ignore'):
        scale_values = norm / np.sum(values, axis=axis)
        scale_values[scale_values == np.inf] = 1
        scale_values[scale_values == -np.inf] = 1
    if axis == 1:
        scale_values.shape = (scale_values.shape[0], 1)
    values = values * scale_values
    errors = errors * scale_values
    return values, errors, xs.copy(), ys.copy()


