'''
    histogram_utils.py
    The functions in this module use a representation of a histogram that is a
    tuple containing an arr of N bin values, an array of N bin errors(symmetric)
    and an array of N+1 bin edges(N lower edges + 1 upper edge).

    For 2d histograms, It is similar, but the arrays are two dimensional and
    there are separate arrays for x-edges and y-edges.
'''

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
        xs[nbins_y][i] = th2.GetXaxis().GetBinUpEdge(i+1)
        ys[nbins_y][i] = th2.GetYaxis().GetBinUpEdge(nbins_y+1)
    for j in range(nbins_y+1):
        xs[j][nbins_x] = th2.GetXaxis().GetBinUpEdge(nbins_x+1)
        ys[j][nbins_x] = th2.GetYaxis().GetBinUpEdge(j+1)

    xs *= rescale_x
    ys *= rescale_y
    values *= rescale_z
    errors *= rescale_z

    return values, errors, xs, ys


def hist_slice(hist, range_):
    values, errors, edges = hist
    lim_low, lim_high = range_
    slice_ = np.logical_and(edges[:-1] > lim_low, edges[1:] < lim_high)
    last = len(slice_) - np.argmax(slice_[::-1])
    return (values[slice_],
            errors[slice_],
            np.concatenate([edges[:-1][slice_], [edges[last]]]))


def hist_normalize(h, norm):
    values, errors, edges = h
    scale = norm/np.sum(values)
    return values*scale, errors*scale, edges


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


def hist_rebin(hist, range_, nbins):
    raise NotImplementedError()
