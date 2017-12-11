#!/usr/bin/env python3

from collections import defaultdict
from io import BytesIO
from base64 import b64encode
import numpy as np
import matplotlib.pyplot as plt
from markdown import Markdown
import latexipy as lp

from filval.histogram_utils import (hist, hist2d, hist_bin_centers, hist_fit,
                                    hist_normalize, hist_stats)

__all__ = ['Plot',
           'decl_plot',
           'grid_plot',
           'render_plots',
           'generate_dashboard',
           'hist_plot',
           'hist_plot_stack',
           'hist2d_plot']


class Plot:
    def __init__(self, subplots, name, title=None, docs="N/A", arg_dicts=None):
        self.subplots = subplots
        self.name = name
        self.title = title
        self.docs = docs
        self.arg_dicts = arg_dicts if arg_dicts is not None else {}


MD = Markdown(extensions=['mdx_math'],
              extension_configs={'mdx_math': {'enable_dollar_delimiter': True}})

lp.latexify(params={'pgf.texsystem': 'pdflatex',
                    'text.usetex': True,
                    'font.family': 'serif',
                    'pgf.preamble': [],
                    'font.size': 15,
                    'axes.labelsize': 15,
                    'axes.titlesize': 13,
                    'legend.fontsize': 13,
                    'xtick.labelsize': 11,
                    'ytick.labelsize': 11,
                    'figure.dpi': 150,
                    'savefig.transparent': False,
                    },
            new_backend='TkAgg')


def _fn_call_to_dict(fn, *args, **kwargs):
    from inspect import signature
    pnames = list(signature(fn).parameters)
    pvals = list(args) + list(kwargs.values())
    return {k: v for k, v in zip(pnames, pvals)}


def _process_docs(fn):
    from inspect import getdoc
    raw = getdoc(fn)
    if raw:
        return MD.convert(raw)
    else:
        return None


def decl_plot(fn):
    from functools import wraps

    @wraps(fn)
    def f(*args, **kwargs):
        fn(*args, **kwargs)
        argdict = _fn_call_to_dict(fn, *args, **kwargs)
        docs = _process_docs(fn)

        return argdict, docs

    return f


def generate_dashboard(plots, title, output='dashboard.html', template='dashboard.j2', source_file=None):
    from jinja2 import Environment, PackageLoader, select_autoescape
    from os.path import join
    from urllib.parse import quote

    env = Environment(
        loader=PackageLoader('filval', 'templates'),
        autoescape=select_autoescape(['htm', 'html', 'xml']),
    )
    env.globals.update({'quote': quote,
                        'enumerate': enumerate,
                        'zip': zip,
                        })

    def get_by_n(objects, n=2):
        objects = list(objects)
        while objects:
            yield objects[:n]
            objects = objects[n:]

    if source_file is not None:
        with open(source_file, 'r') as this_file:
            source = this_file.read()
    else:
        source = "# Not supplied!!"

    with open(join('output', output), 'w') as tempout:
        templ = env.get_template(template)
        tempout.write(templ.render(
            plots=get_by_n(plots, 3),
            title=title,
            source=source,
            outdir="figures/"
        ))


def _add_stats(hist, title=''):
    fmt = r'''\begin{{eqnarray*}}
\sum{{x_i}} &=& {sum:5.3f}                  \\
\sum{{\Delta x_i \cdot x_i}} &=& {int:5.3G} \\
\mu &=& {mean:5.3G}                         \\
\sigma^2 &=& {var:5.3G}                     \\
\sigma &=& {std:5.3G}
\end{{eqnarray*}}'''

    txt = fmt.format(**hist_stats(hist), title=title)
    txt = txt.replace('\n', ' ')

    plt.text(0.7, 0.9, txt,
             bbox={'facecolor': 'white',
                   'alpha': 0.7,
                   'boxstyle': 'square,pad=0.8'},
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             size='small')
    if title:
        plt.text(0.72, 0.97, title,
                 bbox={'facecolor': 'white',
                       'alpha': 0.8},
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='left')


def grid_plot(subplots):
    if any(len(row) != len(subplots[0]) for row in subplots):
        raise ValueError("make_plot requires a rectangular list-of-lists as "
                         "input. Fill empty slots with None")

    def calc_rowspan(fig, row, col):
        span = 1
        for r in range(row + 1, len(fig)):
            if fig[r][col] == "FU":
                span += 1
            else:
                break
        return span

    def calc_colspan(fig, row, col):
        span = 1
        for c in range(col + 1, len(fig[row])):
            if fig[row][c] == "FL":
                span += 1
            else:
                break
        return span

    rows = len(subplots)
    cols = len(subplots[0])

    argdicts = defaultdict(list)
    docs = defaultdict(list)
    for i in range(rows):
        for j in range(cols):
            cell = subplots[i][j]
            if cell in ("FL", "FU", None):
                continue
            if not isinstance(cell, list):
                cell = [cell]
            colspan = calc_colspan(subplots, i, j)
            rowspan = calc_rowspan(subplots, i, j)
            plt.subplot2grid((rows, cols), (i, j),
                             colspan=colspan, rowspan=rowspan)
            for plot in cell:
                plot_fn, args, kwargs = plot
                this_args, this_docs = plot_fn(*args, **kwargs)
                argdicts[(i, j)].append(this_args)
                docs[(i, j)].append(this_docs)
    return argdicts, docs


def render_plots(plots, exts=('png',), scale=1.0, to_disk=True):
    for plot in plots:
        print(f'Building plot {plot.name}')
        plot.data = None
        if to_disk:
            with lp.figure(plot.name, directory='output/figures',
                           exts=exts,
                           size=(scale * 10, scale * 10)):
                argdicts, docs = grid_plot(plot.subplots)
        else:
            out = BytesIO()
            with lp.mem_figure(out,
                               ext=exts[0],
                               size=(scale * 10, scale * 10)):
                argdicts, docs = grid_plot(plot.subplots)
            out.seek(0)
            plot.data = b64encode(out.read()).decode()
        plot.argdicts = argdicts
        plot.docs = docs


def add_decorations(axes, luminosity, energy):
    cms_prelim = r'{\raggedright{}\textsf{\textbf{CMS}}\\ \emph{Preliminary}}'
    axes.text(0.01, 0.98, cms_prelim,
              horizontalalignment='left',
              verticalalignment='top',
              transform=axes.transAxes)

    lumi = ""
    energy_str = ""
    if luminosity is not None:
        lumi = r'${} \mathrm{{fb}}^{{-1}}$'.format(luminosity)
    if energy is not None:
        energy_str = r'({} TeV)'.format(energy)

    axes.text(1, 1, ' '.join([lumi, energy_str]),
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=axes.transAxes)


def hist_plot(h, *args, norm=None, include_errors=False,
              log=False, xlim=None, ylim=None, fit=None,
              grid=False, stats=False, **kwargs):
    """ Plots a 1D ROOT histogram object using matplotlib """
    from inspect import signature
    if norm:
        h = hist_normalize(h, norm)
    values, errors, edges = h

    scale = 1. if norm is None else norm / np.sum(values)
    values = [val * scale for val in values]
    errors = [val * scale for val in errors]

    left, right = np.array(edges[:-1]), np.array(edges[1:])
    x = np.array([left, right]).T.flatten()
    y = np.array([values, values]).T.flatten()

    ax = plt.gca()

    ax.set_xlabel(kwargs.pop('xlabel', ''))
    ax.set_ylabel(kwargs.pop('ylabel', ''))
    title = kwargs.pop('title', '')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # elif not log:
    #     axes.set_ylim((0, None))

    ax.plot(x, y, *args, linewidth=1, **kwargs)
    if include_errors:
        ax.errorbar(hist_bin_centers(h), values, yerr=errors,
                    color='k', marker=None, linestyle='None',
                    barsabove=True, elinewidth=.7, capsize=1)
    if log:
        ax.set_yscale('log')
    if fit:
        f, p0 = fit
        popt, pcov = hist_fit(h, f, p0)
        fit_xs = np.linspace(x[0], x[-1], 100)
        fit_ys = f(fit_xs, *popt)
        ax.plot(fit_xs, fit_ys, '--g')
        arglabels = list(signature(f).parameters)[1:]
        label_txt = "\n".join('{:7s}={: 0.2G}'.format(label, value)
                              for label, value in zip(arglabels, popt))
        ax.text(0.60, 0.95, label_txt, va='top', transform=ax.transAxes,
                fontsize='medium', family='monospace', usetex=False)
    if stats:
        _add_stats(h, title)
    else:
        ax.set_title(title)
    ax.grid(grid, color='#E0E0E0')


def hist2d_plot(h, **kwargs):
    """ Plots a 2D ROOT histogram object using matplotlib """
    try:
        values, errors, xs, ys = h
    except (TypeError, ValueError):
        values, errors, xs, ys = hist2d(h)

    plt.xlabel(kwargs.pop('xlabel', ''))
    plt.ylabel(kwargs.pop('ylabel', ''))
    plt.title(kwargs.pop('title', ''))
    plt.pcolormesh(xs, ys, values, )
    # axes.colorbar() TODO: Re-enable this


def hist_plot_stack(hists: list, labels: list = None):
    """
    Creates a stacked histogram in the current axes.

    :param hists: list of histogram
    :param labels:
    :return:
    """
    if len(hists) == 0:
        return

    if len(set([len(hist[0]) for hist in hists])) != 1:
        raise ValueError("all histograms must have the same number of bins")
    if labels is None:
        labels = [None for _ in hists]
    if len(labels) != len(hists):
        raise ValueError("Label mismatch")

    bottoms = [0 for _ in hists[0][0]]

    for hist, label in zip(hists, labels):
        centers = []
        widths = []
        heights = []
        for left, right, content in zip(hist[2][:-1], hist[2][1:], hist[0]):
            centers.append((right + left) / 2)
            widths.append(right - left)
            heights.append(content)

        plt.bar(centers, heights, widths, bottoms, label=label)
        for i, content in enumerate(hist[0]):
            bottoms[i] += content
