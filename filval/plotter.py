#!/usr/bin/env python3
from collections import namedtuple
import matplotlib as mpl
import numpy as np
from filval.histogram_utils import (hist, hist2d, hist_bin_centers, hist_fit,
                                    hist_normalize)
# mpl.rc('text', usetex=True)
# mpl.rc('figure', dpi=200)
# mpl.rc('savefig', dpi=200)

plot_registry = {}
Plot = namedtuple('Plot', ['name', 'filename', 'title', 'desc', 'args'])


def make_plot(filename=None, title='', scale=1):
    import matplotlib.pyplot as plt
    from functools import wraps
    from os.path import join
    from os import makedirs
    from inspect import signature, getdoc
    from markdown import Markdown

    def fn_call_to_dict(fn, *args, **kwargs):
        pnames = list(signature(fn).parameters)
        pvals = list(args)+list(kwargs.keys())
        return {k: v for k, v in zip(pnames, pvals)}

    def process_docs(fn):
        raw = getdoc(fn)
        if raw:
            md = Markdown(extensions=['mdx_math'],
                          extension_configs={'mdx_math': {'enable_dollar_delimiter': True}})
            return md.convert(raw)
        else:
            return None

    def wrap(fn):
        @wraps(fn)
        def f(*args, **kwargs):
            nonlocal filename
            plt.clf()
            plt.gcf().set_size_inches(scale*10, scale*10)
            fn(*args, **kwargs)
            pdict = fn_call_to_dict(fn, *args, **kwargs)
            if filename is None:
                pstr = ','.join('{}:{}'.format(pname, pval)
                                for pname, pval in pdict.items())
                filename = fn.__name__ + '::' + pstr
                filename = filename.replace('/', '_').replace('.', '_')+".png"
            plt.tight_layout()
            try:
                makedirs('output/figures')
            except FileExistsError:
                pass
            plt.savefig(join('output/figures', filename))
            plot_registry[fn.__name__] = Plot(fn.__name__, join('figures', filename),
                                              title, process_docs(fn), pdict)
        return f

    return wrap


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


def hist_plot(h, *args, axes=None, norm=None, include_errors=False,
              log=False, fig=None, xlim=None, ylim=None, fit=None,
              **kwargs):
    """ Plots a 1D ROOT histogram object using matplotlib """
    from inspect import signature
    if norm:
        h = hist_normalize(h, norm)
    values, errors, edges = h

    scale = 1. if norm is None else norm/np.sum(values)
    values = [val*scale for val in values]
    errors = [val*scale for val in errors]

    left, right = np.array(edges[:-1]), np.array(edges[1:])
    X = np.array([left, right]).T.flatten()
    Y = np.array([values, values]).T.flatten()

    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()

    axes.set_xlabel(kwargs.pop('xlabel', ''))
    axes.set_ylabel(kwargs.pop('ylabel', ''))
    axes.set_title(kwargs.pop('title', ''))
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    # elif not log:
    #     axes.set_ylim((0, None))

    axes.plot(X, Y, *args, linewidth=1, **kwargs)
    if include_errors:
        axes.errorbar(hist_bin_centers(h), values, yerr=errors,
                      color='k', marker=None, linestyle='None',
                      barsabove=True, elinewidth=.7, capsize=1)
    if log:
        axes.set_yscale('log')
    if fit:
        f, p0 = fit
        popt, pcov = hist_fit(h, f, p0)
        fit_xs = np.linspace(X[0], X[-1], 100)
        fit_ys = f(fit_xs, *popt)
        axes.plot(fit_xs, fit_ys, '--g')
        arglabels = list(signature(f).parameters)[1:]
        label_txt = "\n".join('{:7s}={: 0.2G}'.format(label, value)
                              for label, value in zip(arglabels, popt))
        axes.text(0.60, 0.95, label_txt, va='top', transform=axes.transAxes,
                  fontsize='x-small', family='monospace', usetex=False)
    axes.grid()


def hist2d_plot(h, *args, axes=None, **kwargs):
    """ Plots a 2D ROOT histogram object using matplotlib """
    try:
        values, errors, xs, ys = h
    except (TypeError, ValueError):
        values, errors, xs, ys = hist2d(h)

    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()
    axes.set_xlabel(kwargs.pop('xlabel', ''))
    axes.set_ylabel(kwargs.pop('ylabel', ''))
    axes.set_title(kwargs.pop('title', ''))
    axes.pcolormesh(xs, ys, values,)
    # axes.colorbar() TODO: Re-enable this


class StackHist:

    def __init__(self, title=""):
        raise NotImplementedError("need to fix to not use to_bin_list")
        self.title = title
        self.xlabel = ""
        self.ylabel = ""
        self.xlim = (None, None)
        self.ylim = (None, None)
        self.logx = False
        self.logy = False
        self.backgrounds = []
        self.signal = None
        self.signal_stack = True
        self.data = None

    def add_mc_background(self, th1, label, lumi=None, plot_color=''):
        self.backgrounds.append((label, lumi, hist(th1), plot_color))

    def set_mc_signal(self, th1, label, lumi=None, stack=True, scale=1, plot_color=''):
        self.signal = (label, lumi, hist(th1), plot_color)
        self.signal_stack = stack
        self.signal_scale = scale

    def set_data(self, th1, lumi=None, plot_color=''):
        self.data = ('data', lumi, hist(th1), plot_color)
        self.luminosity = lumi

    def _verify_binning_match(self):
        bins_count = [len(bins) for _, _, bins, _ in self.backgrounds]
        if self.signal is not None:
            bins_count.append(len(self.signal[2]))
        if self.data is not None:
            bins_count.append(len(self.data[2]))
        n_bins = bins_count[0]
        if any(bin_count != n_bins for bin_count in bins_count):
            raise ValueError("all histograms must have the same number of bins")
        self.n_bins = n_bins

    def save(self, filename, **kwargs):
        import matplotlib.pyplot as plt
        plt.ioff()
        fig = plt.figure()
        ax = fig.gca()
        self.do_draw(ax, **kwargs)
        fig.savefig("figures/"+filename, transparent=True)
        plt.close(fig)
        plt.ion()

    def do_draw(self, axes):
        self.axeses = [axes]
        self._verify_binning_match()
        bottoms = [0]*self.n_bins

        if self.logx:
            axes.set_xscale('log')
        if self.logy:
            axes.set_yscale('log')

        def draw_bar(label, lumi, bins, plot_color, scale=1, stack=True, **kwargs):
            if stack:
                lefts = []
                widths = []
                heights = []
                for left, right, content in bins:
                    lefts.append(left)
                    widths.append(right-left)
                    if lumi is not None:
                        content *= self.luminosity/lumi
                    content *= scale
                    heights.append(content)

                axes.bar(lefts, heights, widths, bottoms, label=label, color=plot_color, **kwargs)
                for i, (_, _, content) in enumerate(bins):
                    if lumi is not None:
                        content *= self.luminosity/lumi
                    content *= scale
                    bottoms[i] += content
            else:
                xs = [bins[0][0] - (bins[0][1]-bins[0][0])/2]
                ys = [0]
                for left, right, content in bins:
                    width2 = (right-left)/2
                    if lumi is not None:
                        content *= self.luminosity/lumi
                    content *= scale
                    xs.append(left-width2)
                    ys.append(content)
                    xs.append(right-width2)
                    ys.append(content)
                xs.append(bins[-1][0] + (bins[-1][1]-bins[-1][0])/2)
                ys.append(0)
                axes.plot(xs, ys, label=label, color=plot_color, **kwargs)

        if self.signal is not None and self.signal_stack:
            label, lumi, bins, plot_color = self.signal
            if self.signal_scale != 1:
                label = r"{}$\times{:d}$".format(label, self.signal_scale)
            draw_bar(label, lumi, bins, plot_color, scale=self.signal_scale, hatch='/')

        for background in self.backgrounds:
            draw_bar(*background)

        if self.signal is not None and not self.signal_stack:
            # draw_bar(*self.signal, stack=False, color='k')
            label, lumi, bins, plot_color = self.signal
            if self.signal_scale != 1:
                label = r"{}$\times{:d}$".format(label, self.signal_scale)
            draw_bar(label, lumi, bins, plot_color, scale=self.signal_scale, stack=False)

        axes.set_title(self.title)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xlim(*self.xlim)
        # axes.set_ylim(*self.ylim)
        if self.logy:
            axes.set_ylim(None, np.exp(np.log(max(bottoms))*1.4))
        else:
            axes.set_ylim(None, max(bottoms)*1.2)
        axes.legend(frameon=True, ncol=2)
        add_decorations(axes, self.luminosity, self.energy)

    def draw(self, axes, save=False, filename=None, **kwargs):
        self.do_draw(axes, **kwargs)
        if save:
            if filename is None:
                filename = "".join(c for c in self.title if c.isalnum() or c in (' ._+-'))+".png"
            self.save(filename, **kwargs)


class StackHistWithSignificance(StackHist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_draw(self, axes, bin_significance=True, low_cut_significance=False, high_cut_significance=False):
        bottom_box, _, top_box = axes.get_position().splity(0.28, 0.30)
        axes.set_position(top_box)
        super().do_draw(axes)
        axes.set_xticks([])
        rhs_color = '#cc6600'

        bottom = axes.get_figure().add_axes(bottom_box)
        bottom_rhs = bottom.twinx()
        bgs = [0]*self.n_bins
        for (_, _, bins, _) in self.backgrounds:
            for i, (left, right, value) in enumerate(bins):
                bgs[i] += value

        sigs = [0]*self.n_bins
        if bin_significance:
            xs = []
            for i, (left, right, value) in enumerate(self.signal[2]):
                sigs[i] += value
                xs.append(left)
            xs, ys = zip(*[(x, sig/(sig+bg)) for x, sig, bg in zip(xs, sigs, bgs) if (sig+bg) > 0])
            bottom.plot(xs, ys, '.k')

        if high_cut_significance:
            # s/(s+b) for events passing a minimum cut requirement
            min_bg = [sum(bgs[i:]) for i in range(self.n_bins)]
            min_sig = [sum(sigs[i:]) for i in range(self.n_bins)]
            min_xs, min_ys = zip(*[(x, sig/np.sqrt(sig+bg)) for x, sig, bg in zip(xs, min_sig, min_bg)
                                   if (sig+bg) > 0])
            bottom_rhs.plot(min_xs, min_ys, '->', color=rhs_color)

        if low_cut_significance:
            # s/(s+b) for events passing a maximum cut requirement
            max_bg = [sum(bgs[:i]) for i in range(self.n_bins)]
            max_sig = [sum(sigs[:i]) for i in range(self.n_bins)]
            max_xs, max_ys = zip(*[(x, sig/np.sqrt(sig+bg)) for x, sig, bg in zip(xs, max_sig, max_bg)
                                   if (sig+bg) > 0])
            bottom_rhs.plot(max_xs, max_ys, '-<', color=rhs_color)

        bottom.set_ylabel(r'$S/(S+B)$')
        bottom.set_xlim(axes.get_xlim())
        bottom.set_ylim((0, 1.1))
        if low_cut_significance or high_cut_significance:
            bottom_rhs.set_ylabel(r'$S/\sqrt{S+B}$')
            bottom_rhs.yaxis.label.set_color(rhs_color)
            bottom_rhs.tick_params(axis='y', colors=rhs_color, size=4, width=1.5)
        # bottom.grid()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import ResultSet

    rs_TTZ = ResultSet("TTZ",  "../data/TTZToLLNuNu_treeProducerSusyMultilepton_tree.root")
    rs_TTW = ResultSet("TTW",  "../data/TTWToLNu_treeProducerSusyMultilepton_tree.root")
    rs_TTH = ResultSet("TTH", "../data/TTHnobb_mWCutfix_ext1_treeProducerSusyMultilepton_tree.root")
    rs_TTTT = ResultSet("TTTT", "../data/TTTT_ext_treeProducerSusyMultilepton_tree.root")

    sh = StackHist('B-Jet Multiplicity')
    sh.add_mc_background(rs_TTZ.b_jet_count, 'TTZ', lumi=40)
    sh.add_mc_background(rs_TTW.b_jet_count, 'TTW', lumi=40)
    sh.add_mc_background(rs_TTH.b_jet_count, 'TTH', lumi=40)
    sh.set_mc_signal(rs_TTTT.b_jet_count, 'TTTT', lumi=40, scale=10)

    sh.luminosity = 40
    sh.energy = 13
    sh.xlabel = 'B-Jet Count'
    sh.ylabel = r'\# Events'
    sh.xlim = (-.5, 9.5)
    sh.signal_stack = False

    fig = plt.figure()
    sh.draw(fig.gca())
    plt.show()
    # sh.add_data(rs_TTZ.b_jet_count, 'TTZ')
