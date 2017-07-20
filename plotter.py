#!/usr/bin/env python3
import math
import matplotlib as mpl
# mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
mpl.rc('text', usetex=True)
mpl.rc('figure', dpi=200)
mpl.rc('savefig', dpi=200)


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


def to_bin_list(th1, include_errors=False):
    bins = []
    for i in range(th1.GetNbinsX()):
        center = th1.GetBinCenter(i + 1)
        width = th1.GetBinWidth(i + 1)
        content = th1.GetBinContent(i + 1)
        if include_errors:
            error = th1.GetBinError(i + 1)
            bins.append((center-width/2, center+width/2, (content, error)))
        else:
            bins.append((center-width/2, center+width/2, content))
    return bins


def histogram(th1, include_errors=False):
    edges = []
    values = []
    bin_list = to_bin_list(th1, include_errors)
    for (l_edge, _, val) in bin_list:
        edges.append(l_edge)
        values.append(val)
    edges.append(bin_list[-1][1])
    return values, edges


def histogram2d(th2, include_errors=False):
    """ converts TH2 object to something amenable to
    plotting w/ matplotlab's pcolormesh
    """
    import numpy as np
    nbins_x = th2.GetNbinsX()
    nbins_y = th2.GetNbinsY()
    xs = np.zeros((nbins_y, nbins_x), np.float64)
    ys = np.zeros((nbins_y, nbins_x), np.float64)
    zs = np.zeros((nbins_y, nbins_x), np.float64)
    for i in range(nbins_x):
        for j in range(nbins_y):
            xs[j][i] = th2.GetXaxis().GetBinLowEdge(i+1)
            ys[j][i] = th2.GetYaxis().GetBinLowEdge(j+1)
            zs[j][i] = th2.GetBinContent(i+1, j+1)
    # just_xs = np.array([th2.GetXaxes().GetBinLowEdge(i) for i in range(1,nbins_x)] +
    #                     [th2.GetXaxes().GetBinHighEdge(nbins_x-1)])
    # just_ys = np.array([th2.GetYaxes().GetBinLowEdge(i) for i in range(1,nbins_y)] +
    #                     [th2.GetYaxes().GetBinHighEdge(nbins_y-1)])

    return xs, ys, zs


class StackHist:

    def __init__(self, title=""):
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
        self.backgrounds.append((label, lumi, to_bin_list(th1), plot_color))

    def set_mc_signal(self, th1, label, lumi=None, stack=True, scale=1, plot_color=''):
        self.signal = (label, lumi, to_bin_list(th1), plot_color)
        self.signal_stack = stack
        self.signal_scale = scale

    def set_data(self, th1, lumi=None, plot_color=''):
        self.data = ('data', lumi, to_bin_list(th1), plot_color)
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
            axes.set_ylim(None, math.exp(math.log(max(bottoms))*1.4))
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
            xs, ys = zip(*[(x, sig/(sig+bg)) for x, sig, bg in zip(xs, sigs, bgs) if (sig+bg)>0])
            bottom.plot(xs, ys, '.k')

        if high_cut_significance:
            # s/(s+b) for events passing a minimum cut requirement
            min_bg = [sum(bgs[i:]) for i in range(self.n_bins)]
            min_sig = [sum(sigs[i:]) for i in range(self.n_bins)]
            min_xs, min_ys = zip(*[(x, sig/math.sqrt(sig+bg)) for x, sig, bg in zip(xs, min_sig, min_bg)
                                   if (sig+bg) > 0])
            bottom_rhs.plot(min_xs, min_ys, '->', color=rhs_color)

        if low_cut_significance:
            # s/(s+b) for events passing a maximum cut requirement
            max_bg = [sum(bgs[:i]) for i in range(self.n_bins)]
            max_sig = [sum(sigs[:i]) for i in range(self.n_bins)]
            max_xs, max_ys = zip(*[(x, sig/math.sqrt(sig+bg)) for x, sig, bg in zip(xs, max_sig, max_bg)
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

    rs_TTZ =  ResultSet("TTZ",  "../data/TTZToLLNuNu_treeProducerSusyMultilepton_tree.root")
    rs_TTW  = ResultSet("TTW",  "../data/TTWToLNu_treeProducerSusyMultilepton_tree.root")
    rs_TTH  = ResultSet("TTH", "../data/TTHnobb_mWCutfix_ext1_treeProducerSusyMultilepton_tree.root")
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
