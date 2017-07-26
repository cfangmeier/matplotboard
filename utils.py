
import io
import sys
from os.path import dirname, join, abspath, normpath
from math import floor, ceil, sqrt

import ROOT
from graph_vals import parse
from plotter import plot_histogram, plot_histogram2d

PRJ_PATH = normpath(join(dirname(abspath(__file__)), "../"))
EXE_PATH = join(PRJ_PATH, "build/main")

PDG = {1:   'd',   -1:  'd̄',
       2:   'u',   -2:  'ū',
       3:   's',   -3:  's̄',
       4:   'c',   -4:  'c̄',
       5:   'b',   -5:  'b̄',
       6:   't',   -6:  't̄',

       11:  'e-',  -11: 'e+',
       12:  'ν_e', -12: 'ῡ_e',

       13:  'μ-',  -13: 'μ+',
       14:  'ν_μ', -14: 'ῡ_μ',

       15:  'τ-',  -15: 'τ+',
       16:  'ν_τ', -16: 'ῡ_τ',

       21:  'g',
       22:  'γ',
       23:  'Z0',
       24:  'W+',  -24: 'W-',
       25:  'H',
       }


def get_color(val, max_val, min_val=0):
    val = (val-min_val)/(max_val-min_val)
    val = round(val * (ROOT.gStyle.GetNumberOfColors()-1))
    col_idx = ROOT.gStyle.GetColorPalette(val)
    col = ROOT.gROOT.GetColor(col_idx)
    r = floor(256*col.GetRed())
    g = floor(256*col.GetGreen())
    b = floor(256*col.GetBlue())
    gs = (r + g + b)//3
    text_color = 'white' if gs < 100 else 'black'
    return '#{:02x}{:02x}{:02x}'.format(r, g, b), text_color


def show_function(dataset, fname):
    from IPython.display import Markdown

    def md_single(fname_):
        impl = dataset._function_impl_lookup[fname_]
        return '*{}*\n-----\n```cpp\n{}\n```\n\n---'.format(fname_, impl)
    try:
        return Markdown('\n'.join(md_single(fname_) for fname_ in iter(fname)))
    except TypeError:
        return Markdown(md_single(fname))


def show_value(dataset, container):
    from IPython.display import Image
    if type(container) != str:
        container = container.GetName().split(':')[1]
    g, functions = parse(dataset.values[container], container)
    try:
        return Image(g.create_gif()), show_function(dataset, functions)
    except Exception as e:
        print(e)
        print(g.to_string())

def normalize_columns(hist2d):
    normHist = ROOT.TH2D(hist2d)
    cols, rows = hist2d.GetNbinsX(), hist2d.GetNbinsY()
    for col in range(1, cols+1):
        sum_ = 0
        for row in range(1, rows+1):
            sum_ += hist2d.GetBinContent(col, row)
        if sum_ == 0:
            continue
        for row in range(1, rows+1):
            norm = hist2d.GetBinContent(col, row) / sum_
            normHist.SetBinContent(col, row, norm)
    return normHist


class ResultSet:

    def __init__(self, sample_name, input_filename):
        self.sample_name = sample_name
        self.input_filename = input_filename
        self.load_objects()

        ResultSet.add_collection(self)

    def load_objects(self):
        file = ROOT.TFile.Open(self.input_filename)
        l = file.GetListOfKeys()
        self.map = {}
        self.values = dict(file.Get("_value_lookup"))
        for i in range(l.GetSize()):
            name = l.At(i).GetName()
            new_name = ":".join((self.sample_name, name))
            obj = file.Get(name)
            try:
                obj.SetName(new_name)
                obj.SetDirectory(0)  # disconnects Object from file
            except AttributeError:
                pass
            if 'ROOT.vector<int>' in str(type(obj)) and '_count' in name:
                obj = obj[0]
            self.map[name] = obj
            setattr(self, name, obj)
        file.Close()

        # Now add these histograms into the current ROOT directory (in memory)
        # and remove old versions if needed
        for obj in self.map.values():
            try:
                old_obj = ROOT.gDirectory.Get(obj.GetName())
                ROOT.gDirectory.Remove(old_obj)
                ROOT.gDirectory.Add(obj)
            except AttributeError:
                pass

    @classmethod
    def calc_shape(cls, n_plots):
        if n_plots > 3:
            return ceil(n_plots / 3), 3
        else:
            return 1, n_plots

    def draw(self, figure=None, shape=None):
        objs = [(name, obj) for name, obj in self.map.items() if isinstance(obj, ROOT.TH1)]
        shape = self.calc_shape(len(objs))
        if figure is None:
            import matplotlib.pyplot as plt
            figure = plt.gcf() if plt.gcf() is not None else plt.figure()
        figure.clear()
        for i, (name, obj) in enumerate(objs):
            axes = figure.add_subplot(*shape, i+1)
            if isinstance(obj, ROOT.TH2):
                plot_histogram2d(obj, title=obj.GetTitle(), axes=axes)
            else:
                plot_histogram(obj, title=obj.GetTitle(), axes=axes)
        figure.tight_layout()

    @classmethod
    def get_hist_set(cls, attrname):
        return [(sample_name, getattr(h, attrname))
                for sample_name, h in cls.collections.items()]

    @classmethod
    def add_collection(cls, hc):
        if not hasattr(cls, "collections"):
            cls.collections = {}
        cls.collections[hc.sample_name] = hc
