
import io
import sys
import itertools as it
from os.path import dirname, join, abspath, normpath
from math import ceil, floor, sqrt
from collections import deque
from IPython.display import Image

import ROOT
from graph_vals import parse

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

SINGLE_PLOT_SIZE = (600, 450)
MAX_WIDTH = 1800

SCALE = .75
CAN_SIZE_DEF = (int(1600*SCALE), int(1200*SCALE))
CANVAS = ROOT.TCanvas("c1", "", *CAN_SIZE_DEF)
ROOT.gStyle.SetPalette(112)  # set the "virdidis" color map

VALUES = {}


def clear():
    CANVAS.Clear()
    CANVAS.SetCanvasSize(*CAN_SIZE_DEF)


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
    if type(container) != str:
        container = container.GetName().split(':')[1]
    g, functions = parse(VALUES[container], container)
    try:
        return Image(g.create_gif()), show_function(dataset, functions)
    except Exception as e:
        print(e)
        print(g.to_string())


class OutputCapture:
    def __init__(self):
        self.my_stdout = io.StringIO()
        self.my_stderr = io.StringIO()

    def get_stdout(self):
        self.my_stdout.seek(0)
        return self.my_stdout.read()

    def get_stderr(self):
        self.my_stderr.seek(0)
        return self.my_stderr.read()

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self.my_stdout
        sys.stderr = self.my_stderr

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.stdout = None
        self.stderr = None


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
        # self.output_filename = self.input_filename.replace(".root", "_result.root")
        # self.conditional_recompute()
        self.load_objects()

        ResultSet.add_collection(self)

    def load_objects(self):
        file = ROOT.TFile.Open(self.input_filename)
        l = file.GetListOfKeys()
        self.map = {}
        VALUES.update(dict(file.Get("_value_lookup")))
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
        if n_plots*SINGLE_PLOT_SIZE[0] > MAX_WIDTH:
            shape_x = MAX_WIDTH//SINGLE_PLOT_SIZE[0]
            shape_y = ceil(n_plots / shape_x)
            return (shape_x, shape_y)
        else:
            return (n_plots, 1)

    def draw(self, shape=None):
        objs = [obj for obj in self.map.values() if hasattr(obj, "Draw")]
        if shape is None:
            n_plots = len(objs)
            shape = self.calc_shape(n_plots)
        CANVAS.Clear()
        CANVAS.SetCanvasSize(shape[0]*SINGLE_PLOT_SIZE[0], shape[1]*SINGLE_PLOT_SIZE[1])
        CANVAS.Divide(*shape)
        i = 1
        for hist in objs:
            CANVAS.cd(i)
            try:
                hist.SetStats(False)
            except AttributeError:
                pass
            if type(hist) in (ROOT.TH1I, ROOT.TH1F, ROOT.TH1D):
                hist.SetMinimum(0)
            hist.Draw(self.get_draw_option(hist))
            i += 1
        CANVAS.Draw()

    @staticmethod
    def get_draw_option(obj):
        obj_type = type(obj)
        if obj_type in (ROOT.TH1F, ROOT.TH1I, ROOT.TH1D):
            return ""
        elif obj_type in (ROOT.TH2F, ROOT.TH2I, ROOT.TH2D):
            return "COLZ"
        elif obj_type in (ROOT.TGraph,):
            return "A*"
        else:
            return None

    @classmethod
    def get_hist_set(cls, attrname):
        labels, hists = zip(*[(sample_name, getattr(h, attrname))
                              for sample_name, h in cls.collections.items()])
        return labels, hists

    @classmethod
    def add_collection(cls, hc):
        if not hasattr(cls, "collections"):
            cls.collections = {}
        cls.collections[hc.sample_name] = hc

    @classmethod
    def stack_hist(cls,
                   hist_name,
                   title="",
                   enable_fill=False,
                   normalize_to=0,
                   draw=False,
                   draw_canvas=True,
                   draw_option="",
                   make_legend=False,
                   _stacks={}):
        labels, hists = cls.get_hist_set(hist_name)
        if draw_canvas:
            CANVAS.Clear()
            CANVAS.SetCanvasSize(SINGLE_PLOT_SIZE[0],
                                 SINGLE_PLOT_SIZE[1])

        colors = it.cycle([ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kYellow])
        stack = ROOT.THStack(hist_name+"_stack", title)
        if labels is None:
            labels = [hist.GetName() for hist in hists]
        if type(normalize_to) in (int, float):
            normalize_to = [normalize_to]*len(hists)
        ens = enumerate(zip(hists, labels, colors, normalize_to))
        for i, (hist, label, color, norm) in ens:
            hist_copy = hist
            hist_copy = hist.Clone(hist.GetName()+"_clone" + draw_option)
            hist_copy.SetTitle(label)
            if enable_fill:
                hist_copy.SetFillColorAlpha(color, 0.75)
                hist_copy.SetLineColorAlpha(color, 0.75)
            if norm:
                integral = hist_copy.Integral()
                hist_copy.Scale(norm/integral, "nosw2")
                hist_copy.SetStats(True)
            stack.Add(hist_copy)
        if draw:
            stack.Draw(draw_option)
            if make_legend:
                CANVAS.BuildLegend(0.75, 0.75, 0.95, 0.95, "")
        # prevent stack from getting garbage collected
        _stacks[stack.GetName()] = stack
        if draw_canvas:
            CANVAS.Draw()
        return stack

    @classmethod
    def stack_hist_array(cls,
                         hist_names,
                         titles,
                         shape=None, **kwargs):
        n_hist = len(hist_names)
        if shape is None:
            if n_hist <= 4:
                shape = (1, n_hist)
            else:
                shape = (ceil(sqrt(n_hist)),)*2
        CANVAS.SetCanvasSize(SINGLE_PLOT_SIZE[0]*shape[0],
                             SINGLE_PLOT_SIZE[1]*shape[1])
        CANVAS.Divide(*shape)
        for i, hist_name, title in zip(range(1, n_hist+1), hist_names, titles):
            CANVAS.cd(i)
            cls.stack_hist(hist_name, title=title, draw=True,
                           draw_canvas=False, **kwargs)
        CANVAS.cd(n_hist).BuildLegend(0.75, 0.75, 0.95, 0.95, "")

    pts = deque([], 50)

    @classmethod
    def hist_array_single(cls,
                          hist_name,
                          title=None,
                          **kwargs):
        n_hist = len(cls.collections)
        shape = cls.calc_shape(n_hist)
        CANVAS.SetCanvasSize(SINGLE_PLOT_SIZE[0]*shape[0],
                             SINGLE_PLOT_SIZE[1]*shape[1])
        CANVAS.Divide(*shape)
        labels, hists = cls.get_hist_set(hist_name)

        def pave_loc():
            hist.Get
        for i, label, hist in zip(range(1, n_hist+1), labels, hists):
            CANVAS.cd(i)
            hist.SetStats(False)
            hist.Draw(cls.get_draw_option(hist))

            pt = ROOT.TPaveText(0.70, 0.87, 0.85, 0.95, "NDC")
            pt.AddText("Dataset: "+label)
            pt.Draw()
            cls.pts.append(pt)
