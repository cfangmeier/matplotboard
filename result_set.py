import ROOT

from plotter import plot_histogram, plot_histogram2d


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
