import ROOT

__all__ = ["pdg", "show_function", "show_value"]

db = ROOT.TDatabasePDG()


class PDGParticle:

    def __init__(self, tPart):
        self.pdgId = tPart.PdgCode()
        self.name = tPart.GetName()
        self.charge = tPart.Charge() / 3.0
        self.mass = tPart.Mass()
        self.spin = tPart.Spin()

    def __repr__(self):
        return (f"<PDGParticle {self.name}:"
                f"pdgId={self.pdgId}, charge={self.charge}, mass={self.mass:5.4e} GeV, spin={self.spin}>")


def pdg(pdg_id):
    try:
        return PDGParticle(db.GetParticle(pdg_id))
    except ReferenceError:
        raise ValueError(f"unknown pdgId: {pdg_id}")


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
    from graph_vals import parse
    if type(container) != str:
        container = container.GetName().split(':')[1]
    g, functions = parse(dataset.values[container], container)
    try:
        return Image(g.create_gif()), show_function(dataset, functions)
    except Exception as e:
        print(e)
        print(g.to_string())
