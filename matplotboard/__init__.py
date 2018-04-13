"""
    __init__.py
    The functions in this module are meant for plotting the histogram objects created via
    matplotboard.histogram
"""

import re
from io import BytesIO
from base64 import b64encode
from markdown import Markdown
import latexipy as lp


__all__ = ['decl_fig',
           'render',
           'generate_report']


MD = Markdown(extensions=['mdx_math', 'tables'],
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


def decl_fig(fn):
    from functools import wraps

    def _fn_call_to_dict(fn, *args, **kwargs):
        from inspect import signature
        from html import escape
        pnames = list(signature(fn).parameters)
        pvals = list(args) + list(kwargs.values())
        return {escape(str(k)): escape(str(v)) for k, v in zip(pnames, pvals)}

    def _process_docs(fn):
        from inspect import getdoc
        raw = getdoc(fn)
        if raw:
            return MD.convert(raw)
        else:
            return None

    @wraps(fn)
    def f(*args, **kwargs):
        global _decl_counter
        txt = fn(*args, **kwargs)
        argdict = _fn_call_to_dict(fn, *args, **kwargs)
        docs = _process_docs(fn)
        if not txt:
            txt = ''
        html = MD.convert(txt)

        return argdict, docs, html

    return f


def render(figures, scale=1.0):
    from namedlist import namedlist

    Figure = namedlist('Figure', 'name data argdict docs html idx')

    def exec_fig(fig):
        if not isinstance(fig, tuple):
            fn, args, kwargs = fig, (), {}
        elif len(fig) == 1:
            fn, args, kwargs = fig[0], (), {}
        elif len(fig) == 2:
            fn, args, kwargs = fig[0], fig[1], {}
        elif len(fig) == 3:
            fn, args, kwargs = fig[0], fig[1], fig[2]
        else:
            raise ValueError('Plot tuple must be of format (func), '
                             f'or (func, tuple), or (func, tuple, dict). Got {fig}')
        return fn(*args, **kwargs)

    for idx, (name, figure) in enumerate(figures.items()):
        print(f'Building plot #{idx}: {name}')
        out = BytesIO()
        with lp.mem_figure(out,
                           ext='png',
                           size=(scale * 10, scale * 10)):
            argdict, docs, html = exec_fig(figure)
        out.seek(0)
        figures[name] = Figure(name, out, argdict, docs, html, idx)


def generate_report(figures, title, outputdir='report',
                    source=None, ana_source=None, config=None, body=None):
    from os.path import join, dirname, abspath
    from os import mkdir
    from shutil import rmtree, copytree
    from jinja2 import Environment, PackageLoader, select_autoescape, Template
    from urllib.parse import quote

    if body is None:
        raise ValueError("You must supply the body of the report!")

    env = Environment(
        loader=PackageLoader('matplotboard', 'templates'),
        autoescape=select_autoescape(['htm', 'html', 'xml']),
    )
    env.globals.update({'quote': quote,
                        'enumerate': enumerate,
                        'zip': zip,
                        })

    if source is not None:
        with open(source, 'r') as f:
            source = f.read()

    rmtree(outputdir)
    mkdir(outputdir)
    pkgdir = dirname(abspath(__file__))
    copytree(join(pkgdir, 'static', 'js'), join(outputdir, 'js'))
    copytree(join(pkgdir, 'static', 'css'), join(outputdir, 'css'))
    figure_dir = join(outputdir, 'figures')
    mkdir(figure_dir)

    for name, figure in figures.items():
        fname = join(figure_dir, f'{figure.name}.png')
        with open(fname, 'wb') as f:
            f.write(figure.data.read())

    body = re.sub(r'fig::(\w+)', r'{{ fig(figures["\1"]) }}', body)
    body = MD.convert(body)

    report_template = env.from_string(f'''
{{% extends("report.j2")%}}
{{% from 'macros.j2' import fig %}}
{{% block body %}}
{body}
{{% endblock %}}''')

    with open(join(outputdir, 'report.html'), 'w') as f:
        f.write(report_template.render(
            title=title,
            figures=figures,
            source=source,
            ana_source=ana_source,
            config=config,
            ))


# def hists_to_table(hists, row_labels=(), column_labels=(), format="{:.2f}"):
#     table = ['<table class="table table-condensed">']
#     if column_labels:
#         table.append('<thead><tr>')
#         if row_labels:
#             table.append('<th></th>')
#         table.extend(f'<th>{label}</th>' for label in column_labels)
#         table.append('</tr></thead>')
#     table.append('<tbody>\n')
#     for row_label, (vals, *_) in zip_longest(row_labels, hists):
#         table.append('<tr>')
#         if row_label:
#             table.append(f'<td><strong>{row_label}</strong></td>')
#         table.extend(('<td>'+format.format(val)+'</td>') for val in vals)
#         table.append('</tr>\n')
#     table.append('</tbody></table>')
#     return ''.join(table)

