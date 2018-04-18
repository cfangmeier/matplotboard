"""
    __init__.py
    The functions in this module are meant for plotting the histogram objects created via
    matplotboard.histogram
"""

import re
from io import BytesIO
from markdown import Markdown
from namedlist import namedlist
import matplotlib.pyplot as plt


__all__ = ['decl_fig',
           'render',
           'generate_report']


MD = Markdown(extensions=['mdx_math', 'tables'],
              extension_configs={'mdx_math': {'enable_dollar_delimiter': True}})


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
        txt = fn(*args, **kwargs)
        argdict = _fn_call_to_dict(fn, *args, **kwargs)
        docs = _process_docs(fn)
        if not txt:
            txt = ''
        html = MD.convert(txt)

        return argdict, docs, html

    return f


def render(figures, scale=1.0):
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
        plt.gcf().set_size_inches(scale * 10, scale * 10)
        argdict, docs, html = exec_fig(figure)
        plt.savefig(out, ext='png')
        plt.close()
        out.seek(0)
        figures[name] = Figure(name, out, argdict, docs, html, idx)


def generate_report(figures, title, output_dir='report', output_file='report.html',
                    source=None, ana_source=None, config=None, body=None,
                    delete_old=True):
    from os.path import join, dirname, abspath
    from os import mkdir
    from shutil import rmtree, copytree
    from jinja2 import Environment, PackageLoader, select_autoescape
    from urllib.parse import quote

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

    pkg_dir = dirname(abspath(__file__))
    figure_dir = join(output_dir, 'figures')
    if delete_old:
        rmtree(output_dir, ignore_errors=True)
    try:
        mkdir(output_dir)
        copytree(join(pkg_dir, 'static', 'js'), join(output_dir, 'js'))
        copytree(join(pkg_dir, 'static', 'css'), join(output_dir, 'css'))
        mkdir(figure_dir)
    except FileExistsError:
        pass

    for name, figure in figures.items():
        fname = join(figure_dir, f'{figure.name}.png')
        with open(fname, 'wb') as f:
            f.write(figure.data.read())

    if body is not None:
        body = re.sub(r'fig::(\w+)', r'{{ fig(figures["\1"]) }}', body)
        body = MD.convert(body)
    else:
        body = '\n'.join(f'{{{{ fig(figures["{fig_name}"], own_row=False, hide_info=True) }}}}' for fig_name in figures)

    report_template = env.from_string(f'''
{{% extends("report.j2")%}}
{{% from 'macros.j2' import fig %}}
{{% block body %}}
{body}
{{% endblock %}}''')

    with open(join(output_dir, output_file), 'w') as f:
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
