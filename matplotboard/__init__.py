"""
    __init__.py
    The functions in this module are meant for plotting the histogram objects created via
    matplotboard.histogram
"""

from markdown import Markdown
from namedlist import namedlist
import matplotlib.pyplot as plt


__all__ = ['decl_fig',
           'render',
           'generate_report',
           'configure']


MD = Markdown(extensions=['mdx_math', 'tables'],
              extension_configs={'mdx_math': {'enable_dollar_delimiter': True}})
CONFIG = {'output_dir': 'dashboard',
          'scale': 1.0,
          'multiprocess': False}


def configure(**config):
    CONFIG.update(config)


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
            return ''

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


def _exec_fig(fig):
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


def _render_one(args):
    from os.path import join
    from json import dumps
    idx, nplots, name, figure, figure_dir = args;
    print(f'Building plot #{idx+1}/{nplots}: {name}')
    scale = CONFIG['scale']

    plt.gcf().set_size_inches(scale * 10, scale * 10)
    argdict, docs, retval = _exec_fig(figure)
    fig_fname = join(figure_dir, f'{name}.png')
    plt.savefig(fig_fname)
    plt.close()
    with open(join(figure_dir, f'{name}.docs.html'), 'w') as f:
        f.write(docs)
    with open(join(figure_dir, f'{name}.retval.html'), 'w') as f:
        f.write(retval)
    with open(join(figure_dir, f'{name}.args.json'), 'w') as f:
        plain_args = {}
        for key, val in argdict.items():
            plain_args[str(key)] = str(val)
        f.write(dumps(plain_args))


def render(figures, titles=None, refresh=True, ncores=None):
    from shutil import rmtree, copytree
    from os import makedirs
    from os.path import join, dirname, abspath
    from json import loads
    from multiprocessing import Pool, cpu_count
    Figure = namedlist('Figure', 'name fig_fname title argdict docs html idx')

    pkg_dir = dirname(abspath(__file__))
    output_dir = CONFIG['output_dir']
    figure_dir = join(output_dir, 'figures')

    if refresh:
        rmtree(output_dir, ignore_errors=True)
        makedirs(output_dir, exist_ok=True)
        copytree(join(pkg_dir, 'static', 'js'), join(output_dir, 'js'))
        copytree(join(pkg_dir, 'static', 'css'), join(output_dir, 'css'))
        copytree(join(pkg_dir, 'static', 'icons'), join(output_dir, 'icons'))
        makedirs(figure_dir, exist_ok=True)

        nplots = len(figures)
        args = ((idx, nplots, name, figure, figure_dir) for idx, (name, figure) in enumerate(figures.items()))
        if CONFIG['multiprocess']:
            pool = Pool(ncores if ncores is not None else cpu_count())
            pool.map(_render_one, args)
        else:
            for arg in args:
                _render_one(arg)
        # for idx, (name, figure) in enumerate(figures.items()):
        #
        #     print(f'Building plot #{idx+1}/{nplots}: {name}')
        #
        #     plt.gcf().set_size_inches(scale * 10, scale * 10)
        #     argdict, docs, retval = _exec_fig(figure)
        #     fig_fname = join(figure_dir, f'{name}.png')
        #     plt.savefig(fig_fname)
        #     plt.close()
        #     with open(join(figure_dir, f'{name}.docs.html'), 'w') as f:
        #         f.write(docs)
        #     with open(join(figure_dir, f'{name}.retval.html'), 'w') as f:
        #         f.write(retval)
        #     with open(join(figure_dir, f'{name}.args.json'), 'w') as f:
        #         plain_args = {}
        #         for key, val in argdict.items():
        #             plain_args[str(key)] = str(val)
        #         f.write(dumps(plain_args))

    try:
        for idx, (name, _) in enumerate(figures.items()):
            fig_fname = join(figure_dir, f'{name}.png')
            with open(join(figure_dir, f'{name}.docs.html'), 'r') as f:
                docs = f.read()
            with open(join(figure_dir, f'{name}.retval.html'), 'r') as f:
                retval = f.read()
            with open(join(figure_dir, f'{name}.args.json'), 'r') as f:
                argdict = loads(f.read())
            title = titles[name] if titles is not None else name
            figures[name] = Figure(name, fig_fname, title, argdict, docs, retval, idx)
    except FileNotFoundError as e:
        print("File not found, you probably need to generate the plots! (ie set refresh=True)")
        raise e


def generate_report(figures, title, output='report.html',
                    source=None, ana_source=None, config=None, body=None):
    from os.path import join
    from json import dumps
    from jinja2 import Environment, PackageLoader, select_autoescape
    from urllib.parse import quote
    output_dir = CONFIG['output_dir']

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

    # if body is not None:
    #     body = re.sub(r'fig::(\w+)', r'{{ fig(figures["\1"]) }}', body)
    #     body = MD.convert(body)
    # else:
    #     body = '\n'.join(f'{{{{ fig(figures["{fig_name}"]) }}}}' for fig_name in figures)

    body = dumps([fig._asdict() for fig in figures.values()])

    report_template = env.from_string(f'''
{{% extends("report.j2")%}}
{{% block data %}}
var figures = {body};
{{% endblock %}}''')

    with open(join(output_dir, output), 'w') as f:
        f.write(report_template.render(
            title=title,
            source=source,
            ana_source=ana_source,
            config=config,
            ))
