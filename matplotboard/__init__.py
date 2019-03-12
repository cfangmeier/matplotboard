"""
    __init__.py
    The functions in this module used to declare, render, and compile figures generated with matplotlib.
"""
from __future__ import print_function

import sys
import traceback
from markdown import Markdown
from namedlist import namedlist
import matplotlib.pyplot as plt

PY3 = sys.version_info.major == 3

__all__ = ['decl_fig',
           'render',
           'generate_report',
           'configure',
           'publish']
MD = Markdown(extensions=['tables',
                          'markdown.extensions.meta',
                          'markdown.extensions.fenced_code',
                          'markdown.extensions.codehilite',
                          'markdown.extensions.toc',
                          ],
              extension_configs={
                  'markdown.extensions.codehilite': {'css_class': 'highlight',
                                                     'linenums': True}}
              )
CONFIG = {'output_dir': 'dashboard',
          'scale': 1.0,
          'multiprocess': False,
          'publish_remote': None,
          'publish_url': None,
          'publish_dir': 'published',
          'early_abort': False,
          'publish_data': None,
          }


def configure(**config):
    CONFIG.update(config)


class Figure(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.html = ''
        self.argdict = {}
        self.docs = ''
        self.render_fn = lambda *args, **kwargs: None

    def __call__(self):
        txt_raw = self.render_fn(*self.args, **self.kwargs)
        if not txt_raw:
            txt_raw = ''
        self.html = MD.convert(txt_raw)
        self.argdict = self._fn_call_to_dict(self.render_fn, *self.args, **self.kwargs)
        self.docs = self._process_docs(self.render_fn)

    @staticmethod
    def _fn_call_to_dict(fn, *args, **kwargs):
        from inspect import getargspec
        from html import escape
        pnames = list(getargspec(fn).args)
        pvals = list(args) + list(kwargs.values())
        return {escape(str(k)): escape(str(v)) for k, v in zip(pnames, pvals)}

    @staticmethod
    def _process_docs(fn):
        from inspect import getdoc
        raw = getdoc(fn)
        if raw:
            return MD.convert(raw)
        else:
            return ''


def decl_fig(fn):
    from functools import update_wrapper

    class WrappedPlotter:
        def __call__(self, *args, **kwargs):
            figure = Figure(*args, **kwargs)
            figure.render_fn = fn
            return figure

    return update_wrapper(WrappedPlotter(), fn)


def _render_one(args):
    from os.path import join
    from json import dumps
    idx, nplots, name, figure, figure_dir = args
    print('Building plot #{}/{}: {}'.format(idx+1, nplots, name))
    scale = CONFIG['scale']

    plt.gcf().set_size_inches(scale * 10, scale * 10)
    try:
        figure()
    except Exception as e:
        if CONFIG['early_abort']:
            logging.exception(e)
            sys.exit(-1)
        else:
            print('Error while building plot \'{}\'\n{}'.format(name, traceback.format_exc()),
                  file=sys.stderr)

    fig_fname = join(figure_dir, name+'.png')
    plt.savefig(fig_fname)
    plt.close()
    with open(join(figure_dir, name+'.docs.html'), 'w') as f:
        f.write(figure.docs)
    with open(join(figure_dir, name+'.retval.html'), 'w') as f:
        f.write(figure.html)
    with open(join(figure_dir, name+'.args.json'), 'w') as f:
        plain_args = {}
        for key, val in figure.argdict.items():
            plain_args[str(key)] = str(val)
        f.write(dumps(plain_args))
    return True


def makedirs(path):
    from os import makedirs as osmakedirs
    if PY3:
        osmakedirs(path, exist_ok=True)
    else:
        try:
            osmakedirs(path)
        except OSError:  # raised from file exists
            pass


def render(figures, titles=None, build=True, ncores=None):
    from shutil import rmtree, copytree
    from os.path import join, dirname, abspath
    from json import loads
    from pathos.multiprocessing import Pool, cpu_count
    RenderedFigure = namedlist('RenderedFigure', 'name fig_fname title argdict docs html idx')

    pkg_dir = dirname(abspath(__file__))
    output_dir = CONFIG['output_dir']
    figure_dir = join(output_dir, 'figures')

    if build:
        rmtree(output_dir, ignore_errors=True)
        makedirs(output_dir)
        # copytree(join(pkg_dir, 'static', 'js'), join(output_dir, 'js'))
        copytree(join(pkg_dir, 'static', 'css'), join(output_dir, 'css'))
        copytree(join(pkg_dir, 'static', 'icons'), join(output_dir, 'icons'))
        makedirs(join(output_dir, 'aux_figures'))
        makedirs(figure_dir)

        nplots = len(figures)
        args = []
        for idx, (name, figure) in enumerate(figures.items()):
            if type(figure) != Figure:
                raise TypeError(name+' must be Figure, found: ' + str(type(figure)))
            args.append((idx, nplots, name, figure, figure_dir))
        if CONFIG['multiprocess']:
            pool = Pool(ncores if ncores is not None else cpu_count())
            pool.map(_render_one, args)
        else:
            for arg in args:
                _render_one(arg)
    try:
        for idx, (name, _) in enumerate(figures.items()):
            fig_fname = join(figure_dir, name+'.png')
            with open(join(figure_dir, name+'.docs.html'), 'r') as f:
                docs = f.read()
            with open(join(figure_dir, name+'.retval.html'), 'r') as f:
                retval = f.read()
            with open(join(figure_dir, name+'.args.json'), 'r') as f:
                argdict = loads(f.read())
            title = titles[name] if titles is not None else name
            figures[name] = RenderedFigure(name, fig_fname, title, argdict, docs, retval, idx)
    except IOError:
        print("File not found, you probably need to generate the plots! (ie set refresh=True)")
        sys.exit(-1)


def generate_report(figures, title, output='report.html',
                    source=None, ana_source=None, config=None, body=None):
    import re
    from os.path import join
    from shutil import copy
    from json import dumps
    from jinja2 import Environment, PackageLoader, select_autoescape
    if PY3:
        from urllib.parse import quote
    else:
        from urllib import quote
    output_dir = CONFIG['output_dir']

    env = Environment(
        loader=PackageLoader('matplotboard', 'templates'),
        autoescape=select_autoescape(['htm', 'html', 'xml']),
    )
    env.globals.update({'quote': quote,
                        'enumerate': enumerate,
                        'zip': zip,
                        })

    ext_rex = re.compile(r'extfig::([^|]+)(\|(.*$))?', flags=re.MULTILINE)
    loc_rex = re.compile(r'locfig::([^|]+)(\|(.*$))?', flags=re.MULTILINE)
    rex = re.compile(r'fig(!?)::([a-zA-Z0-9_\-]+)(\|(.*$))?', flags=re.MULTILINE)

    if source is not None:
        with open(source, 'r') as f:
            source = f.read()
            source = '```python\n {}\n```'.format(source)
            source = MD.convert(source)

    if body is not None:
        if body.endswith(".md"):
            with open(body, "r") as f:
                body = f.read()
        for match in loc_rex.finditer(body):
            src_file = match.expand(r"\1")
            dst_file = join(output_dir, 'aux_figures',
                            src_file.replace('/', '_').replace('..', '_'))
            print(src_file, dst_file)
            copy(src_file, dst_file)

        body = loc_rex.sub(r'{{ figure("\1", "", "\3", True, True) }}', body)
        body = ext_rex.sub(r'{{ figure("\1", "", "\3", True) }}', body)
        body = rex.sub(r'{{ figure("\2", "", "\4", False, False, "\1") }}', body)
        html = MD.convert(body)

        if MD.Meta.get('slides', False):
            template = env.from_string('''
{{% extends("slides.j2")%}}
{{% block body %}}
{}
{{% endblock %}}
        '''.format(body))
        else:
            template = env.from_string('''
{{% extends("report.j2")%}}
{{% block body %}}
<p class="provenance"> {} <br> {}</p>
{}
{{% endblock %}}
        '''.format(', '.join(MD.Meta['authors']), MD.Meta['date'][0], html))
    else:

        body = dumps([figures[fig_name]._asdict()
                      for fig_name in sorted(figures.keys())])
        template = env.from_string('''
{{% extends("dump.j2")%}}
{{% block data %}}
var figures = {};
{{% endblock %}}
    '''.format(body))

    with open(join(output_dir, output), 'w') as f:
        f.write(template.render(
            figures=figures,
            title=title,
            source=source,
            ana_source=ana_source,
            config=config,
        ))


def publish():
    from datetime import datetime as dt
    from shutil import rmtree, copytree
    from os.path import join

    par_dir = CONFIG['output_dir']
    dir_name = '{}_{}'.format(par_dir, dt.strftime(dt.now(), "%Y_%m_%d_%H"))
    rmtree(dir_name, ignore_errors=True)
    copytree(par_dir, dir_name)

    if CONFIG['publish_remote'] in ('local', 'localhost'):
        makedirs(CONFIG['publish_dir'])
        pubdir = join(CONFIG['publish_dir'], dir_name)
        rmtree(pubdir, ignore_errors=True)
        copytree(dir_name, pubdir)
        if CONFIG['publish_url'] is not None:
            print('The plots are available at ' + CONFIG['publish_url']+'/' + dir_name)
    elif CONFIG['publish_remote'] is not None:
        from openssh_wrapper import SSHConnection
        print("connecting to remote server... ", end='')
        username, remote = CONFIG['publish_remote'].split('@')
        conn = SSHConnection(remote, login=username)
        print('done.')
        print("preparing destination... ", end='')
        conn.run('mkdir -p {}'.format(CONFIG["publish_dir"]))
        conn.run('rm -rf {}/{}'.format(CONFIG["publish_dir"], dir_name))
        print('done.')
        conn.timeout = 0
        print("copying plots... ", end='')
        conn.scp((dir_name, ), CONFIG["publish_dir"])
        print('done.')
        if CONFIG['publish_data'] is not None:
            print("copying data... ", end='')
            conn.scp((CONFIG['publish_data'], ), join(CONFIG["publish_dir"], dir_name, 'data'))
            print('done.')
        print("fixing permissions... ", end='')
        conn.run('chmod a+r -R {}'.format(join(CONFIG["publish_dir"], dir_name)))
        print('done.')
        if CONFIG['publish_url'] is not None:
            print('The plots are available at ' + join(CONFIG['publish_url'], dir_name))