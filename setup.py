from setuptools import setup

with open('requirements.txt') as req:
    install_requires = [l.strip() for l in req.readlines()]

with open('README.md') as f:
    readme = f.read()

setup(
    name='matplotboard',
    version='0.2.1',
    description='Generate html dashboards using matplotlib, Jinja2, and Markdown.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Caleb Fangmeier',
    author_email='caleb@fangmeier.tech',
    url='https://github.com/cfangmeier/matplotboard/',
    keywords=['Markdown', 'Jinja2', 'matplotlib', 'dashboard'],
    packages=['matplotboard'],
    package_data={'matplotboard': ['templates/*.j2',
                                   'static/css/*.css',
                                   'static/js/*.js',
                                   ]},
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Internet :: WWW/HTTP :: Site Management',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Markup :: HTML',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=install_requires,
    dependency_links=[
        "git+ssh://git@github.com/cfangmeier/latexipy.git#egg=latexipy"
    ],
)
