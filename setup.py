from setuptools import setup

with open('requirements.txt') as req:
    install_requires = [l.strip() for l in req.readlines()]

with open('README.md') as f:
    desc = f.read()

setup(
    name='matplotboard',
    version='0.4.1',
    description='Generate simple HTML dashboards using matplotlib',
    long_description=desc,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    dependency_links=[
        "git+ssh://git@github.com/cfangmeier/latexipy.git#egg=latexipy"
    ],
    packages=['matplotboard'],
    package_data={'matplotboard': ['templates/*.j2',
                                   'static/css/*.css',
                                   'static/js/*.js',
                                   ]},
)
