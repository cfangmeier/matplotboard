from setuptools import setup

with open('requirements.txt') as req:
    install_requires = [l.strip() for l in req.readlines()]

setup(
    name='matplotboard',
    version='0.3.0',
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
