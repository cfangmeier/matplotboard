from setuptools import setup

with open('requirements.txt') as req:
    install_requires = [l.strip() for l in req.readlines()]

print(install_requires)

setup(
    name='filval',
    version='0.1',
    install_requires=install_requires,
    dependency_links=[
        "git+ssh://git@github.com/cfangmeier/latexipy.git#egg=latexipy"
    ],
    packages=['filval'],
    scripts=['scripts/merge.py',
             'scripts/process_parallel.py'
             ],
    package_data={'filval': ['templates/*.j2']},
)
