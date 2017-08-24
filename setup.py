from setuptools import setup

with open('requirements.txt') as req:
    install_requires = req.readlines()

setup(
    name='filval',
    version='0.1',
    install_requires=install_requires,
    packages=['filval'],
    scripts=['scripts/merge.py',
             'scripts/process_parallel.py'
             ],
)
