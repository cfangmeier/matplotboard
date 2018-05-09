from setuptools import setup

with open('README.md') as f:
    desc = f.read()

setup(
    author='Caleb Fangmeier',
    author_email='caleb@fangmeier.tech',
    url='https://github.com/cfangmeier/matplotboard',
    name='matplotboard',
    version='0.5.2',
    description='Generate simple HTML dashboards using matplotlib',
    long_description=desc,
    long_description_content_type='text/markdown',
    install_requires=['matplotlib',
                      'Jinja2',
                      'Markdown',
                      'python-markdown-math',
                      'namedlist'],
    packages=['matplotboard'],
    package_data={'matplotboard': ['templates/*.j2',
                                   'static/css/*.css',
                                   'static/js/*.js',
                                   ]},
)
