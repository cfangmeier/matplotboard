from setuptools import setup

with open('README.md') as f:
    desc = f.read()

setup(
    author='Caleb Fangmeier',
    author_email='caleb@fangmeier.tech',
    url='https://github.com/cfangmeier/matplotboard',
    name='matplotboard',
    version='0.9.1',
    description='Generate simple HTML dashboards using matplotlib',
    long_description=desc,
    long_description_content_type='text/markdown',
    install_requires=['matplotlib',
                      'Jinja2',
                      'Markdown',
                      'python-markdown-math',
                      'namedlist',
                      'openssh-wrapper',
                      'pathos'],
    packages=['matplotboard'],
    package_data={'matplotboard': ['templates/*.j2',
                                   'static/css/*.css',
                                   'static/js/*.js',
                                   'static/icons/*',
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
)
