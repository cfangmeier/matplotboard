from setuptools import setup
from matplotboard import __version__

with open("README.md") as f:
    desc = f.read()


with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    author="Caleb Fangmeier",
    author_email="caleb@fangmeier.tech",
    url="https://github.com/cfangmeier/matplotboard",
    name="matplotboard",
    version=__version__,
    description="Generate simple HTML dashboards using matplotlib",
    long_description=desc,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=["matplotboard"],
    entry_points={"console_scripts": ["mbp=matplotboard:main"]},
    package_data={
        "matplotboard": [
            "templates/*.j2",
            "static/css/*.css",
            "static/js/*.js",
            "static/icons/*",
        ]
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Internet :: WWW/HTTP :: Site Management",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup :: HTML",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
