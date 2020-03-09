from setuptools import setup
from distutils.util import convert_path

with open("README.md") as f:
    desc = f.read()


with open("requirements.txt") as f:
    requirements = f.readlines()

# Below is to acquire the version without actually importing the package.
main_ns = {}
ver_path = convert_path("matplotboard/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    author="Caleb Fangmeier",
    author_email="caleb@fangmeier.tech",
    url="https://github.com/cfangmeier/matplotboard",
    name="matplotboard",
    version=main_ns["__version__"],
    description="Generate simple HTML dashboards using matplotlib",
    long_description=desc,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=["matplotboard"],
    entry_points={"console_scripts": ["mpb=matplotboard.__main__:main"]},
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
