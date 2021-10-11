import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stad",
    version="2.0.1",
    author="Jan Aerts",
    author_email="jan.aerts@uhasselt.be",
    description="Dimensionality reduction through Simplified Topological Abstraction of Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vda-lab/pySTAD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3',
    install_requires=[
        "python-igraph",
        "panel",
        "vega"
    ]
)
