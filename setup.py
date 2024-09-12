import os
import setuptools
import subprocess

version = subprocess.check_output(['git', 'describe','--tag', '--abbrev=0']).decode('ascii').strip()

setuptools.setup(
    name="neurom",
    version=version,
    author="Martin Genet",
    author_email="martin.genet@polytechnique.edu",
    description=open("README.md", "r").readlines()[1][:-1],
    long_description = open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AlexandreDabySeesaram/NeuROM",
    packages=setuptools.find_packages(include=['neurom', 'neurom.*']),                            # Automatically finds neurom and its subpackages
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["argparse","pytorch", "matplotlib", "meshio", "numpy"],
)
