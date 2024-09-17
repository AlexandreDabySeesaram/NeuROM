import os
import setuptools
import subprocess

version = subprocess.check_output(['git', 'describe','--tag', '--abbrev=0']).decode('ascii').strip()

setuptools.setup(
    name="NeuROM-Py",
    version=version,
    author="Alexandre Daby-Seesaram, Katerina Skardova, Martin Genet",
    author_email="[alexandre.daby-seesaram,katerina.skardova,martin.genet]@polytechnique.edu",
    description="NeuROM, a NN-PGD architecture based on the HiDeNN Framework (Finite Element Neural Network Interpolation)",
    long_description = open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AlexandreDabySeesaram/NeuROM",
    package_dir = {
    'neurom.src': 'neurom/src',
    'neurom.Post': 'neurom/Post'},
    packages=['neurom', 'neurom.src', 'neurom.Post'],                            # Automatically finds neurom and its subpackages
    entry_points={
        'console_scripts': [
            'neurom = neurom:main',  
        ],
    },
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["argparse","torch", "matplotlib", "meshio", "numpy", "ipython", "scipy", "gmsh", "vtk", "pyvista"],
)
