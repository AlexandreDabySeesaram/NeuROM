# NeuROM, a NN-PGD architecture based on the HiDeNN Framework

## ⚠️ Important Notice

The code is currently under a complete **refactor**. If you want to work on the version associated with the publications, it is advised to use the [latest release](https://github.com/AlexandreDabySeesaram/NeuROM/releases/tag/v4.0.5.post0). To clone it and work from there:
```bash
git clone https://github.com/AlexandreDabySeesaram/NeuROM.git --branch v4.0.5.post0 source
cd source
git checkout -B my-branch
```

You can also use the package available on the [`PyPI` repo](https://pypi.org/project/neurom/). To install it in a `conda` environment: 
```bash
conda create -n neurom-env python=3.12
pip3 install neurom-py
```

To improve performances it is recommanded to further install the optimisation of einsum using conda as follows
```bash
conda install opt_einsum -c conda-forge
```

## Project's description

<h1 align="center">
<img src="https://alexandredabyseesaram.github.io/Resources/Images/NeuROM_ter.svg" width="350">
</h1><br>

<!-- ![NeuROM logo](Illustrations/NeuROM_logo_sansserif.png)  -->

[![GitHub license](https://img.shields.io/github/license/alexandredabyseesaram/neurom)](https://github.com/alexandredabyseesaram/neurom)
[![PyPI Downloads](https://img.shields.io/pypi/dm/NeuROM-Py.svg?label=PyPI%20downloads)](
https://pypi.org/project/NeuROM-Py/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13772741.svg)](https://doi.org/10.5281/zenodo.13772741)


 `NeuROM` provides an interpretable deep neural network architecture that uses tensor decomposition to give a parametric reduced-order model. This network interpretability comes from the HiDeNN architecture that offers an interpolation framework through a deep neural network in which weights and biases are constrained so that the interpolation matches a finite element interpolation (P1 or P2). 
The first hidden layers play the role of the shape functions, while the last layer, called the interpolation layer in the remainder of the document, utilises the output of the shape functions to interpolate the output. Training the weights of that last hidden layer is the same as solving a FEM problem on a fixed mesh. The weights of the interpolation layer directly correspond to the nodal values associated with each shape function. Therefore, prescribing Dirichlet boundary conditions is straightforward by freezing the weights associated with the prescribed values of fixed DoFs. However, learning the parameters related to the first layers accounts for mesh adaptation.


This code implements a Finite Element Neural Network Interpolation (FENNI) based on the HiDeNN framework. The layer's input is the coordinate $\underline{x}$ where the output is evaluated and the value of the parameters $\underline{\mu}$ for which the output is computed. In this case, the output of the network is the displacement $\underline{u}\left(\underline{x},\underline{\mu}\right)$

## Installation

NeuROM uses [`uv`](https://docs.astral.sh/uv/) as a package and project manager. To install the code from source:
```bash
uv pip install -e .
```

## Licensing

  Copyright (C) 2024, UMR 7649, École Polytechnique, IPP, CNRS, MΞDISIM Team, INRIA
 
  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
 
