# Package for modelling vocal folds

This package contains tools to model coupled vocal fold (VF) motion.
Vocal fold models are based on the finite element method (FEM) while fluid models are based on 1D fluid models.

## Installation

To install this package, you will need to install the following dependencies:

- The 'Fenics' project, a tool for modelling finite element problems, and associated FEM tools
  - `fenics` (https://fenicsproject.org/)
  - `fenics` dependencies
    - `petsc4py`
    - `slepc4py`
- Common scientific Python packages
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `h5py`
  - `jupyter`
- Automatic differentiation
  - `jax` (https://github.com/google/jax)
- Meshing
  - `meshio`
  - `gmsh`
- Visualization
  - `pyvista`
- Utilities
  - `pytest`
  - `lxml`
  - `tqdm`
- Other libraries
  - `BlockArray` (https://github.com/jon-deng/block-array)
  - `nonlineq` (https://github.com/jon-deng/nonlinear-equation)
