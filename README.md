# Package for modelling vocal folds

This package contains tools to model coupled vocal fold (VF) motion.
Vocal fold models are based on the finite element method (FEM) while fluid models are based on 1D fluid models.

## Installation

To install this package, you will need to install the following dependencies:

- Common scientific Python packages
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `jupyter`
  - `h5py`
- Utilities
  - `ipython` (not strictly needed but useful)
  - `meshio`
  - `lxml`
  - `pyvista`
- FEniCS dependencies
  - `pybind11`
  - `cython`
  - `mpi4py`
  - `sympy`
- Numerical linear algebra software
  - `PETSc` and `petsc4py` (https://petsc.org/release/)
  - `BlockArray` (https://github.com/jon-deng/block-array)
- The 'Fenics' project, a tool for modelling finite element problems, and associated FEM tools
  - `fenics` (https://fenicsproject.org/)
  - `gmsh` (https://gmsh.info/)
- The automatic differentiation package JAX
  - `jax` (https://github.com/google/jax)
- Other miscellaneous packages
  - `nonlineq` (https://github.com/jon-deng/nonlinear-equation)
