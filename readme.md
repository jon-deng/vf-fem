This project implements a collection of vocal fold (VF) finite element (FE) models and 1D fluid
models that can be coupled together to simulated VF flow induced self-oscillation.

Todo
--------
- Implement a block vector to store tuples of states, for example, (u, v, a) or (q, p)
- Implement a newton method for solving the fully coupled FSI problem
- Refactor the class definitions for functionals. I have a feeling many definitions included in the
  functional are not needed.
- Implement functionality to allow functional objects to be added/multiplied/etc. together. This is
  commonly used, for example, in forming penalty functional when one functional is the objective
  and another functional acts as a penalty.
- Formalize/refactor how optimization progress is saved
- Add post-processing functionality to quickly evaluate optimization results
