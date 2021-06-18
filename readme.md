This project implements a collection of vocal fold (VF) finite element (FE) models and 1D fluid
models that can be coupled together to simulated VF flow induced self-oscillation.

TODO
--------
- [] Refactor Bernoulli model!
     The Bernoulli model is written in a super hard to understand way, for example, using the 
     surface displacements instead of areas. You should figure out the core inputs of the model
     and refactor it around those inputs. One change will be to modify the displacement inputs
     to cross-sectional areas.
- [] Implement a general interface to compute some summary quantity from a state. This would be very useful for post-processing as I always compute some quantities as a function of time.

- [] Refactor logging. I'm pretty sure the way logging is currently used is not proper and will lead
     to weird behaviour.
     
- [] Implement a block matrix format to represent block operations on states (u, v, a, q, p, etc.)
- [x] Implement a block vector to store tuples of states, for example, (u, v, a) or (q, p)
- [] Speed up block matrix formation/matrix permutation methods
- [x] Implement a newton method for solving the fully coupled FSI problem

- [] Refactor the class definitions for functionals. I have a feeling many definitions included in the
  functional are not needed.
- [x] Implement functionality to allow functional objects to be added/multiplied/etc. together. This is
  commonly used, for example, in forming penalty functional when one functional is the objective
  and another functional acts as a penalty.


- [] Create a generic parameterization that includes all variable parameters. Other parameterizations
     can then be derived from this generic one by removing known parameters, or specifying constant
     values.

- [] Add post-processing functionality to quickly evaluate optimization results
- [] Formalize/refactor how optimization progress is saved
- [x] Refactor hard coded callback functions out of objective function object
- [x] Fix bug where exceptions are 'hidden' somehow in the optimization loop; this occurs when an
     exception such as accessing missing keys just gets ignored and doesn't cause the optimization
     code to throw an error and stop. This one is very confusing.
     - The source of the bug is the fact that cyipopt does ignores exceptions. If cyipopt calls the
     `objective` method and an exception is raised, cyipopt will internally just try the next
     thing according to what ipopt want. I believe in this case, the function that was called
     counts as returning `None` and ipopt considers this to be a bad point and tries something else.
     I think to fix this bug, you need to return a special value that notifies ipopt that something
     is very wrong, not just that the function couldn't compute at a bad point.
