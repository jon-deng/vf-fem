# TODO: To test the parameterization adjoint method, you have to implement functionals that act on a
# single input state, i.e. (uva0, qp0, solid_props, fluid_props, t)
# This functional should then have methods that return sensitivities w.r.t the appropriate
# parameters.
# A special case is the time variable, since parameterization produce a list of times, rather
# than a single time instance, which will require some further though on what should be done there.

