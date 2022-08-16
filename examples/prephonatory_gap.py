"""
Solve the statics problem for a given prehonatory gap

If the prephonatory gap results in no collision, this problem is trivial but for
prephonatory gaps with collision, this results in finding the equilibrium
position of the vocal folds under contact with the collision plane.
"""

import dolfin as dfn
from femvf import models
from femvf.solverconst import DEFAULT_NEWTON_SOLVER_PRM

def solve_prephonatory_configuration(solid):
    # Set the initial guess u=0 and constants (v, a) = (0, 0)
    state = solid.state0.copy()
    state.set(0.0)
    solid.set_fin_state(state)
    solid.set_ini_state(state)

    # Set initial pressure as 0 for the static problem
    control = solid.control.copy()
    control['p'][:] = 0.0

    jac = dfn.derivative(solid.forms['form.un.f1uva'], solid.forms['coeff.state.u1'])
    dfn.solve(solid.forms['form.un.f1uva'] == 0.0, solid.forms['coeff.state.u1'],
              bcs=[solid.bc_base], J=jac, solver_parameters={"newton_solver": DEFAULT_NEWTON_SOLVER_PRM})

    u = solid.state1['u']
    return u

if __name__ == '__main__':
    ## Load the solid model
    mesh_path = '../meshes/M5-3layers-cl0_50.xml'
    solid = models.load.load_solid_model(mesh_path, models.solid.Approximate3DKelvinVoigt)

    props = solid.get_properties_vec()
    props['ycontact'][:] = solid.mesh.coordinates()[:, 1].max() + 60.0#- 0.1
    props['emod'][:] = 10e3 * 10 # factor converts [Pa] to the [cgs] equivalent
    props['nu'][:] = 0.45
    props['eta'][:] = 5.0
    props['kcontact'][()] = 1e13
    props['length'][:] = 1.5
    # props['muscle_stress'][:] = 0.0
    solid.set_props(props)

    res = dfn.assemble(solid.forms['form.un.f1uva'])

    breakpoint()
    u = solve_prephonatory_configuration(solid)
    XREF = solid.mesh.coordinates()
    U = u[solid.vert_to_vdof].reshape(-1, 2)
    XCUR = XREF+U
    # the max y coordinate should be slightly above the contact plane location if contact occurs
    print(XCUR[:, 1].max())
