"""
Generic code for setting up commonly used components in benchmarking
"""

from typing import Tuple

from femvf.load import load_transient_fsi_model
from femvf.models.transient import (fluid as tfmd, solid as tsmd, base as tbase)

from blockarray import (blockvec as bv)

def setup_model(mesh_path: str) -> tbase.BaseTransientModel:
    """
    Return a common model to integrate
    """
    model = load_transient_fsi_model(
        mesh_path, None,
        SolidType=tsmd.KelvinVoigtWEpithelium,
        FluidType=tfmd.BernoulliAreaRatioSep,
        fsi_facet_labels=['pressure'],
        fixed_facet_labels=['fixed'],
        coupling='explicit'
    )
    return model

def setup_transient_args(
        model: tbase.BaseTransientModel
    ) -> Tuple[bv.BlockVector, bv.BlockVector, bv.BlockVector]:
    """
    Return a (initial state, control, properties) tuple
    """
    state0 = model.state0.copy()
    state0[:] = 0

    control = model.control.copy()
    control[:] = 0
    control['psub'] = 8e3

    prop = model.prop.copy()
    ymax = model.solid.forms['mesh.mesh'].coordinates()[:, 1].max()
    prop['emod'] = 5e4
    prop['rho'] = 1
    prop['eta'] = 3
    prop['nu'] = 0.45
    prop['ycontact'] = ymax+0.05
    prop['kcontact'] = 1e8
    return state0, control, prop
