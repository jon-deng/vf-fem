import unittest

import numpy as np

from femvf.models import load_fsi_model, load_fsai_model, solid, fluid
from femvf.load import load_transient_fsi_model, load_fsai_model
from femvf.parameters.parameterization import SubsetParameterization

from blocktensor import linalg

class TestParameterization(unittest.TestCase):

    def setUp(self):
        mesh_path = '../meshes/M5-3layers.xml'
        model = load_transient_fsi_model(mesh_path, None, SolidType=solid.KelvinVoigt, FluidType=fluid.Bernoulli)

        control = model.get_control_vec()
        control['psub'][:] = 8000.0

        times = np.linspace(0.0, 1e-3, 10)
        self.param = SubsetParameterization(model, 10, ('emod',))

    def test_dconvert(self):
        p0 = self.param.copy()
        dp_bvec_true = p0.bvector.copy()
        dp_bvec_true.set(1e-4)

        p1 = self.param.copy()
        p1.bvector[:] = p0.bvector + dp_bvec_true

        args0 = p0.convert()
        args1 = p1.convert()

        dargs = []
        for n, (arg0, arg1) in enumerate(zip(args0, args1)):
            if n == 1:
                dargs.append([arg1[0]-arg0[0]])
            else:
                dargs.append(arg1-arg0)

        dp_bvec = p0.dconvert(*dargs)

        err =  dp_bvec_true - dp_bvec
        err_norm = linalg.dot(err, err)
        breakpoint()

if __name__ == '__main__':
    test = TestParameterization()
    test.setUp()
    test.test_dconvert()