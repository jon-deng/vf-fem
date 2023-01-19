"""
Benchmark/profile time integration of the forward model
"""


import numpy as np

from blockarray.blockvec import BlockVector

from femvf.forward import integrate
from femvf import statefile as sf

from setup import setup_model, setup_transient_args
from benchmarkutils import setup_argument_parser, benchmark

def run_forward(model, state, control, prop, times):
    """
    Integrate the forward model over time
    """
    with sf.StateFile(model, 'out/test.h5', mode='w') as f:
        return integrate(model, f, state, [control], prop, times, write=False)

if __name__ == '__main__':
    parser = setup_argument_parser()
    args = parser.parse_args()

    model = setup_model('../meshes/M5-3layers.msh')
    state0, control, prop = setup_transient_args(model)

    _times = 1e-4*np.arange(100)
    times = BlockVector((_times,), (1,))

    statement = 'run_forward(model, state0, control, prop, times)'
    benchmark(
        statement, './benchmark_forward.profile',
        profile=args.profile, globals=globals()
    )
