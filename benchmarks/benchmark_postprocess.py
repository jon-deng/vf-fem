"""
Benchmark `femvf.postprocess`
"""

from os import path

import numpy as np

from blockarray import (blockvec as bv)

from femvf import (statefile as sf, forward)
from femvf.postprocess import (solid as ppsld, base as ppbase)

from setup import setup_model, setup_transient_args
from benchmarkutils import setup_argument_parser, benchmark

if __name__ == '__main__':
    parser = setup_argument_parser()
    args = parser.parse_args()

    model = setup_model('../meshes/M5-3layers.msh')
    state0, control, prop = setup_transient_args(model)

    times = 1e-4*np.arange(100)

    fpath = './out/.benchmark_postprocess.h5'
    if not path.isfile(fpath):
        with sf.StateFile(model, fpath, mode='w') as f:
            forward.integrate(model, f, state0, [control], prop, times, write=True)
    else:
        print("Using existing `StateFile` to post-process")

    postproc_field = ppsld.StressVonMisesField(model)
    postproc_ts_field = ppbase.TimeSeries(postproc_field)

    with sf.StateFile(model, fpath, mode='r') as f:
        statement = 'postproc_ts_field(f)'
        benchmark(
            statement, './benchmark_postprocess.profile',
            profile=args.profile, globals=globals()
        )