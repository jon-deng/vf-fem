"""
Benchmark the performance of reading/writing from a statefile

conclusion: hdf5 chunking/caching sucks big time
"""
import os
from time import perf_counter
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dfn

import femvf as fvf
from femvf.statefile import StateFile

h5path = f'out/benchmark_statefile.h5'

mesh_path = '../meshes/M5-3layers-cl0_50.xml'
model = fvf.load_fsi_model(mesh_path, None, Solid=fvf.solids.KelvinVoigt, Fluid=fvf.fluids.Bernoulli)

def benchmark_chunksize(ntime_chunks=10):
    if os.path.isfile(h5path):
        os.remove(h5path)

    with StateFile(model, h5path, mode='w', NCHUNK=ntime_chunks) as f:
        f.init_layout()

        _x = dfn.Function(model.solid.vector_fspace).vector()
        uva = (_x, _x.copy(), _x.copy())

        qp = model.fluid.get_state_vec()

        for i in range(100):
            f.append_time(i)
            f.append_state(uva)
            f.append_fluid_state(qp)

    t_fluid = []
    t_solid = []
    with StateFile(model, h5path, mode='r', rdcc_nbytes=2**2*1024, rdcc_w0=1, rdcc_nslots=1e3) as f:
        # print(f.file.id.get_access_plist().get_cache())
        for i in range(100):

            ts = perf_counter()
            # f.get_fluid_state(i)
            f.file['q'][i, ...]
            f.file['p'][i, ...]
            t_fluid.append(perf_counter()-ts)

            ts = perf_counter()
            # f.get_state(i)
            f.file['u'][i, ...]
            f.file['v'][i, ...]
            f.file['a'][i, ...]
            t_solid.append(perf_counter()-ts)

    return t_fluid, t_solid

def benchmark_chunksize_chunked_read(ntime_chunks=10):
    if os.path.isfile(h5path):
        os.remove(h5path)

    with StateFile(model, h5path, mode='w', NCHUNK=ntime_chunks) as f:
        f.init_layout()

        _x = dfn.Function(model.solid.vector_fspace).vector()
        uva = (_x, _x.copy(), _x.copy())

        qp = model.fluid.get_state_vec()

        for i in range(100):
            f.append_time(i)
            f.append_state(uva)
            f.append_fluid_state(qp)

    t_fluid = []
    t_solid = []
    with StateFile(model, h5path, mode='r', rdcc_nbytes=2**2*1024, rdcc_w0=1, rdcc_nslots=1e3) as f:
        # print(f.file.id.get_access_plist().get_cache())
        for m in range(ceil(100/ntime_chunks)):

            ts = perf_counter()
            start, stop = m*ntime_chunks, min((m+1)*ntime_chunks, 100)
            print(start, stop)
            f.file['q'][start:stop, ...]
            f.file['p'][start:stop, ...]
            t_fluid.append(perf_counter()-ts)

            ts = perf_counter()
            f.file['u'][start:stop, ...]
            f.file['v'][start:stop, ...]
            f.file['a'][start:stop, ...]
            t_solid.append(perf_counter()-ts)

    return t_fluid, t_solid

if __name__ == '__main__':
    tfluids = []
    tsolids = []
    for n in [1, 10, 50, 100]:
        tfluid, tsolid = benchmark_chunksize_chunked_read(n)
        tfluids.append(tfluid)
        tsolids.append(tsolid)

        print(f"{n} chunks, total time to read all data:  {np.sum(tfluid)} s, {np.sum(tsolid)} s")

    fig, ax = plt.subplots(1, 1)
    for ii, n in enumerate([1, 10, 50, 100]):
        ax.plot(tsolids[ii], label=f"{n}")

    fig.legend()
    plt.show()
