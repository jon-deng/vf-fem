"""
This modules contains tests for dynamical models 
"""

from os import path

from blocklinalg import linalg as bla
from femvf import dynamicalmodels as dynmod
from femvf.dynamicalmodels import solid as slmodel, fluid as flmodel
from femvf import load


mesh_name = 'M5-3layers'
mesh_path = path.join('../meshes', mesh_name+'.xml')

solid_mesh = mesh_path
fluid_mesh = None
SolidType = slmodel.KelvinVoigt
FluidType = flmodel.Bernoulli1DDynamicalSystem
model = load.load_dynamical_fsi_model(
    solid_mesh, fluid_mesh, SolidType, FluidType, 
    fsi_facet_labels=('pressure',), fixed_facet_labels=('fixed',))

if __name__ == '__main__':
    print("yoyo whatup")