model = load_fsi_model(mesh_path, None, Solid=solid.KelvinVoigt, Fluid=fluid.Bernoulli)

output_dir = f'periodic_solution_{optimizer}-retryfortesting'

xdmfutils.export_vertex_values(model, 'periodic_optimum_run.h5', 'periodic_optimum_run-vis.h5')
xdmfutils.write_xdmf(model, 'periodic_optimum_run-vis.h5', )