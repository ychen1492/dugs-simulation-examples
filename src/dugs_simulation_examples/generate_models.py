import math
import os
import pickle
import pyvista as pv
import numpy as np
from darts.engines import redirect_darts_output
from model import Model


def generate_poro_normalized_distribution(task_id, size=50):
     # Define mean and standard deviation
    mean = 0.15
    std_dev = 0.05

    np.random.seed(int(task_id))
    while True:
        porosity = np.random.normal(mean, std_dev, size)
        porosity = porosity[(porosity >= 0.04) & (porosity <= 0.28)]
        if len(porosity) > 0:
            break

    return porosity


def upscale_porosity(poro, nz):
    # Use linear interpolation to upscale the array
    upscaled_array = np.interp(np.linspace(0, 1, nz),
                           np.linspace(0, 1, poro.size),
                           poro)
    

    return upscaled_array


def stratified_model_simulation(nx, ny, nz, dx, dy, dz, n_points=32,overburden=0,total_time=30*365, rate=None):
    redirect_darts_output(f'log_str_rate_{rate}.txt')
    org_poro = generate_poro_normalized_distribution(100)
    po = upscale_porosity(org_poro, nz)

    poros = np.concatenate([np.ones(nx * ny) * p for p in po], axis=0)

    org_perm = np.array([pow(10, x) for x in
                         (-3.523e-7) * (poros * 100) ** 5 +
                         4.278e-5 * (poros * 100) ** 4 -
                         1.723e-3 * (poros * 100) ** 3 +
                         1.896e-2 * (poros * 100) ** 2 +
                         0.333 * (poros * 100) -
                         3.222])
    perms = org_perm

    proxy_model = Model(set_nx=nx, set_ny=ny, set_nz=nz, set_dx=dx, set_dy=dy, set_dz=dz, perms=perms, poro=poros,
                        overburden=overburden, n_points=n_points, total_time=total_time)
  
    proxy_model.init()

    return proxy_model


def heterogeneous_model_simulation(nx, ny, nz, dx, dy, dz,n_points=32, overburden=0, total_time=30*365, rate=None):
    redirect_darts_output(f'log_he_rate_{rate}.txt')
    cur_path = os.path.dirname(__file__)
    path_to_pickle = os.path.join(cur_path, "..", "..","poro_he_new.pkl")
    perms = lambda poro: ((-3.523e-7) * (poro * 100) ** 5 +
                          4.278e-5 * (poro * 100) ** 4 -
                          1.723e-3 * (poro * 100) ** 3 +
                          1.896e-2 * (poro * 100) ** 2 +
                          0.333 * (poro * 100) -
                          3.222)
    with open(path_to_pickle, 'rb') as por:
        org_porosity = pickle.load(file=por)

    shape = (500, 700, 50)

    spacing = (4500/shape[0], 4200/shape[1], 100/shape[2])
    origin = (spacing[0] / 2., spacing[1] / 2., -2300)
    x = np.linspace(origin[0] - spacing[0] / 2., origin[0] + spacing[0] * (shape[0] - 0.5), shape[0] + 1)
    y = np.linspace(origin[1] - spacing[1] / 2., origin[1] + spacing[1] * (shape[1] - 0.5), shape[1] + 1)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = np.linspace(np.full(x.shape, origin[2] - spacing[2] / 2.),
                    np.full(x.shape, origin[2] + spacing[2] * (shape[2] - 0.5)),
                    num=shape[2] + 1,
                    axis=-1)
    x = np.repeat(x[..., np.newaxis], shape[2] + 1, axis=-1)
    y = np.repeat(y[..., np.newaxis], shape[2] + 1, axis=-1)

    grid = pv.StructuredGrid(x, y, z)

    grid.cell_data['Base_Porosity'] = org_porosity

    mid = pv.create_grid(grid, (nx, ny, nz)).sample(grid)
    porosity = mid.point_data['Base_Porosity']
    porosity[porosity == 0] = org_porosity.min()
    permeability = np.power(10, perms(porosity))

    poro, perm = porosity, permeability

    proxy_model = Model(set_nx=nx, set_ny=ny, set_nz=nz, set_dx=dx, set_dy=dy, set_dz=dz, perms=perm, poro=poro,
                        overburden=overburden,n_points=n_points,total_time=total_time)
    proxy_model.init()

    return proxy_model


def homogeneous_model_simulation(nx, ny, nz, dx, dy, dz, n_points=32, overburden=0, total_time=30*365, rate=None):
    redirect_darts_output(f'log_ho_rate_{rate}.txt')
    perms = np.ones(nx * ny * nz) * 800
    poros = np.ones(nx * ny * nz) * 0.2

    proxy_model = Model(set_nx=nx, set_ny=ny, set_nz=nz, set_dx=dx, set_dy=dy, set_dz=dz, perms=perms, poro=poros,
                        overburden=overburden, n_points=n_points, total_time=total_time)
    proxy_model.init()
    
    return proxy_model