from generate_models import stratified_model_simulation, \
    homogeneous_model_simulation, heterogeneous_model_simulation
import numpy as np


def execute(proxy_model, file_name, output_dir):
    # now we start to run for the time report--------------------------------------------------------------
    time_step = proxy_model.report_time
    even_end = int(proxy_model.total_time / time_step) * time_step
    time_step_arr = np.ones(int(proxy_model.total_time / time_step)) * time_step
    if proxy_model.runtime - even_end > 0:
        time_step_arr = np.append(time_step_arr, proxy_model.total_time - even_end)
    time_index = 0
    properties = proxy_model.output_properties()
    pressure = properties[0, :]
    temperature = properties[2, :]
    proxy_model.xrdata.set_dynamic_grid_xarray(
        time_index,
        "Pressure",
        np.rot90(
            pressure.reshape(
                (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
            )
        ),
    )
    proxy_model.xrdata.set_dynamic_grid_xarray(
        time_index,
        "Temperature",
        np.rot90(
            temperature.reshape(
                (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
            )
        ),
    )
    for ts in time_step_arr:
        for _, w in enumerate(proxy_model.reservoir.wells):
            if w.name.lower().startswith("i"):
                w.control = proxy_model.physics.new_mass_rate_water_inj(417000, 2358)
                w.constraint = proxy_model.physics.new_bhp_water_inj(1858 * 0.15, 300)
            else:
                w.control = proxy_model.physics.new_mass_rate_water_prod(417000)


        proxy_model.run(ts)

        time_index += 1
        properties = proxy_model.output_properties()
        pressure = properties[0, :]
        temperature = properties[2, :]
        proxy_model.xrdata.set_dynamic_grid_xarray(
            time_index,
            "Pressure",
            np.rot90(
                pressure.reshape(
                    (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
                )
            ),
        )
        proxy_model.xrdata.set_dynamic_grid_xarray(
            time_index,
            "Temperature",
            np.rot90(
                temperature.reshape(
                    (proxy_model.nx, proxy_model.ny, proxy_model.nz), order="F"
                )
            ),
        )
        proxy_model.physics.engine.report()
    proxy_model.print_timers()
    proxy_model.print_stat()

    time_data_report = pd.DataFrame.from_dict(
        proxy_model.physics.engine.time_data_report
    )
    proxy_model.xrdata.set_dynamic_xarray(proxy_model.reservoir.wells, time_data_report)

    proxy_model.xrdata.write_xarray(file_name=file_name, output_dir=output_dir)

    return time_data_report

def generate_base():
    geothermal_model = homogeneous_model_simulation(nx, ny, nz, dx, dy, dz, overburden=4)
    # geothermal_model = stratified_model_simulation(nx, ny, nz, dx, dy, dz, overburden=4)
    # geothermal_model = heterogeneous_model_simulation(
    #     nx, ny, nz, dx, dy, dz, overburden=4
    # )
    _ = execute(geothermal_model, file_name="base_100_steps_he", output_dir="base")



if __name__ == '__main__':
    x_spacing = 4500
    y_spacing = 4200
    z_spacing = 100
    dx = dy = 18
    dz = 5
    nx = int(x_spacing / dx)
    ny = int(y_spacing / dy)
    nz = int(z_spacing / dz)
    generate_base()
