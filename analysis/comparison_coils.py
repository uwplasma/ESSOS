import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from jax import block_until_ready
from essos.fields import BiotSavart as BiotSavart_essos
from essos.coils import Coils, Curves
from simsopt import load
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import BiotSavart as BiotSavart_simsopt, coils_via_symmetries
from simsopt.configs import get_ncsx_data, get_w7x_data, get_hsx_data, get_giuliani_data

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_segments = 100

LandremanPaulQA_json_file = os.path.join(os.path.dirname(__file__), '../examples/', 'input_files', 'SIMSOPT_biot_savart_LandremanPaulQA.json')
nfp_array      = [3, 2, 5, 4, 2]
curves_array   = [get_ncsx_data()[0], LandremanPaulQA_json_file, get_w7x_data()[0], get_hsx_data()[0], get_giuliani_data()[0]]
currents_array = [get_ncsx_data()[1], None, get_w7x_data()[1], get_hsx_data()[1], get_giuliani_data()[1]]
name_array     = ["NCSX", "QA(json)", "W7-X", "HSX", "Giuliani"]

print(f'Output being saved to {output_dir}')
print(f'SIMSOPT LandremanPaulQA json file location: {LandremanPaulQA_json_file}')
for nfp, curves_stel, currents_stel, name in zip(nfp_array, curves_array, currents_array, name_array):
    print(f' Running {name} and saving to output directory...')
    if currents_stel is None:
        json_file_stel = curves_stel
        field_simsopt = load(json_file_stel)
        coils_simsopt = field_simsopt.coils
        curves_simsopt = [coil.curve for coil in coils_simsopt]
        currents_simsopt = [coil.current for coil in coils_simsopt]
        coils_essos = Coils.from_simsopt(json_file_stel, nfp)
        curves_essos = Curves.from_simsopt(json_file_stel, nfp)
    else:
        coils_simsopt  = coils_via_symmetries(curves_stel, currents_stel, nfp, True)
        curves_simsopt = [c.curve for c in coils_simsopt]
        currents_simsopt = [c.current for c in coils_simsopt]
        field_simsopt = BiotSavart_simsopt(coils_simsopt)

        coils_essos = Coils.from_simsopt(coils_simsopt, nfp)
        curves_essos = Curves.from_simsopt(curves_simsopt, nfp)

    field_essos = BiotSavart_essos(coils_essos)
    
    coils_essos_to_simsopt = coils_essos.to_simsopt()
    curves_essos_to_simsopt = curves_essos.to_simsopt()
    field_essos_to_simsopt = BiotSavart_simsopt(coils_essos_to_simsopt)

    # curves_to_vtk(curves_simsopt, os.path.join(output_dir,f"curves_simsopt_{name}"))
    # curves_essos.to_vtk(os.path.join(output_dir,f"curves_essos_{name}"))
    # curves_to_vtk(curves_essos_to_simsopt, os.path.join(output_dir,f"curves_essos_to_simsopt_{name}"))

    base_coils_simsopt = coils_simsopt[:int(len(coils_simsopt)/2/nfp)]
    R = jnp.mean(jnp.array([jnp.sqrt(coil.curve.x[coil.curve.local_dof_names.index('xc(0)')]**2
                +coil.curve.x[coil.curve.local_dof_names.index('yc(0)')]**2)
        for coil in base_coils_simsopt]))
    x = jnp.array([R+0.01,R,R])
    y = jnp.array([R,R+0.01,R-0.01])
    z = jnp.array([0.05,0.06,0.07])

    positions = jnp.array((x,y,z))

    def update_nsegments_simsopt(curve_simsopt, n_segments):
        new_curve = CurveXYZFourier(n_segments, curve_simsopt.order)
        new_curve.x = curve_simsopt.x
        return new_curve
   
    coils_essos.n_segments = n_segments
    
    base_curves_simsopt = [update_nsegments_simsopt(coil_simsopt.curve, n_segments) for coil_simsopt in base_coils_simsopt]
    coils_simsopt = coils_via_symmetries(base_curves_simsopt, currents_simsopt[0:len(base_coils_simsopt)], nfp, True)
    curves_simsopt = [c.curve for c in coils_simsopt]
    
    # Running the first time for compilation
    [curve.gamma() for curve in curves_simsopt]
    [curve.gammadash() for curve in curves_simsopt]
    [curve.gammadashdash() for curve in curves_simsopt]
    coils_essos.gamma
    
    # Running the second time for coils characteristics comparison
    start_time = time()
    gamma_curves_simsopt = block_until_ready(jnp.array([curve.gamma() for curve in curves_simsopt]))
    t_gamma_avg_simsopt = time() - start_time
    
    start_time = time()
    gamma_curves_essos = block_until_ready(jnp.array(coils_essos.gamma))
    t_gamma_avg_essos = time() - start_time
    
    start_time = time()
    gammadash_curves_simsopt = block_until_ready(jnp.array([curve.gammadash() for curve in curves_simsopt]))
    t_gammadash_avg_simsopt = time() - start_time
    
    start_time = time()
    gammadash_curves_essos = block_until_ready(jnp.array(coils_essos.gamma_dash))
    t_gammadash_avg_essos = time() - start_time

    start_time = time()
    gammadashdash_curves_simsopt = block_until_ready(jnp.array([curve.gammadashdash() for curve in curves_simsopt]))
    t_gammadashdash_avg_simsopt = time() - start_time

    start_time = time()
    gammadashdash_curves_essos = block_until_ready(jnp.array(coils_essos.gamma_dashdash))
    t_gammadashdash_avg_essos = time() - start_time

    start_time = time()
    curvature_curves_simsopt = block_until_ready(jnp.array([curve.kappa() for curve in curves_simsopt]))
    t_curvature_avg_simsopt = time() - start_time

    start_time = time()
    curvature_curves_essos = block_until_ready(jnp.array(coils_essos.curvature))
    t_curvature_avg_essos = time() - start_time

    gamma_error_avg         = jnp.linalg.norm(gamma_curves_essos         - gamma_curves_simsopt)
    gammadash_error_avg     = jnp.linalg.norm(gammadash_curves_essos     - gammadash_curves_simsopt)
    gammadashdash_error_avg = jnp.linalg.norm(gammadashdash_curves_essos - gammadashdash_curves_simsopt)
    curvature_error_avg     = jnp.linalg.norm(curvature_curves_essos     - curvature_curves_simsopt)

    # Magnetic field comparison

    field_essos = BiotSavart_essos(coils_essos)
    field_simsopt = BiotSavart_simsopt(coils_simsopt)

    t_B_avg_essos = 0
    t_B_avg_simsopt = 0
    B_error_avg = 0
    t_dB_by_dX_avg_essos = 0
    t_dB_by_dX_avg_simsopt = 0
    dB_by_dX_error_avg = 0

    for position in positions:
        field_essos.B(position)
        time1 = time()
        result_B_essos = field_essos.B(position)
        t_B_avg_essos = t_B_avg_essos + time() - time1
        normB_essos = jnp.linalg.norm(result_B_essos)

        field_simsopt.set_points(jnp.array([position]))
        field_simsopt.B()
        time3 = time()
        field_simsopt.set_points(jnp.array([position]))
        result_simsopt = field_simsopt.B()
        t_B_avg_simsopt = t_B_avg_simsopt + time() - time3
        normB_simsopt = jnp.linalg.norm(jnp.array(result_simsopt))

        B_error_avg = B_error_avg + jnp.abs(normB_essos - normB_simsopt)
        
        field_essos.dB_by_dX(position)
        time1 = time()
        field_simsopt.set_points(jnp.array([position]))
        result_dB_by_dX_essos = field_essos.dB_by_dX(position)
        t_dB_by_dX_avg_essos = t_dB_by_dX_avg_essos + time() - time1
        norm_dB_by_dX_essos = jnp.linalg.norm(result_dB_by_dX_essos)
        
        field_simsopt.dB_by_dX()
        time3 = time()
        field_simsopt.set_points(jnp.array([position]))
        result_dB_by_dX_simsopt = field_simsopt.dB_by_dX()
        t_dB_by_dX_avg_simsopt = t_dB_by_dX_avg_simsopt + time() - time3
        norm_dB_by_dX_simsopt = jnp.linalg.norm(jnp.array(result_dB_by_dX_simsopt))
        
        dB_by_dX_error_avg = dB_by_dX_error_avg + jnp.abs(norm_dB_by_dX_essos - norm_dB_by_dX_simsopt)

    # Labels and corresponding absolute errors (ESSOS - SIMSOPT)
    quantities_errors = [
        (r"$B$",        jnp.abs(B_error_avg)),
        (r"$B'$",       jnp.abs(dB_by_dX_error_avg)),
        (r"$\Gamma$",   jnp.abs(gamma_error_avg)),
        (r"$\Gamma'$",  jnp.abs(gammadash_error_avg)),
        (r"$\Gamma''$", jnp.abs(gammadashdash_error_avg)),
        (r"$\kappa$",   jnp.abs(curvature_error_avg)),
    ]

    labels = [q[0] for q in quantities_errors]
    error_vals = [q[1] for q in quantities_errors]

    X_axis = jnp.arange(len(labels))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(X_axis, error_vals, bar_width, color="darkorange", edgecolor="black")

    ax.set_xticks(X_axis)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute error")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-12)
    ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_error_BiotSavart_{name}.pdf"), transparent=True)
    plt.close()


    # Labels and corresponding timings
    quantities = [
        (r"$B$",        t_B_avg_essos,             t_B_avg_simsopt),
        (r"$B'$",       t_dB_by_dX_avg_essos,      t_dB_by_dX_avg_simsopt),
        (r"$\Gamma$",   t_gamma_avg_essos,         t_gamma_avg_simsopt),
        (r"$\Gamma'$",  t_gammadash_avg_essos,     t_gammadash_avg_simsopt),
        (r"$\Gamma''$", t_gammadashdash_avg_essos, t_gammadashdash_avg_simsopt),
        (r"$\kappa$",   t_curvature_avg_essos,     t_curvature_avg_simsopt),
    ]

    labels = [q[0] for q in quantities]
    essos_vals = [q[1] for q in quantities]
    simsopt_vals = [q[2] for q in quantities]

    X_axis = jnp.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(X_axis - bar_width/2, essos_vals, bar_width, label="ESSOS", color="red", edgecolor="black")
    ax.bar(X_axis + bar_width/2, simsopt_vals, bar_width, label="SIMSOPT", color="blue", edgecolor="black")

    ax.set_xticks(X_axis)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Computation time (s)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-1)
    ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_time_BiotSavart_{name}.pdf"), transparent=True)
    plt.close()
