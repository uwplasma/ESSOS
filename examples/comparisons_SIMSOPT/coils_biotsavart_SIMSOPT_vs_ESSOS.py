import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import block_until_ready
from essos.fields import BiotSavart as BiotSavart_essos
from essos.coils import Coils_from_simsopt, Curves_from_simsopt
from simsopt import load
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import BiotSavart as BiotSavart_simsopt, coils_via_symmetries
from simsopt.configs import get_ncsx_data, get_w7x_data, get_hsx_data, get_giuliani_data

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

list_segments = [30, 100, 300, 1000, 3000]

LandremanPaulQA_json_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', 'SIMSOPT_biot_savart_LandremanPaulQA.json')
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
        coils_essos = Coils_from_simsopt(json_file_stel, nfp)
        curves_essos = Curves_from_simsopt(json_file_stel, nfp)
    else:
        coils_simsopt  = coils_via_symmetries(curves_stel, currents_stel, nfp, True)
        curves_simsopt = [c.curve for c in coils_simsopt]
        currents_simsopt = [c.current for c in coils_simsopt]
        field_simsopt = BiotSavart_simsopt(coils_simsopt)
        
        coils_essos = Coils_from_simsopt(coils_simsopt, nfp)
        curves_essos = Curves_from_simsopt(curves_simsopt, nfp)
        
    field_essos = BiotSavart_essos(coils_essos)
    
    coils_essos_to_simsopt = coils_essos.to_simsopt()
    curves_essos_to_simsopt = curves_essos.to_simsopt()
    field_essos_to_simsopt = BiotSavart_simsopt(coils_essos_to_simsopt)

    curves_to_vtk(curves_simsopt, os.path.join(output_dir,f"curves_simsopt_{name}"))
    curves_essos.to_vtk(os.path.join(output_dir,f"curves_essos_{name}"))
    curves_to_vtk(curves_essos_to_simsopt, os.path.join(output_dir,f"curves_essos_to_simsopt_{name}"))

    base_coils_simsopt = coils_simsopt[:int(len(coils_simsopt)/2/nfp)]
    R = jnp.mean(jnp.array([jnp.sqrt(coil.curve.x[coil.curve.local_dof_names.index('xc(0)')]**2
                +coil.curve.x[coil.curve.local_dof_names.index('yc(0)')]**2)
        for coil in base_coils_simsopt]))
    x = jnp.array([R+0.01,R,R])
    y = jnp.array([R,R+0.01,R-0.01])
    z = jnp.array([0.05,0.06,0.07])

    positions = jnp.array((x,y,z))

    len_list_segments = len(list_segments)
    t_gamma_avg_essos = jnp.zeros(len_list_segments)
    t_gamma_avg_simsopt = jnp.zeros(len_list_segments)
    gamma_error_avg = jnp.zeros(len_list_segments)
    t_gammadash_avg_essos = jnp.zeros(len_list_segments)
    t_gammadash_avg_simsopt = jnp.zeros(len_list_segments)
    gammadash_error_avg = jnp.zeros(len_list_segments)
    t_gammadashdash_avg_essos = jnp.zeros(len_list_segments)
    t_gammadashdash_avg_simsopt = jnp.zeros(len_list_segments)
    gammadashdash_error_avg = jnp.zeros(len_list_segments)
    t_curvature_avg_essos = jnp.zeros(len_list_segments)
    t_curvature_avg_simsopt = jnp.zeros(len_list_segments)
    curvature_error_avg = jnp.zeros(len_list_segments)
    t_B_avg_essos = jnp.zeros(len_list_segments)
    t_B_avg_simsopt = jnp.zeros(len_list_segments)
    B_error_avg = jnp.zeros(len_list_segments)
    t_dB_by_dX_avg_essos = jnp.zeros(len_list_segments)
    t_dB_by_dX_avg_simsopt = jnp.zeros(len_list_segments)
    dB_by_dX_error_avg = jnp.zeros(len_list_segments)
    
    gamma_error_simsopt_to_essos = 0
    gamma_error_essos_to_simsopt = 0
    
    for i, (coil_simsopt, coil_essos_gamma, coil_essos_to_simsopt) in enumerate(zip(coils_simsopt, coils_essos.gamma, coils_essos_to_simsopt)):
        gamma_error_simsopt_to_essos += jnp.linalg.norm(coil_simsopt.curve.gamma()-coil_essos_gamma)
        gamma_error_essos_to_simsopt += jnp.linalg.norm(coil_simsopt.curve.gamma()-coil_essos_to_simsopt.curve.gamma())

    B_error_avg_simsopt_to_essos = 0
    B_error_avg_essos_to_simsopt = 0
    for j, position in enumerate(positions):
        field_simsopt.set_points([position])
        field_essos_to_simsopt.set_points([position])
        B_simsopt = field_simsopt.B()
        B_essos_to_simsopt = field_essos_to_simsopt.B()
        B_simsopt_to_essos = field_essos.B(position)
        B_error_avg_simsopt_to_essos += jnp.abs(jnp.linalg.norm(B_simsopt) - jnp.linalg.norm(B_simsopt_to_essos))
        B_error_avg_essos_to_simsopt += jnp.abs(jnp.linalg.norm(B_simsopt) - jnp.linalg.norm(B_essos_to_simsopt))
    B_error_avg_simsopt_to_essos = B_error_avg_simsopt_to_essos/len(positions)
    B_error_avg_essos_to_simsopt = B_error_avg_essos_to_simsopt/len(positions)

    fig = plt.figure(figsize = (8, 6))
    X_axis = jnp.arange(2)
    plt.bar(X_axis[0] - 0.2, gamma_error_simsopt_to_essos+1e-19, 0.3, label='SIMSOPT to ESSOS coils', color='blue', edgecolor='black', hatch='/')
    plt.bar(X_axis[0] + 0.2, gamma_error_essos_to_simsopt+1e-19, 0.3, label='ESSOS to SIMSOPT coils', color='red', edgecolor='black', hatch='-')
    plt.bar(X_axis[1] - 0.2, B_error_avg_simsopt_to_essos+1e-19, 0.3, label=r'SIMSOPT to ESSOS $B$', color='blue', edgecolor='black', hatch='||')
    plt.bar(X_axis[1] + 0.2, B_error_avg_essos_to_simsopt+1e-19, 0.3, label=r'ESSOS to SIMSOPT $B$', color='red', edgecolor='black', hatch='*')
    plt.xticks(X_axis, ['Coil Error', 'B Error'])
    plt.xlabel('Parameter', fontsize=14)
    plt.ylabel('Error Magnitude', fontsize=14)
    plt.yscale('log')
    plt.ylim(1e-20, 1e-11)
    plt.legend(fontsize=14)
    plt.grid(axis='y')
    plt.title(f"{name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"error_gamma_B_SIMSOPT_vs_ESSOS_{name}.pdf"), transparent=True)
    plt.close()

    def update_nsegments_simsopt(curve_simsopt, n_segments):
        new_curve = CurveXYZFourier(n_segments, curve_simsopt.order)
        new_curve.x = curve_simsopt.x
        return new_curve
   
    for index, n_segments in enumerate(list_segments):
        coils_essos.n_segments = n_segments
        
        base_curves_simsopt = [update_nsegments_simsopt(coil_simsopt.curve, n_segments) for coil_simsopt in base_coils_simsopt]
        coils_simsopt = coils_via_symmetries(base_curves_simsopt, currents_simsopt[0:len(base_coils_simsopt)], nfp, True)
        curves_simsopt = [c.curve for c in coils_simsopt]
        
        [curve.gamma() for curve in curves_simsopt]
        coils_essos.gamma
        
        start_time = time()
        gamma_curves_simsopt = block_until_ready(jnp.array([curve.gamma() for curve in curves_simsopt]))
        t_gamma_avg_simsopt = t_gamma_avg_simsopt.at[index].set(t_gamma_avg_simsopt[index] + time() - start_time)
        
        start_time = time()
        gamma_curves_essos = block_until_ready(jnp.array(coils_essos.gamma))
        t_gamma_avg_essos = t_gamma_avg_essos.at[index].set(t_gamma_avg_essos[index] + time() - start_time)
        
        start_time = time()
        gammadash_curves_simsopt = block_until_ready(jnp.array([curve.gammadash() for curve in curves_simsopt]))
        t_gammadash_avg_simsopt = t_gammadash_avg_simsopt.at[index].set(t_gammadash_avg_simsopt[index] + time() - start_time)
        
        start_time = time()
        gammadash_curves_essos = block_until_ready(jnp.array(coils_essos.gamma_dash))
        t_gammadash_avg_essos = t_gammadash_avg_essos.at[index].set(t_gammadash_avg_essos[index] + time() - start_time)

        start_time = time()
        gammadashdash_curves_simsopt = block_until_ready(jnp.array([curve.gammadashdash() for curve in curves_simsopt]))
        t_gammadashdash_avg_simsopt = t_gammadashdash_avg_simsopt.at[index].set(t_gammadashdash_avg_simsopt[index] + time() - start_time)

        start_time = time()
        gammadashdash_curves_essos = block_until_ready(jnp.array(coils_essos.gamma_dashdash))
        t_gammadashdash_avg_essos = t_gammadashdash_avg_essos.at[index].set(t_gammadashdash_avg_essos[index] + time() - start_time)

        start_time = time()
        curvature_curves_simsopt = block_until_ready(jnp.array([curve.kappa() for curve in curves_simsopt]))
        t_curvature_avg_simsopt = t_curvature_avg_simsopt.at[index].set(t_curvature_avg_simsopt[index] + time() - start_time)

        start_time = time()
        curvature_curves_essos = block_until_ready(jnp.array(coils_essos.curvature))
        t_curvature_avg_essos = t_curvature_avg_essos.at[index].set(t_curvature_avg_essos[index] + time() - start_time)

        gamma_error_avg         = gamma_error_avg.        at[index].set(gamma_error_avg[index]         + jnp.linalg.norm(gamma_curves_essos         - gamma_curves_simsopt))
        gammadash_error_avg     = gammadash_error_avg.    at[index].set(gammadash_error_avg[index]     + jnp.linalg.norm(gammadash_curves_essos     - gammadash_curves_simsopt))
        gammadashdash_error_avg = gammadashdash_error_avg.at[index].set(gammadashdash_error_avg[index] + jnp.linalg.norm(gammadashdash_curves_essos - gammadashdash_curves_simsopt))
        curvature_error_avg = curvature_error_avg.at[index].set(curvature_error_avg[index] + jnp.linalg.norm(curvature_curves_essos - curvature_curves_simsopt))

        field_essos = BiotSavart_essos(coils_essos)
        field_simsopt = BiotSavart_simsopt(coils_simsopt)
        
        for j, position in enumerate(positions):
            field_essos.B(position)
            time1 = time()
            result_B_essos = field_essos.B(position)
            t_B_avg_essos = t_B_avg_essos.at[index].set(t_B_avg_essos[index] + time() - time1)
            normB_essos = jnp.linalg.norm(result_B_essos)

            field_simsopt.set_points(jnp.array([position]))
            field_simsopt.B()
            time3 = time()
            field_simsopt.set_points(jnp.array([position]))
            result_simsopt = field_simsopt.B()
            t_B_avg_simsopt = t_B_avg_simsopt.at[index].set(t_B_avg_simsopt[index] + time() - time3)
            normB_simsopt = jnp.linalg.norm(jnp.array(result_simsopt))

            B_error_avg = B_error_avg.at[index].set(B_error_avg[index] + jnp.abs(normB_essos - normB_simsopt))
            
            field_essos.dB_by_dX(position)
            time1 = time()
            field_simsopt.set_points(jnp.array([position]))
            result_dB_by_dX_essos = field_essos.dB_by_dX(position)
            t_dB_by_dX_avg_essos = t_dB_by_dX_avg_essos.at[index].set(t_dB_by_dX_avg_essos[index] + time() - time1)
            norm_dB_by_dX_essos = jnp.linalg.norm(result_dB_by_dX_essos)
            
            field_simsopt.dB_by_dX()
            time3 = time()
            field_simsopt.set_points(jnp.array([position]))
            result_dB_by_dX_simsopt = field_simsopt.dB_by_dX()
            t_dB_by_dX_avg_simsopt = t_dB_by_dX_avg_simsopt.at[index].set(t_dB_by_dX_avg_simsopt[index] + time() - time3)
            norm_dB_by_dX_simsopt = jnp.linalg.norm(jnp.array(result_dB_by_dX_simsopt))
            
            dB_by_dX_error_avg = dB_by_dX_error_avg.at[index].set(dB_by_dX_error_avg[index] + jnp.abs(norm_dB_by_dX_essos - norm_dB_by_dX_simsopt))

    X_axis = jnp.arange(len_list_segments)

    fig = plt.figure(figsize = (8, 6))
    plt.bar(X_axis-0.2, B_error_avg, 0.1, label = r"$B_{\text{essos}} - B_{\text{simsopt}}$", color="green", edgecolor="black", hatch="/")
    plt.bar(X_axis-0.1, dB_by_dX_error_avg, 0.1, label = r"${B'}_{\text{essos}} - {B'}_{\text{simsopt}}$", color="purple", edgecolor="black", hatch="x")
    plt.bar(X_axis+0.0, gamma_error_avg, 0.1, label = r"$\Gamma_{\text{essos}} - \Gamma_{\text{simsopt}}$", color="orange", edgecolor="black", hatch="|")
    plt.bar(X_axis+0.1, gammadash_error_avg, 0.1, label = r"${\Gamma'}_{\text{essos}} - {\Gamma'}_{\text{simsopt}}$", color="gray", edgecolor="black", hatch="-")
    plt.bar(X_axis+0.2, gammadashdash_error_avg, 0.1, label = r"${\Gamma''}_{\text{essos}} - {\Gamma''}_{\text{simsopt}}$", color="black", edgecolor="black", hatch="*")
    plt.bar(X_axis+0.3, curvature_error_avg, 0.1, label = r"$\kappa_{\text{essos}} - \kappa_{\text{simsopt}}$", color="brown", edgecolor="black", hatch="\\")
    plt.xticks(X_axis, list_segments) 
    plt.xlabel("Number of segments of each coil", fontsize=14)
    plt.ylabel(f"Difference SIMSOPT vs ESSOS", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(fontsize=14)
    plt.yscale("log")
    plt.grid(axis='y')
    plt.ylim(1e-18, 1e-10)
    plt.title(f"{name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"error_BiotSavart_SIMSOPT_vs_ESSOS_{name}.pdf"), transparent=True)
    plt.close()

    fig = plt.figure(figsize = (8, 6))
    plt.bar(X_axis - 0.30, t_B_avg_essos, 0.05, label = r'B ESSOS', color="red", edgecolor="black")
    plt.bar(X_axis - 0.25, t_B_avg_simsopt, 0.05, label = r'B SIMSOPT', color="blue", edgecolor="black")
    plt.bar(X_axis - 0.20, t_dB_by_dX_avg_essos, 0.05, label = r"$B'$ ESSOS", color="red", edgecolor="black")
    plt.bar(X_axis - 0.15, t_dB_by_dX_avg_simsopt, 00.05, label = r"$B'$ SIMSOPT", color="blue", edgecolor="black")
    plt.bar(X_axis - 0.10, t_gamma_avg_essos, 0.05, label = r'$\Gamma$ ESSOS', color="red", edgecolor="black", hatch="//")
    plt.bar(X_axis - 0.05, t_gamma_avg_simsopt, 0.05, label = r'$\Gamma$ SIMSOPT', color="blue", edgecolor="black", hatch="-")
    plt.bar(X_axis + 0.0, t_gammadash_avg_essos, 0.05, label = r"${\Gamma'}$ ESSOS", color="red", edgecolor="black", hatch="\\")
    plt.bar(X_axis + 0.05, t_gammadash_avg_simsopt, 0.05, label = r"${\Gamma'}$ SIMSOPT", color="blue", edgecolor="black", hatch="||")
    plt.bar(X_axis + 0.10, t_gammadashdash_avg_essos, 0.05, label = r"${\Gamma''}$ ESSOS", color="red", edgecolor="black", hatch="*")
    plt.bar(X_axis + 0.15, t_gammadashdash_avg_simsopt, 0.05, label = r"${\Gamma''}$ SIMSOPT", color="blue", edgecolor="black", hatch="|")
    plt.bar(X_axis + 0.20, t_curvature_avg_essos, 0.05, label = r"$\kappa$ ESSOS", color="red", edgecolor="black", hatch="x")
    plt.bar(X_axis + 0.25, t_curvature_avg_simsopt, 0.05, label = r"$\kappa$ SIMSOPT", color="blue", edgecolor="black", hatch="+")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks(X_axis, list_segments) 
    plt.xlabel("Number of segments of each coil", fontsize=14)
    plt.ylabel("Time to evaluate SIMSOPT vs ESSOS (s)", fontsize=14)
    plt.grid(axis='y')
    # plt.gca().set_ylim((None,0.03))
    plt.yscale("log")
    plt.legend(fontsize=14)
    plt.title(f"{name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"time_BiotSavart_SIMSOPT_vs_ESSOS_{name}.pdf"), transparent=True)
    plt.close()
