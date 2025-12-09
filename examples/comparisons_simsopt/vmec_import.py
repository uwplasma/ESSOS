import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from jax import block_until_ready, random
from essos.fields import Vmec as Vmec_essos
from simsopt.mhd import Vmec as Vmec_simsopt, vmec_compute_geometry


output_dir = os.path.join(os.path.dirname(__file__), '../output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wout_array = [os.path.join(os.path.dirname(__file__), '../../examples/', 'input_files', "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
              os.path.join(os.path.dirname(__file__), '../../examples/', 'input_files', "wout_n3are_R7.75B5.7.nc")]
name_array = ["LandremanPaulQA", 'NCSX']


print(f'Output being saved to {output_dir}')
for name, wout in zip(name_array, wout_array):
    print(f' Running comparison with VMEC file located at: {wout}')

    vmec_essos = Vmec_essos(wout)
    vmec_simsopt = Vmec_simsopt(wout)

    s_array=jnp.linspace(0.2, 0.9, 10)
    key = random.key(42)

    def absB_simsopt_func(s, theta, phi):
        return vmec_compute_geometry(vmec_simsopt, s, theta, phi).modB[0][0][0]
    def absB_essos_func(s, theta, phi):
        return vmec_essos.AbsB([s, theta, phi])
    def B_simsopt_func(s, theta, phi):
        g = vmec_compute_geometry(vmec_simsopt, s, theta, phi)
        return jnp.array([g.B_sub_s * g.grad_s_X + g.B_sub_theta_vmec * g.grad_theta_vmec_X + g.B_sub_phi * g.grad_phi_X,
                        g.B_sub_s * g.grad_s_Y + g.B_sub_theta_vmec * g.grad_theta_vmec_Y + g.B_sub_phi * g.grad_phi_Y,
                        g.B_sub_s * g.grad_s_Z + g.B_sub_theta_vmec * g.grad_theta_vmec_Z + g.B_sub_phi * g.grad_phi_Z])[:,0,0,0]
    def B_essos_func(s, theta, phi):
        return vmec_essos.B([s, theta, phi])

    def timed_B(s, function):
        theta = random.uniform(key=key, minval=0, maxval=2 * jnp.pi)
        phi = random.uniform(key=key, minval=0, maxval=2 * jnp.pi)
        function(s, theta, phi)
        time1 = time()
        B = block_until_ready(function(s, theta, phi))
        time_taken = time()-time1
        return time_taken, B

    average_time_modB_simsopt = 0
    average_time_modB_essos = 0
    average_time_B_essos = 0
    average_time_B_simsopt = 0
    error_modB = 0
    error_B = 0
    for s in s_array:
        time_modB_simsopt, modB_simsopt = timed_B(s, absB_simsopt_func)
        average_time_modB_simsopt += time_modB_simsopt
        
        time_modB_essos, modB_essos = timed_B(s, absB_essos_func)
        average_time_modB_essos += time_modB_essos
        
        time_B_essos, B_essos = timed_B(s, B_essos_func)
        average_time_B_essos += time_B_essos
        
        time_B_simsopt, B_simsopt = timed_B(s, B_simsopt_func)
        average_time_B_simsopt += time_B_simsopt
        
        error_modB += jnp.abs((modB_simsopt-modB_essos)/modB_simsopt)
        error_B += jnp.abs((B_simsopt-B_essos)/B_simsopt)

    average_time_modB_simsopt /= len(s_array)
    average_time_modB_essos /= len(s_array)
    average_time_B_essos /= len(s_array)
    average_time_B_simsopt /= len(s_array)
    error_modB /= len(s_array)
    error_B /= len(s_array)

    # Labels and corresponding absolute errors (ESSOS - SIMSOPT)
    quantities_errors = [
        (r"$B$",          jnp.mean(error_modB)),
        (r"$\mathbf{B}$", jnp.mean(error_B)),
    ]

    labels = [q[0] for q in quantities_errors]
    error_vals = [q[1] for q in quantities_errors]

    X_axis = jnp.arange(len(labels))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(X_axis, error_vals, bar_width, color="darkorange", edgecolor="black")

    ax.set_xticks(X_axis)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative error")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-2)
    ax.grid(axis='y', which='both', linestyle='--', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparisons_VMEC_error_{name}.pdf"), transparent=True)

    # Labels and corresponding timings
    print(f"Average time to compute |B| in SIMSOPT: {average_time_modB_simsopt:.6f} s")
    print(f"Average time to compute B in SIMSOPT: {average_time_B_simsopt:.6f} s")
    print(f"Average time to compute |B| in ESSOS: {average_time_modB_essos:.6f} s")
    print(f"Average time to compute B in ESSOS: {average_time_B_essos:.6f} s")
    print(f"Relative error in |B|: {jnp.mean(error_modB):.6f}")
    print(f"Relative error in B: {jnp.mean(error_B):.6f}")

    quantities = [
        (r"$B$",          average_time_modB_essos, average_time_modB_simsopt),
        (r"$\mathbf{B}$", average_time_B_essos, average_time_B_simsopt),
    ]

    labels = [q[0] for q in quantities]
    essos_vals = [q[1] for q in quantities]
    simsopt_vals = [q[2] for q in quantities]

    X_axis = jnp.arange(len(labels))
    bar_width = 0.4

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
    plt.savefig(os.path.join(output_dir, f"comparisons_VMEC_time_{name}.pdf"), transparent=True)

    plt.show()