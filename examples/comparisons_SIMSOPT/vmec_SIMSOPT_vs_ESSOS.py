import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import block_until_ready, random
from essos.fields import Vmec as Vmec_essos
from simsopt.mhd import Vmec as Vmec_simsopt, vmec_compute_geometry

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wout_array = [os.path.join(os.path.dirname(__file__), '..', 'input_files', "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"),
              os.path.join(os.path.dirname(__file__), '..', 'input_files', "wout_n3are_R7.75B5.7.nc")]
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
    
    fig = plt.figure(figsize = (8, 6))
    X_axis = jnp.arange(4)
    Y_axis = [average_time_modB_simsopt, average_time_B_simsopt, average_time_modB_essos, average_time_B_essos]
    colors = ['blue', 'blue', 'red', 'red']
    hatches = ['/', '\\', '/', '\\']
    bars = plt.bar(X_axis, Y_axis, width=0.4, color=colors)
    for bar, hatch in zip(bars, hatches): bar.set_hatch(hatch)
    plt.xticks(X_axis, [r"$|\boldsymbol{B}|$ SIMSOPT", r"$\boldsymbol{B}$ SIMSOPT", r"$|\boldsymbol{B}|$ ESSOS", r"$\boldsymbol{B}$ ESSOS"], fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.ylabel("Time to evaluate VMEC field (s)", fontsize=14)
    plt.grid(axis='y')
    plt.yscale("log")
    plt.ylim(1e-6, 1)
    plt.title(name, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"time_VMEC_SIMSOPT_vs_ESSOS_{name}.pdf"), transparent=True)
    plt.close()

    fig = plt.figure(figsize = (8, 6))
    X_axis = jnp.arange(2)
    Y_axis = [jnp.mean(error_modB), jnp.mean(error_B)]
    colors = ['purple', 'orange']
    hatches = ['/', '//']
    bars = plt.bar(X_axis, Y_axis, width=0.4, color=colors)
    for bar, hatch in zip(bars, hatches): bar.set_hatch(hatch)
    plt.xticks(X_axis, [r"$|\boldsymbol{B}|$", r"$\boldsymbol{B}$"], fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.ylabel("Relative error SIMSOPT vs ESSOS (%)", fontsize=14)
    plt.grid(axis='y')
    plt.yscale("log")
    plt.ylim(1e-6, 1e-1)
    plt.title(name, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"error_VMEC_SIMSOPT_vs_ESSOS_{name}.pdf"), transparent=True)
    plt.close()