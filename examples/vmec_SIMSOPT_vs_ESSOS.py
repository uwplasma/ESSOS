from time import time
import jax.numpy as jnp
from jax import block_until_ready, random
from essos.fields import Vmec as Vmec_essos
from simsopt.mhd import Vmec as Vmec_simsopt, vmec_compute_geometry

wout = "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"

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
print(f"Average time for modB simsopt: {average_time_modB_simsopt:.2e}s")
print(f"Average time for modB essos: {average_time_modB_essos:.2e}s")
print(f"Average time for B simsopt: {average_time_B_simsopt:.2e}s")
print(f"Average time for B essos: {average_time_B_essos:.2e}s")
print(f"Average relative error in modB: {(error_modB*100):.2e}%")
print(f"Maximum relative error in modB: {(jnp.max(error_modB)*100):.2e}%")
print(f"Average relative error in B: {error_B*100}%")
print(f"Maximum relative error in B: {(jnp.max(error_B)*100):.2e}%")