import os
os.mkdir("images") if not os.path.exists("images") else None
os.mkdir("images/optimization") if not os.path.exists("images/optimization") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'

import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))

import jax
import jax.numpy as jnp

import numpy as np

from simsopt.field import BiotSavart, Current, Coil
from simsopt.geo import CurveXYZFourier

from time import time

import matplotlib.pyplot as plt
import matplotlib



from ESSOS import Curves, Coils, CreateEquallySpacedCurves, Particles
from Dynamics import GuidingCenter
from MagneticField import B, norm_B, grad_B, grad_B_vector


matplotlib.rcParams.update({'font.size': 16})

def simsopt_create_coil(dofs: jnp.ndarray, n_segments: int, order: float) -> CurveXYZFourier:
    curve = CurveXYZFourier(n_segments, order)
    curve.x = dofs
    curve.x = dofs
    return curve

debugging = False
#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

n_curves = 1
order = 1
coil_current=7e6

R = 3
A = 4 # Aspect ratio
r = R/A
r_init = r/4

n_particles = 10000
particles = Particles(n_particles)

stel_curves = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=1, stellsym=False)
stel = Coils(stel_curves, jnp.array([coil_current]*n_curves))

# stel.plot(show=True)

dofs = jnp.reshape(stel.dofs, (n_curves, -1))
# print(dofs)    
# print(jnp.shape(dofs))

currents = jnp.array([coil_current]*n_curves)

#------------------------------------------------------------------------#
# Setting the initial conditions
#------------------------------------------------------------------------#


# initial_values = stel.initial_conditions(particles, R, r_init, model="Guiding Center")
x = -3*jnp.ones(n_particles)
y = jnp.linspace(0, 2*jnp.pi, n_particles)
z = jnp.zeros(n_particles)
initial_values = jnp.array((x,y,z))
real_norm_B = 2*jnp.pi*1e-7*coil_current*r**2/(r**2 + y**2)**(3/2)

#------------------------------------------------------------------------#
# Creating the coils
#------------------------------------------------------------------------#

list_segments = [10, 100, 1000]#, 10000] 
len_list_segments = len(list_segments)
t_avg = np.zeros(len_list_segments)
t_avg_simsopt = np.zeros(len_list_segments)
B_error_avg = np.zeros(len_list_segments)
B_error_avg_simsopt = np.zeros(len_list_segments)

for index, n_segments in enumerate(list_segments):
    stel.n_segments = n_segments

    curves = np.empty(n_curves, dtype=object)
    curves_points = jnp.empty((n_curves, n_segments, 3))
    dash_points = jnp.empty((n_curves, n_segments, 3))
    coils = np.empty(n_curves, dtype=object)

    for i in range(n_curves):
        # Creating a curve with "n_segments" segments and "order" order of the Fourier series
        curves[i] = simsopt_create_coil(dofs[i], n_segments, order)

        # Creating a coil
        coils[i] = Coil(curves[i], Current(currents[i]))
        
        # Getting the curve points  
        curves_points_simsopt = curves_points.at[i].set(curves[i].gamma())
        dash_points_simsopt = curves_points.at[i].set(curves[i].gammadash())
        curves_points = stel.gamma
        dash_points = stel.gamma_dash

        print("All close gamma", jnp.allclose(curves_points,curves_points_simsopt))
        print("All close gamma dash", jnp.allclose(dash_points, dash_points_simsopt))

        #------------------------------------------------------------------------#
        # Magnetic Field Calcultions
        #------------------------------------------------------------------------#

    for particle in range(particles.number):
        time1 = time()
        gammadash = stel.gamma_dash
        result_B = grad_B_vector(jnp.transpose(initial_values[:3])[particle], curves_points, dash_points, currents)
        time2 = time()
        t_avg[index]+=(time2 - time1)
        normB = jnp.linalg.norm(result_B)

        field = BiotSavart(coils)
        
        field.set_points([jnp.transpose(initial_values[:3])[particle]])
        time3 = time()
        result_simsopt = field.dB_by_dX()
        time4 = time()

        t_avg_simsopt[index]+=(time4 - time3)

        normB_simsopt = np.linalg.norm(np.array(result_simsopt))

        B_error_avg[index] += np.abs(normB - real_norm_B[particle])/real_norm_B[particle]
        B_error_avg_simsopt[index] += np.abs(normB_simsopt - real_norm_B[particle])/real_norm_B[particle]

    B_error_avg[index] /= particles.number
    t_avg[index] /= particles.number
    t_avg_simsopt[index]/=particles.number

    print(f"Error in B: {B_error_avg[index]}")
    print(f"Time: {t_avg[index]}")
    print(f"Error in B simsopt: {B_error_avg_simsopt[index]}")
    print(f"Time simsopt: {t_avg_simsopt[index]}")
    print("------------------------------------------------------------------------")

# creating the dataset

fig = plt.figure(figsize = (10, 5))

X_axis = np.arange(len_list_segments) 

# creating the bar plot
plt.bar(X_axis - 0.2, B_error_avg, 0.4, label = 'ESSOS') 
plt.bar(X_axis + 0.2, B_error_avg_simsopt, 0.4, label = 'simsopt') 
  
plt.xticks(X_axis, list_segments) 
plt.xlabel("Number of segments of each coil")
plt.ylabel(r"error in |$\mathbf{B}$|")
plt.yscale("log")
plt.grid(axis='y')
plt.legend()

fig.tight_layout()
plt.savefig("images/error_B.pdf", transparent=True)

plt.figure()
  
plt.bar(X_axis - 0.2, t_avg, 0.4, label = 'ESSOS') 
plt.bar(X_axis + 0.2, t_avg_simsopt, 0.4, label = 'simsopt') 
  
plt.xticks(X_axis, list_segments) 
plt.xlabel("Number of segments of each coil") 
plt.ylabel(r"Time to evaluate $\mathbf{B}$ [s]") 
plt.yscale("log")
plt.grid(axis='y')
plt.legend()

fig.tight_layout()
plt.savefig("images/time_comparison.pdf", transparent=True)

plt.show()