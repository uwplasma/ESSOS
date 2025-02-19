import jax.numpy as jnp
from coils import CreateEquallySpacedCurves, Coils
from fields import BiotSavart as BiotSavart_essos
import matplotlib.pyplot as plt
from simsopt.field import BiotSavart as BiotSavart_simsopt, Current, Coil
from simsopt.geo import CurveXYZFourier
from time import time

def simsopt_create_coil(dofs: jnp.ndarray, n_segments: int, order: float) -> CurveXYZFourier:
    curve = CurveXYZFourier(n_segments, order)
    curve.x = dofs
    curve.x = dofs
    return curve

debugging = False
#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

n_curves = 3
order = 1
coil_current=7e6
n_segments = 40
R = 3
A = 4 # Aspect ratio
r = R/A
r_init = r/4

curves_essos = CreateEquallySpacedCurves(n_curves, order, R, r, nfp=1, stellsym=False, n_segments=n_segments)
coils_essos = Coils(curves_essos, jnp.array([coil_current]*n_curves))
field_essos = BiotSavart_essos()
dofs = jnp.reshape(coils_essos.dofs, (n_curves, -1))

coils_simsopt = [Coil(simsopt_create_coil(dofs[i], n_segments, order), Current(coil_current)) for i in range(n_curves)]
field_simsopt = BiotSavart_simsopt(coils_simsopt)

# initial_values = stel.initial_conditions(particles, R, r_init, model="Guiding Center")
x = jnp.array([R+0.01,R,R])
y = jnp.array([R,R+0.01,R-0.01])
z = jnp.array([0.05,0.06,0.07])
positions = jnp.array((x,y,z))
real_norm_B = 2*jnp.pi*1e-7*coil_current*r**2/(r**2 + y**2)**(3/2)

#------------------------------------------------------------------------#
# Creating the coils
#------------------------------------------------------------------------#

list_segments = [10, 100, 1000]#, 10000] 
len_list_segments = len(list_segments)
t_avg = jnp.zeros(len_list_segments)
t_avg_simsopt = jnp.zeros(len_list_segments)
B_error_avg = jnp.zeros(len_list_segments)
B_error_avg_simsopt = jnp.zeros(len_list_segments)

for index, n_segments in enumerate(list_segments):
    coils_essos.n_segments = n_segments

    for i in range(n_curves):
        curve = simsopt_create_coil(dofs[i], n_segments, order)
        
        curves_points_simsopt = curve.gamma()
        dash_points_simsopt = curve.gammadash()
        curves_points = coils_essos.gamma[i]
        dash_points = coils_essos.gamma_dash[i]

        print("All close gamma", jnp.allclose(curves_points,curves_points_simsopt))
        print("All close gamma dash", jnp.allclose(dash_points, dash_points_simsopt))

    for j, position in enumerate(positions):
        time1 = time()
        result_B = field_essos.B(position, coils_essos)
        time2 = time()
        t_avg = t_avg.at[index].set(t_avg[index] + time2 - time1)
        normB = jnp.linalg.norm(result_B)

        field_simsopt.set_points(jnp.array([position]))
        time3 = time()
        result_simsopt = field_simsopt.B()
        time4 = time()

        t_avg_simsopt = t_avg_simsopt.at[index].set(t_avg_simsopt[index] + time4 - time3)

        normB_simsopt = jnp.linalg.norm(jnp.array(result_simsopt))

        B_error_avg = B_error_avg.at[index].set(B_error_avg[index] + jnp.abs(normB - real_norm_B[j])/real_norm_B[j])
        B_error_avg_simsopt = B_error_avg_simsopt.at[index].set(B_error_avg_simsopt[index] + jnp.abs(normB_simsopt - real_norm_B[j])/real_norm_B[j])

    print(f"Error in B: {B_error_avg[index]}")
    print(f"Time: {t_avg[index]}")
    print(f"Error in B simsopt: {B_error_avg_simsopt[index]}")
    print(f"Time simsopt: {t_avg_simsopt[index]}")
    print("------------------------------------------------------------------------")

# creating the dataset

fig = plt.figure(figsize = (10, 5))

X_axis = jnp.arange(len_list_segments) 

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
plt.savefig("error_B.pdf", transparent=True)

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
plt.savefig("time_comparison.pdf", transparent=True)

plt.show()