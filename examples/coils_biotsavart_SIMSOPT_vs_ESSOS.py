from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import block_until_ready
from simsopt.geo import CurveXYZFourier
from simsopt.configs import get_ncsx_data
from essos.fields import BiotSavart as BiotSavart_essos
from essos.coils import CreateEquallySpacedCurves, Coils, Coils_from_simsopt, Curves_from_simsopt
from simsopt.field import BiotSavart as BiotSavart_simsopt, Current, Coil, coils_via_symmetries

n_curves = 4
order = 1
coil_current=7e6
R = 3
r = 0.75
x = jnp.array([R+0.01,R,R])
y = jnp.array([R,R+0.01,R-0.01])
z = jnp.array([0.05,0.06,0.07])
list_segments = [30, 100, 300, 1000, 3000]

positions = jnp.array((x,y,z))

len_list_segments = len(list_segments)
t_gamma_avg_essos = jnp.zeros(len_list_segments)
t_gamma_avg_simsopt = jnp.zeros(len_list_segments)
gamma_error_avg = jnp.zeros(len_list_segments)
t_gammadash_avg_essos = jnp.zeros(len_list_segments)
t_gammadash_avg_simsopt = jnp.zeros(len_list_segments)
gammadash_error_avg = jnp.zeros(len_list_segments)
t_B_avg_essos = jnp.zeros(len_list_segments)
t_B_avg_simsopt = jnp.zeros(len_list_segments)
B_error_avg = jnp.zeros(len_list_segments)
t_dB_by_dX_avg_essos = jnp.zeros(len_list_segments)
t_dB_by_dX_avg_simsopt = jnp.zeros(len_list_segments)
dB_by_dX_error_avg = jnp.zeros(len_list_segments)

nfp = 3
n_curves = 3
curves_ncsx, currents_ncsx, _ = get_ncsx_data()

coils_simsopt  = coils_via_symmetries(curves_ncsx, currents_ncsx, nfp, True)
curves_simsopt = [c.curve for c in coils_simsopt]
currents_simsopt = [c.current for c in coils_simsopt]

coils_essos = Coils_from_simsopt(coils_simsopt[0:n_curves], nfp)
curves_essos = Curves_from_simsopt(curves_simsopt[0:n_curves], nfp)

coils_essos_to_simsopt = coils_essos.to_simsopt()
curves_essos_to_simsopt = curves_essos.to_simsopt()

relative_error_simsopt_to_essos = 0
relative_error_essos_to_simsopt = 0
for i, (coil_simsopt, coil_essos_gamma, coil_essos_to_simsopt) in enumerate(zip(coils_simsopt, coils_essos.gamma, coils_essos_to_simsopt)):
    gamma_error_simsopt_to_essos = jnp.linalg.norm(coil_simsopt.curve.gamma()-coil_essos_gamma)
    gamma_error_essos_to_simsopt = jnp.linalg.norm(coil_simsopt.curve.gamma()-coil_essos_to_simsopt.curve.gamma())

field_simsopt = BiotSavart_simsopt(coils_simsopt)
field_essos = BiotSavart_essos(coils_essos)
field_essos_to_simsopt = BiotSavart_simsopt(coils_essos_to_simsopt)

B_error_avg_simsopt_to_essos = 0
B_error_avg_essos_to_simsopt = 0
for j, position in enumerate(positions):
    field_simsopt.set_points([position])
    field_essos_to_simsopt.set_points([position])
    B_simsopt = field_simsopt.B()
    B_essos_to_simsopt = field_essos_to_simsopt.B()
    B_simsopt_to_essos = field_essos.B(position)
    B_error_avg_simsopt_to_essos = B_error_avg_simsopt_to_essos + jnp.abs(jnp.linalg.norm(B_simsopt) - jnp.linalg.norm(B_simsopt_to_essos))
    B_error_avg_essos_to_simsopt = B_error_avg_essos_to_simsopt + jnp.abs(jnp.linalg.norm(B_simsopt) - jnp.linalg.norm(B_essos_to_simsopt))
B_error_avg_simsopt_to_essos = B_error_avg_simsopt_to_essos/len(positions)
B_error_avg_essos_to_simsopt = B_error_avg_essos_to_simsopt/len(positions)

fig = plt.figure(figsize = (8, 6))
X_axis = jnp.arange(2)
plt.bar(X_axis[0] - 0.2, gamma_error_simsopt_to_essos+1e-19, 0.3, label='SIMSOPT to ESSOS coils', color='blue', edgecolor='black', hatch='/')
plt.bar(X_axis[0] + 0.2, gamma_error_essos_to_simsopt+1e-19, 0.3, label='ESSOS to SIMSOPT coils', color='red', edgecolor='black', hatch='-')
plt.bar(X_axis[1] - 0.2, B_error_avg_simsopt_to_essos+1e-19, 0.3, label=r'SIMSOPT to ESSOS $B$', color='blue', edgecolor='black', hatch='||')
plt.bar(X_axis[1] + 0.2, B_error_avg_essos_to_simsopt+1e-19, 0.3, label=r'ESSOS to SIMSOPT $B$', color='red', edgecolor='black', hatch='*')
plt.xticks(X_axis, ['Coil Error', 'B Error'])
plt.xlabel('Error Type', fontsize=14)
plt.ylabel('Error Magnitude', fontsize=14)
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid(axis='y')
fig.tight_layout()
plt.savefig("error_gamma_B_SIMSOPT_vs_ESSOS.pdf", transparent=True)

for index, n_segments in enumerate(list_segments):
    for i in range(n_curves):
        curve = coils_simsopt[i].curve
        
        curve.gamma()
        curve.gammadash()
        coils_essos.gamma[i]
        coils_essos.gamma_dash[i]
        
        t_gamma_avg_simsopt = t_gamma_avg_simsopt.at[index].set(t_gamma_avg_simsopt[index] + 0)
        
        time1 = time()
        curves_points_simsopt = block_until_ready(curve.gamma())
        t_gamma_avg_simsopt = t_gamma_avg_simsopt.at[index].set(t_gamma_avg_simsopt[index] + time() - time1)
        time1 = time()
        curves_points = block_until_ready(coils_essos.gamma[i])
        t_gamma_avg_essos = t_gamma_avg_essos.at[index].set(t_gamma_avg_essos[index] + time() - time1)
        time1 = time()
        dash_points_simsopt = block_until_ready(curve.gammadash())
        t_gammadash_avg_simsopt = t_gammadash_avg_simsopt.at[index].set(t_gammadash_avg_simsopt[index] + time() - time1)
        time1 = time()
        dash_points = block_until_ready(coils_essos.gamma_dash[i])
        t_gammadash_avg_essos = t_gammadash_avg_essos.at[index].set(t_gammadash_avg_essos[index] + time() - time1)
        
        gamma_error_avg = gamma_error_avg.at[index].set(gamma_error_avg[index] + jnp.linalg.norm(curves_points - curves_points_simsopt))
        gammadash_error_avg = gammadash_error_avg.at[index].set(gammadash_error_avg[index] + jnp.linalg.norm(dash_points - dash_points_simsopt))

    for j, position in enumerate(positions):
        field_essos.B(position)
        time1 = time()
        result_B_essos = field_essos.B(position)
        t_B_avg_essos = t_B_avg_essos.at[index].set(t_B_avg_essos[index] + time() - time1)
        normB_essos = jnp.linalg.norm(result_B_essos)

        field_simsopt.set_points(jnp.array([position]))
        field_simsopt.B()
        time3 = time()
        result_simsopt = field_simsopt.B()
        t_B_avg_simsopt = t_B_avg_simsopt.at[index].set(t_B_avg_simsopt[index] + time() - time3)
        normB_simsopt = jnp.linalg.norm(jnp.array(result_simsopt))

        B_error_avg = B_error_avg.at[index].set(B_error_avg[index] + jnp.abs(normB_essos - normB_simsopt))
        
        field_essos.dB_by_dX(position)
        time1 = time()
        result_dB_by_dX_essos = field_essos.dB_by_dX(position)
        t_dB_by_dX_avg_essos = t_dB_by_dX_avg_essos.at[index].set(t_dB_by_dX_avg_essos[index] + time() - time1)
        norm_dB_by_dX_essos = jnp.linalg.norm(result_dB_by_dX_essos)
        
        field_simsopt.dB_by_dX()
        time3 = time()
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
plt.xticks(X_axis, list_segments) 
plt.xlabel("Number of segments of each coil", fontsize=14)
plt.ylabel(f"Difference SIMSOPT vs ESSOS", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.legend(fontsize=14)
plt.yscale("log")
plt.grid(axis='y')
fig.tight_layout()
plt.savefig("error_BiotSavart_SIMSOPT_vs_ESSOS.pdf", transparent=True)

fig = plt.figure(figsize = (8, 6))
plt.bar(X_axis - 0.3, t_B_avg_essos, 0.1, label = r'B ESSOS', color="red", edgecolor="black", hatch="/")
plt.bar(X_axis - 0.2, t_B_avg_simsopt, 0.1, label = r'B SIMSOPT', color="blue", edgecolor="black", hatch="-")
plt.bar(X_axis - 0.1, t_dB_by_dX_avg_essos, 0.1, label = r"$B'$ ESSOS", color="red", edgecolor="black", hatch="/")
plt.bar(X_axis + 0.0, t_dB_by_dX_avg_simsopt, 0.1, label = r"$B'$ SIMSOPT", color="blue", edgecolor="black", hatch="-")
plt.bar(X_axis + 0.1, t_gamma_avg_essos, 0.1, label = r'$\Gamma$ ESSOS', color="red", edgecolor="black", hatch="/")
plt.bar(X_axis + 0.2, t_gamma_avg_simsopt, 0.1, label = r'$\Gamma$ SIMSOPT', color="blue", edgecolor="black", hatch="-")
plt.bar(X_axis + 0.3, t_gammadash_avg_essos, 0.1, label = r"${\Gamma'}$ ESSOS", color="red", edgecolor="black", hatch="/")
plt.bar(X_axis + 0.4, t_gammadash_avg_simsopt, 0.1, label = r"${\Gamma'}$ SIMSOPT", color="blue", edgecolor="black", hatch="-")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.xticks(X_axis, list_segments) 
plt.xlabel("Number of segments of each coil", fontsize=14)
plt.ylabel("Time to evaluate SIMSOPT vs ESSOS (s)", fontsize=14)
plt.grid(axis='y')
plt.legend(fontsize=14)
fig.tight_layout()
plt.savefig("time_BiotSavart_SIMSOPT_vs_ESSOS.pdf", transparent=True)

plt.show()