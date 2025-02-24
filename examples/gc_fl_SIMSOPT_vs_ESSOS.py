import os
import time
import jax.numpy as jnp
from simsopt import load
from simsopt.field import (particles_to_vtk, compute_fieldlines, plot_poincare_data)
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
from essos.fields import BiotSavart as BiotSavart_essos

tmax_fl = 600
nfieldlines = 3
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nfieldlines)
nfp = 2

Z0 = jnp.zeros(nfieldlines)
phi0 = jnp.zeros(nfieldlines)

phis_poincare = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
json_file = os.path.join(os.path.dirname(__file__), 'input', 'biot_savart_opt.json')
field_simsopt = load(json_file)
field_essos = BiotSavart_essos(Coils_from_simsopt(json_file, nfp))

tracing = Tracing(field=field_essos, model='FieldLine')
fieldlines_essos = tracing.trace(initial_conditions=jnp.array([R0*jnp.cos(phi0), R0*jnp.sin(phi0), Z0]).T)
print(fieldlines_essos)
exit()

t1 = time.time()
fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(field_simsopt, R0, Z0, tmax=tmax_fl, tol=1e-11, phis=phis_poincare)
t2 = time.time()
print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}")

particles_to_vtk(fieldlines_tys, os.path.join(output_dir,f'fieldlines_SIMSOPT'))
plot_poincare_data(fieldlines_phi_hits, phis_poincare, os.path.join(output_dir,f'poincare_fieldline_SIMSOPT.pdf'), dpi=150)