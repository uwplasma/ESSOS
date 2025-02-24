import os
from essos.dynamics import GuidingCenter, Lorentz, FieldLine
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
import jax.numpy as jnp
from essos.coils import Curves, Coils
from simsopt import load
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
import time

tmax_fl = 1000
nfieldlines = 2
axis_shft=0.02
R0 = jnp.linspace(1.2125346+axis_shft, 1.295-axis_shft, nfieldlines)
nfp = 2

Z0 = jnp.zeros(nfieldlines)
phis = [(i/4)*(2*jnp.pi/nfp) for i in range(4)]

output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
json_file = os.path.join(os.path.dirname(__file__), 'input', 'biot_savart_opt.json')
bs = load(json_file)

t1 = time.time()
fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(bs, R0, Z0, tmax=tmax_fl, tol=1e-11, phis=phis)
t2 = time.time()
print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}")

particles_to_vtk(fieldlines_tys, os.path.join(output_dir,f'fieldlines_SIMSOPT'))
plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(output_dir,f'poincare_fieldline_SIMSOPT.pdf'), dpi=150)