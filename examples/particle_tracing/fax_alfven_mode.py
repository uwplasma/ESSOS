"""3.5 MeV alphas in a reactor-scale QA with and without a FAX Alfven eigenmode (about 2 CPU minutes).

The mode (m = 3..6, n = 2, 22 kHz) is the lowest global radial eigenmode of FAX v0.0.2 on this wout,
saved in examples/input_files/fax_mode. It is regenerated with FAX (needs booz_xform_jax)::

    geometry = run_booz_xform_jax_geometry_from_wout(WOUT, mboz=8, nboz=8, surfaces=tuple(np.linspace(.04, .96, 24)))
    problem = radial_block_problem_from_boozer_geometry(geometry, [(3, 2), (4, 2), (5, 2), (6, 2)], mass_density=3.34e-7,
                  radial_stiffness_method="mass", radial_coupling_fraction=.01, theta_points=16, zeta_points=16)
    result = solve_radial_block_eigenmodes(problem)  # then write global_eigenmodes.csv / global_mode_structures.csv

Only the potential structure and frequency come from FAX; amplitude is an input and there is no feedback on the mode.
Needs ``pip install -e '.[booz]'``. From the checkout root: python examples/particle_tracing/fax_alfven_mode.py
"""

from pathlib import Path

import jax
import numpy as np
from booz_xform_jax import Booz_xform
from netCDF4 import Dataset

from essos.boozer import BoozerField, psi0_from_vmec, trace_boozer
from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY
from essos.perturbations import FaxEigenmode

PARTICLES, AMPLITUDE, TMAX, SEED = 64, 2e4, 5e-4, 1  # amplitude: peak potential [V] of the dominant harmonic
EXAMPLES = Path(__file__).resolve().parents[1]
WOUT = EXAMPLES/"input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc"

booz = Booz_xform(verbose=0, mboz=8, nboz=8)
booz.read_wout(str(WOUT), flux=False)
with Dataset(WOUT) as nc:
    psi0 = psi0_from_vmec(float(nc.variables["phi"][-1]))
field = BoozerField.from_booz_xform(booz, psi0)
mode = FaxEigenmode.from_fax(EXAMPLES/"input_files/fax_mode", AMPLITUDE)

rng = np.random.default_rng(SEED)
s, theta, zeta, pitch = rng.uniform(.2, .8, PARTICLES), *(rng.uniform(0, 2*np.pi, PARTICLES) for _ in range(2)), \
    rng.uniform(-1, 1, PARTICLES)
speed = np.sqrt(2*float(FUSION_ALPHA_PARTICLE_ENERGY)/ALPHA_PARTICLE_MASS)
for name, wave in (("no wave", None), ("FAX mode", mode)):
    out = trace_boozer(field, s, theta, zeta, pitch, speed=speed, mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                       tmax=TMAX, n_save=11, method="adaptive8", tolerance=1e-8, perturbation=wave, progress=False,
                       devices=jax.devices()[:1])
    change = out.energy_error if wave is None else out.kinetic_energy_change
    print(f"{name:9s} lost {100*out.loss_fractions()[-1]:5.1f}%  max |dE/E| {np.max(abs(change)):.1e}")
