"""Guiding-center or full-orbit reference confinement case with checkpointed native reverse AD.

Edit PARTICLE_OBJECTIVE and PARAMETERS below (or pass options), then run from the checkout root:
python -m examples.coil_optimization.optimize_coils_particle_confinement_native --model gc|lorentz

Physical setups follow optimize_coils_particle_confinement_{guidingcenter,fullorbit}.py. Solver,
checkpointing, DOF bounds and optimizer use the native example path. No optimization runs on import.
"""
import argparse
from pathlib import Path
from .native_particle_optimization import run

# "final_radius", "loss_fraction" or "radial_increment" (full-orbit metric with its time indexing fixed).
PARTICLE_OBJECTIVE = "final_radius"

PARAMETERS = dict(
    coils=None,                       # generate planar circular coils: 3 per half period, nfp 2, stellsym
    coil_count=3, coil_nfp=2, coil_stellsym=True, coil_order=6, coil_segments=60,
    coil_major_radius=7.74, coil_minor_radius=4.5, coil_current=1.84e7,   # m, m, A
    particles=10, particle_setup="axis",    # equally spaced around R=7.74 m, Z=0; fixed uniform pitches
    energy_ev=3.52e6, gyrophase=0.,         # alpha particles
    horizon=1e-4, initial_step=1e-14,       # s; initial adaptive guess, not a fixed step
    tolerance=1e-8, saves=1000, maxiter=200, max_change=.01,
    collisions=False,                       # True: split orbit/collision reverse AD with one background species
    length_target=31., curvature_target=.4, axis_field_weight=1.,
    target_surface="torus", reference_major_radius=7.74, target_minor_radius=3.5,   # inside the coils
    safety_minor_radius=4.4,                # stop 10 cm inside the initial coil circles
    softness=10./7.74)
PRESETS = dict(
    gc=dict(model="gc", coil_scaling_factor=1.2, accepted_capacity=8192, max_steps=32768,
            output=Path("output/native_gc_confinement")),
    lorentz=dict(model="lorentz", coil_scaling_factor=0., accepted_capacity=65536, max_steps=262144,
                 output=Path("output/native_fullorbit_confinement")))


def main(argv=None):
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--model", choices=tuple(PRESETS), default="gc")
    run(PARTICLE_OBJECTIVE, argv, defaults=PARAMETERS | PRESETS[model.parse_known_args(argv)[0].model])


if __name__ == "__main__":
    main()
