"""Full-orbit reference case with the existing checkpointed native AD runner.

Edit PARTICLE_OBJECTIVE and PARAMETERS below, then run from the checkout root:
python -m examples.coil_optimization.optimize_coils_particle_confinement_fullorbit_native

CLI arguments optionally override these settings. No optimization runs on import.
"""
from pathlib import Path
from .native_particle_optimization import run

# Select "final_radius", "loss_fraction", or "radial_increment". The last
# matches the intended legacy full-orbit metric with its time indexing fixed.
PARTICLE_OBJECTIVE = "final_radius"

# Physical setup from optimize_coils_particle_confinement_fullorbit.py.
# Solver, checkpointing, DOF bounds and optimizer use the native example path.
PARAMETERS = dict(
    coils=None,                       # Generate planar circular coils.
    coil_count=3,                     # Independent coils per half field period.
    coil_nfp=2,
    coil_stellsym=True,               # 12 physical coils after symmetry.
    coil_order=6,
    coil_segments=60,
    coil_major_radius=7.74,           # m
    coil_minor_radius=4.5,            # m
    coil_current=1.84e7,              # A
    coil_scaling_factor=0.,           # Same Fourier normalization as reference.
    particles=10,
    particle_setup="axis",           # Equally spaced around R=7.74 m, Z=0.
    birth_seed=42,                    # Fixed uniform pitch sample in [-1, 1].
    pitch_min=-1.,
    pitch_max=1.,
    energy_ev=3.52e6,                 # Alpha-particle mass and charge.
    gyrophase=0.,                     # radians; full-orbit birth conversion is active in AD.
    model="lorentz",
    horizon=1e-4,                     # s; final tracing time.
    initial_step=1e-14,               # Initial adaptive guess, not a fixed step.
    tolerance=1e-8,
    saves=1000,
    method="dopri8",
    checkpoint_steps=16,
    parameter_accumulation="segment",  # Shared GC/Lorentz reverse accumulation policy.
    accepted_capacity=65536,
    max_steps=262144,
    collisions=False,                 # Opt in to full split orbit/collision reverse AD.
    collision_adaptive=True,
    collision_controller="conservative",
    collision_tolerance=.003,
    collision_step=1e-7,              # s; initial adaptive or fixed split interval.
    collision_dtmax=1e-6,             # s; independent operator-splitting ceiling.
    collision_spatial_diffusion=False, # GC only.
    collision_checkpoint_steps=16,
    collision_accepted_capacity=4096, # Outer maps; accepted_capacity bounds the total inner RK tape.
    max_orbit_steps=4096,             # Inner orbit trial limit per collision interval.
    background_density=1e20,         # m^-3; one uniform Maxwellian species.
    background_temperature_ev=1000.,
    background_mass_proton=1.,
    background_charge_e=1.,
    coulomb_log_model="nrl",
    maxiter=200,
    max_evaluations=200,              # Native L-BFGS-B function evaluation limit.
    max_change=.01,                   # Bound in public normalized coil DOFs.
    particle_weight=1.,
    length_weight=1.,                 # Set any objective weight to zero to omit it.
    length_target=31.,                # m
    curvature_weight=1.,
    curvature_target=.4,              # 1/m
    bdotn_weight=1.,
    wout=Path(__file__).resolve().parents[1] / "input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc",
    target_surface="torus",          # Fixed classifier and Bn target, inside the coils.
    reference_major_radius=7.74,      # m; torus major radius.
    target_minor_radius=3.5,          # m; smaller than coil_minor_radius=4.5 m.
    safety_minor_radius=4.4,          # m; stop 10 cm inside the initial coil circles.
    surface_ntheta=30,
    surface_nphi=30,
    softness=10./7.74,                # 1/m; same relative sigmoid width as native example.
    axis_field_weight=1.,
    axis_field_target=5.7,            # T
    axis_field_points=15,
    regularization=0.,
    save_plots=True,                  # Matched 3D coil/orbit and R-Z views, PNG + SVG.
    plot_particles=10,                # Random subset of traced particles; same IDs in both views.
    plot_seed=42,
    output=Path(".tmp/native_fullorbit_confinement"),
)


def main(argv=None):
    run(PARTICLE_OBJECTIVE, argv, defaults=PARAMETERS)


if __name__ == "__main__":
    main()
