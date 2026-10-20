import os
from time import time

# Keep this standalone check robust on machines with limited GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import (
    ELECTRON_MASS,
    PROTON_MASS,
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier


def build_species():
    number_species = 2
    n0 = 4.0e20
    t0 = 6.6e3
    return BackgroundSpecies(
        number_species=number_species,
        mass_array=jnp.array([ELECTRON_MASS / PROTON_MASS, 2.0]),
        charge_array=jnp.array([-1.0, 1.0]),
        n_array=jnp.array([n0, n0]),
        T_array=jnp.array([t0, t0]),
    )


def build_inputs():
    base_dir = os.path.dirname(__file__)
    json_file = os.path.join(base_dir, "../input_files", "QI_opt_coils_essos_5.json")
    wout_file = os.path.join(
        base_dir,
        "../input_files",
        "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc",
    )

    coils_template = Coils.from_json(json_file)

    vmec = SurfaceRZFourier.from_wout_file(
        wout_file,
        s=1,
        ntheta=60,
        nphi=60,
        range_torus="full torus",
    )
    vmec_axis = SurfaceRZFourier.from_wout_file(
        wout_file,
        s=0,
        ntheta=60,
        nphi=60,
        range_torus="full period",
    )

    boundary = SurfaceClassifier(vmec, h=0.1)
    species = build_species()

    particles = Particles.InitializeParticlesAroundSurfaceAxis(
        surface=vmec_axis,
        boundary_surface=vmec,
        n_particles=6,
        distance_from_axis=0.7,
        distance_mode="fraction_to_boundary",
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=FUSION_ALPHA_PARTICLE_ENERGY,
    )

    return coils_template, boundary, particles, species


def loss_from_dofs(
    dofs,
    coils_template,
    boundary,
    particles,
    species,
    t=2.0e-6,
    times_to_trace=6,
    timestep=2.0e-8,
    model="GuidingCenterCollisionsMuFixed",
):
    # Rebuild from template each call to keep this objective independent and side-effect free.
    coils = coils_template.copy()
    coils.dofs = dofs
    field = BiotSavart(coils)

    tracing = Tracing(
        field=field,
        particles=particles,
        maxtime=t,
        model=model,
        times_to_trace=times_to_trace,
        timestep=timestep,
        species=species,
        boundary=boundary,
    )

    return tracing.loss_perpendicular_heat_flux_classifier_differentiable(
        boundary=boundary,
        softness=30.0,
        stride=1,
        final_timestep_only=False,
    )


def main():
    jax.config.update("jax_enable_x64", True)

    print("Building setup...")
    t0 = time()
    coils_template, boundary, particles, species = build_inputs()
    print(f"Setup done in {time() - t0:.2f} s")

    dofs0 = coils_template.dofs
    objective = lambda d: loss_from_dofs(d, coils_template, boundary, particles, species)

    print("Computing value and gradient...")
    t1 = time()
    loss_value, grad_value = jax.value_and_grad(objective)(dofs0)
    print(f"Gradient evaluation done in {time() - t1:.2f} s")

    loss_is_finite = bool(jnp.isfinite(loss_value))
    grad_finite_mask = jnp.isfinite(grad_value)
    grad_all_finite = bool(jnp.all(grad_finite_mask))

    print("\n=== Gradient Finiteness Check ===")
    print(f"loss value              : {float(loss_value):.6e}")
    print(f"loss is finite          : {loss_is_finite}")
    print(f"gradient shape          : {grad_value.shape}")
    print(f"gradient L2 norm        : {float(jnp.linalg.norm(grad_value)):.6e}")
    print(f"gradient max abs        : {float(jnp.max(jnp.abs(grad_value))):.6e}")
    print(f"gradient all finite     : {grad_all_finite}")

    n_nonfinite = int(jnp.size(grad_finite_mask) - jnp.sum(grad_finite_mask))
    print(f"gradient non-finite cnt : {n_nonfinite}")

    if n_nonfinite > 0:
        bad_idx = jnp.where(~grad_finite_mask)[0]
        print(f"first non-finite indices: {bad_idx[:10]}")

    if not (loss_is_finite and grad_all_finite):
        raise RuntimeError("Non-finite value detected in loss and/or gradient.")

    print("PASS: loss and gradient are finite.")


if __name__ == "__main__":
    main()
