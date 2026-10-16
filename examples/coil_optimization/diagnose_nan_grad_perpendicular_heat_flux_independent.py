import os
from time import time

# Keep this diagnostic robust on machines with limited GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import lax

from essos.background_species import BackgroundSpecies
from essos.coils import Coils
from essos.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    ELECTRON_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
    PROTON_MASS,
    SPEED_OF_LIGHT,
)
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier


jax.config.update("jax_enable_x64", True)


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
        ntheta=40,
        nphi=40,
        range_torus="full torus",
    )
    vmec_axis = SurfaceRZFourier.from_wout_file(
        wout_file,
        s=0,
        ntheta=40,
        nphi=40,
        range_torus="full period",
    )

    boundary = SurfaceClassifier(vmec, h=0.1)
    species = build_species()

    particles = Particles.InitializeParticlesAroundSurfaceAxis(
        surface=vmec_axis,
        boundary_surface=vmec,
        n_particles=4,
        distance_from_axis=0.7,
        distance_mode="fraction_to_boundary",
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=FUSION_ALPHA_PARTICLE_ENERGY,
    )

    return coils_template, boundary, particles, species


def make_tracing(dofs, coils_template, particles, boundary, species):
    coils = coils_template.copy()
    coils.dofs = dofs
    field = BiotSavart(coils)
    return Tracing(
        field=field,
        particles=particles,
        maxtime=1.5e-6,
        model="GuidingCenterCollisionsMuFixed",
        times_to_trace=5,
        timestep=3.0e-8,
        species=species,
        boundary=boundary,
    )


def _common_terms(tracing, boundary, softness=30.0, stride=1, final_timestep_only=False):
    trajectories = tracing.trajectories
    trajectories_xyz = trajectories[:, :, :3]

    if final_timestep_only:
        trajectories_xyz = trajectories_xyz[:, -1:, :]
        trajectories_sampled = trajectories[:, -1:, :]
    else:
        trajectories_xyz = trajectories_xyz[:, ::stride, :]
        trajectories_sampled = trajectories[:, ::stride, :]

    nparticles, nt = trajectories_xyz.shape[:2]

    finite_xyz = jnp.all(jnp.isfinite(trajectories_xyz), axis=2)
    trajectories_xyz_safe = jnp.where(finite_xyz[:, :, None], trajectories_xyz, 0.0)

    distances_flat = jax.vmap(boundary.evaluate_xyz)(trajectories_xyz_safe.reshape(-1, 3))
    distances = distances_flat.reshape(nparticles, nt)

    outside_prob = jax.nn.sigmoid(-softness * distances)
    outside_prob = jnp.where(finite_xyz, outside_prob, 0.0)

    cumulative_loss_prob = lax.associative_scan(jnp.maximum, outside_prob, axis=1)
    first_loss_prob = jnp.concatenate(
        [cumulative_loss_prob[:, :1], cumulative_loss_prob[:, 1:] - cumulative_loss_prob[:, :-1]],
        axis=1,
    )
    first_loss_prob = jnp.maximum(first_loss_prob, 0.0)

    return trajectories_sampled, finite_xyz, first_loss_prob


def manual_qperp_raw(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    trajectories_sampled, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    mass = tracing.particles.mass
    vpar = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    energy_sampled = tracing.energy()[:, ::1]

    finite_energy = jnp.isfinite(energy_sampled)
    finite_vpar = jnp.isfinite(vpar)
    valid_mask = finite_xyz & finite_energy & finite_vpar

    e_parallel = 0.5 * mass * jnp.square(vpar)
    e_perp = jnp.maximum(energy_sampled - e_parallel, 0.0)
    e_perp = jnp.where(valid_mask, e_perp, 0.0)
    first_loss_prob = jnp.where(valid_mask, first_loss_prob, 0.0)

    weighted_q_perp = jnp.sum(first_loss_prob * e_perp)
    total_loss_weight = jnp.sum(first_loss_prob)
    return weighted_q_perp / (total_loss_weight + 1e-8)


def manual_qperp_mask_safe(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    trajectories_sampled, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    mass = tracing.particles.mass
    vpar_raw = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    energy_raw = tracing.energy()[:, ::1]

    finite_energy = jnp.isfinite(energy_raw)
    finite_vpar = jnp.isfinite(vpar_raw)
    valid_mask = finite_xyz & finite_energy & finite_vpar

    vpar = jnp.where(finite_vpar, vpar_raw, 0.0)
    energy = jnp.where(finite_energy, energy_raw, 0.0)
    delta = energy - 0.5 * mass * jnp.square(vpar)
    delta = jnp.where(valid_mask, delta, 0.0)

    # Smooth non-negative clamp avoids max() kinks in the diagnostic variant.
    e_perp = jax.nn.softplus(delta * 1e10) / 1e10
    e_perp = jnp.nan_to_num(e_perp, nan=0.0, posinf=0.0, neginf=0.0)

    first_loss_prob = jnp.where(valid_mask, first_loss_prob, 0.0)
    first_loss_prob = jnp.nan_to_num(first_loss_prob, nan=0.0, posinf=0.0, neginf=0.0)

    weighted_q_perp = jnp.sum(first_loss_prob * e_perp)
    total_loss_weight = jnp.sum(first_loss_prob)
    return weighted_q_perp / (total_loss_weight + 1e-8)


def builtin_qperp(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    return tracing.loss_perpendicular_heat_flux_classifier_differentiable(
        boundary=boundary,
        softness=30.0,
        stride=1,
        final_timestep_only=False,
    )


def builtin_stopgrad_prob(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    trajectories_sampled, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    mass = tracing.particles.mass
    vpar = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    energy_sampled = tracing.energy()[:, ::1]
    finite_energy = jnp.isfinite(energy_sampled)
    finite_vpar = jnp.isfinite(vpar)
    valid_mask = finite_xyz & finite_energy & finite_vpar

    e_parallel = 0.5 * mass * jnp.square(vpar)
    e_perp = jnp.maximum(energy_sampled - e_parallel, 0.0)
    e_perp = jnp.where(valid_mask, e_perp, 0.0)

    prob = jax.lax.stop_gradient(jnp.where(valid_mask, first_loss_prob, 0.0))
    weighted_q_perp = jnp.sum(prob * e_perp)
    total_loss_weight = jnp.sum(prob)
    return weighted_q_perp / (total_loss_weight + 1e-8)


def builtin_stopgrad_eperp(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    trajectories_sampled, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    mass = tracing.particles.mass
    vpar = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    energy_sampled = tracing.energy()[:, ::1]
    finite_energy = jnp.isfinite(energy_sampled)
    finite_vpar = jnp.isfinite(vpar)
    valid_mask = finite_xyz & finite_energy & finite_vpar

    e_parallel = 0.5 * mass * jnp.square(vpar)
    e_perp = jnp.maximum(energy_sampled - e_parallel, 0.0)
    e_perp = jax.lax.stop_gradient(jnp.where(valid_mask, e_perp, 0.0))

    prob = jnp.where(valid_mask, first_loss_prob, 0.0)
    weighted_q_perp = jnp.sum(prob * e_perp)
    total_loss_weight = jnp.sum(prob)
    return weighted_q_perp / (total_loss_weight + 1e-8)


def energy_only_branch(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    _, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    energy_sampled = tracing.energy()[:, ::1]
    finite_energy = jnp.isfinite(energy_sampled)
    valid_mask = finite_xyz & finite_energy

    e_perp = jnp.where(valid_mask, energy_sampled, 0.0)
    prob = jnp.where(valid_mask, first_loss_prob, 0.0)
    weighted = jnp.sum(prob * e_perp)
    total = jnp.sum(prob)
    return weighted / (total + 1e-8)


def vpar_only_branch(dofs, coils_template, particles, boundary, species):
    tracing = make_tracing(dofs, coils_template, particles, boundary, species)
    trajectories_sampled, finite_xyz, first_loss_prob = _common_terms(tracing, boundary)

    mass = tracing.particles.mass
    vpar = SPEED_OF_LIGHT * trajectories_sampled[:, :, 3]
    finite_vpar = jnp.isfinite(vpar)
    valid_mask = finite_xyz & finite_vpar

    e_par = 0.5 * mass * jnp.square(jnp.where(finite_vpar, vpar, 0.0))
    e_perp = jnp.where(valid_mask, e_par, 0.0)
    prob = jnp.where(valid_mask, first_loss_prob, 0.0)
    weighted = jnp.sum(prob * e_perp)
    total = jnp.sum(prob)
    return weighted / (total + 1e-8)


def run_variant(name, fn, dofs0, coils_template, particles, boundary, species):
    print(f"\n--- {name} ---")
    objective = lambda d: fn(d, coils_template, particles, boundary, species)
    t0 = time()
    value, grad = jax.value_and_grad(objective)(dofs0)
    dt = time() - t0

    finite_grad = jnp.isfinite(grad)
    n_total = int(grad.size)
    n_bad = int(n_total - jnp.sum(finite_grad))

    print(f"value: {float(value):.6e}")
    print(f"value finite: {bool(jnp.isfinite(value))}")
    print(f"grad finite: {bool(jnp.all(finite_grad))}")
    print(f"grad non-finite: {n_bad}/{n_total}")
    if n_bad > 0:
        bad_idx = jnp.where(~finite_grad)[0]
        print(f"first bad idx: {bad_idx[:10]}")
    else:
        print(f"grad l2: {float(jnp.linalg.norm(grad)):.6e}")
    print(f"elapsed: {dt:.2f} s")


def main():
    print("Building diagnostic setup...")
    t0 = time()
    coils_template, boundary, particles, species = build_inputs()
    print(f"Setup done in {time() - t0:.2f} s")

    dofs0 = coils_template.dofs
    print(f"DOFs: {dofs0.shape}")

    variants = [
        ("builtin", builtin_qperp),
        ("manual_raw", manual_qperp_raw),
        ("stopgrad_prob", builtin_stopgrad_prob),
        ("stopgrad_eperp", builtin_stopgrad_eperp),
        ("energy_only_branch", energy_only_branch),
        ("vpar_only_branch", vpar_only_branch),
        ("manual_mask_safe", manual_qperp_mask_safe),
    ]

    for name, fn in variants:
        run_variant(name, fn, dofs0, coils_template, particles, boundary, species)


if __name__ == "__main__":
    main()
