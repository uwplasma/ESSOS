import os
import jax
import jax.numpy as jnp

from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from essos.dynamics import GuidingCenter, Particles
from essos.objective_functions import normB_axis


def describe_case(label, coils, vmec_surface, initial_xyz, initial_vpar):
    print(f"\n=== {label} ===")
    print("n_base_curves:", coils.curves.n_base_curves)
    print("n_segments:", coils.n_segments)
    print("nfp:", coils.nfp, "stellsym:", coils.stellsym)
    print("dofs_curves shape:", coils.dofs_curves.shape)
    print("dofs_currents_raw shape:", coils.dofs_currents_raw.shape)
    print("currents raw min/max:", float(jnp.min(coils.dofs_currents_raw)), float(jnp.max(coils.dofs_currents_raw)))
    print("currents scale:", float(coils.currents_scale))
    print("currents normalized min/max:", float(jnp.min(coils.dofs_currents)), float(jnp.max(coils.dofs_currents)))
    print("gamma shape:", coils.gamma.shape)
    print("gamma finite:", bool(jnp.all(jnp.isfinite(coils.gamma))))
    print("gamma_dash finite:", bool(jnp.all(jnp.isfinite(coils.gamma_dash))))
    print("gamma_dashdash finite:", bool(jnp.all(jnp.isfinite(coils.gamma_dashdash))))

    field = BiotSavart(coils)
    B_axis = normB_axis(field, npoints=200)
    print("axis B mean before renorm:", float(jnp.mean(B_axis)))
    coils.dofs_currents = coils.dofs_currents * 5.7 / jnp.mean(B_axis)
    field = BiotSavart(coils)
    print("axis B mean after renorm:", float(jnp.mean(normB_axis(field, npoints=200))))

    print("surface gamma finite:", bool(jnp.all(jnp.isfinite(vmec_surface.gamma))))
    print("initial xyz:", initial_xyz)
    print("field.B(initial):", field.B(initial_xyz))
    print("field.AbsB(initial):", float(field.AbsB(initial_xyz)))
    print("field.dAbsB_by_dX(initial):", field.dAbsB_by_dX(initial_xyz))
    print("field.curl_b(initial):", field.curl_b(initial_xyz))
    print("field.kappa(initial):", field.kappa(initial_xyz))

    finite_checks = {
        "B": jnp.all(jnp.isfinite(field.B(initial_xyz))),
        "AbsB": jnp.isfinite(field.AbsB(initial_xyz)),
        "dAbsB_by_dX": jnp.all(jnp.isfinite(field.dAbsB_by_dX(initial_xyz))),
        "curl_b": jnp.all(jnp.isfinite(field.curl_b(initial_xyz))),
        "kappa": jnp.all(jnp.isfinite(field.kappa(initial_xyz))),
    }
    print("finite checks:", {k: bool(v) for k, v in finite_checks.items()})

    particles = Particles(
        initial_xyz=jnp.expand_dims(initial_xyz, 0),
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=FUSION_ALPHA_PARTICLE_ENERGY,
    )
    initial_condition = jnp.array([initial_xyz[0], initial_xyz[1], initial_xyz[2], initial_vpar])
    rhs = GuidingCenter(0.0, initial_condition, (field, particles, type("E", (), {"E_covariant": lambda self, x: jnp.zeros(3)})()))
    print("GuidingCenter rhs:", rhs)
    print("GuidingCenter rhs finite:", bool(jnp.all(jnp.isfinite(rhs))))


def main():
    print(jax.devices())

    simsopt_json = os.path.join("examples", "input_files", "QH_simple_scaled.json")
    wout_file = os.path.join("examples", "input_files", "wout_QH_simple_scaled.nc")
    vmec = Vmec(wout_file)

    R0 = 17.0
    initial_xyz = jnp.array([R0, 0.0, 0.0])
    initial_vpar = 0.0

    coils_true = Coils.from_simsopt(simsopt_json, nfp=4, stellsym=True)
    describe_case("from_simsopt stellsym=True", coils_true, vmec.surface, initial_xyz, initial_vpar)

    coils_false = Coils.from_simsopt(simsopt_json, nfp=4, stellsym=False)
    describe_case("from_simsopt stellsym=False", coils_false, vmec.surface, initial_xyz, initial_vpar)

    from simsopt import load

    bs = load(simsopt_json)
    base_coils = bs.coils[:6]
    print("\n=== direct simsopt base coils ===")
    print("n base coils:", len(base_coils))
    print("current values:", [float(c.current.get_value()) for c in base_coils])
    print("curve gamma shape:", jnp.asarray(base_coils[0].curve.gamma()).shape)
    print("curve gamma first point:", jnp.asarray(base_coils[0].curve.gamma())[0])


if __name__ == "__main__":
    main()
