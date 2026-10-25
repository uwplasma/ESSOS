"""Forward native Boozer orbits, with optional independent CUDA particle loops."""
from functools import partial
import jax
import jax.numpy as jnp
from .boozer import _chart, guiding_center_rhs

def _boozer_error_norm(error, speed, rtol, atol):
    """Independent flux-radius, toroidal-radian and parallel-speed budgets."""
    return jnp.maximum(jnp.linalg.norm(error[:2]), jnp.maximum(
        jnp.abs(error[2]), jnp.abs(error[3])/speed)) / (rtol + atol)


def native_orbits(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                  method, adaptive, controller, pcoeff, icoeff, max_steps, *, execution="jax"):
    """Select portable JAX or fixed-field CUDA execution without changing physics.

    CUDA supports fixed steps and adaptive I control for cosine Fourier fields. The portable
    path also supports fixed steps, PI control, sine modes and angle tables.
    The choice is explicit: no backend, field layout or controller is changed.
    """
    if execution not in ("jax", "cuda"):
        raise ValueError("native_execution must be 'jax' or 'cuda'")
    solver = None
    if execution == "cuda":
        if controller != "i":
            raise ValueError("CUDA native orbits require I control")
        from .rk_boozer_gpu_orbit import prepare_orbit_solver
        solver = prepare_orbit_solver(field, mass, charge, float(speed), times,
                                      dt0, rtol, atol, max_steps, method=method, adaptive=adaptive)
    return _native_orbits(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                          method, adaptive, controller, pcoeff, icoeff, max_steps, solver)


@partial(jax.jit, static_argnames=("method", "adaptive", "controller", "max_steps", "orbit_solver"))
def _native_orbits(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                   method, adaptive, controller, pcoeff, icoeff, max_steps, orbit_solver=None):
    """One shared native RK solve per orbit, independent of the save grid."""
    from essos.rk import RKStatus, solve_rk

    def one(y0, mu):
        def magnetic_field(y):
            _, r, theta, *_ = _chart(y)
            return field.modB_derivatives(r, theta, y[2])[0]

        def rhs(t, y, args):
            del t, args
            value = guiding_center_rhs(field, y, mu, mass, charge)
            B = magnetic_field(y)
            return jnp.where(jnp.isfinite(B) & (B > 0), value, jnp.nan)

        def error_norm(y0, y1, error):
            del y0, y1
            # This field supplies |B| and profiles, not a Cartesian embedding.
            # Use a unit flux-radius chart, one radian for zeta and birth speed
            # for v_parallel. No scale grows with angle winding or vanishes at
            # a parallel turning point; these are chart errors, not metres.
            return _boozer_error_norm(error, speed, rtol, atol)

        if orbit_solver is None:
            solution = solve_rk(rhs, y0, t1=times[-1], dt0=dt0, ts=times,
                method=method, adaptive=adaptive, rtol=rtol, atol=atol,
                error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff,
                max_steps=max_steps, event=lambda t, y, args: 1. - y[0]**2 - y[1]**2)
        else:
            solution = orbit_solver(y0, mu)

        def output(y):
            s, _, theta, *_ = _chart(y)
            B = magnetic_field(y)
            energy = .5*y[3]**2 + mu*B
            state = jnp.array([s, theta, y[2], y[3], jnp.sqrt(jnp.maximum(2*energy, 0.))])
            valid = jnp.isfinite(state).all() & jnp.isfinite(energy) & (energy > 0) & (B > 0)
            return state, energy, valid

        # Hold an orbit at the root-found loss point or last accepted state
        # after failure. The solver itself leaves unreached saves infinite.
        sampled = jnp.where(jnp.isfinite(solution.ys).all(axis=1)[:, None], solution.ys, solution.y)
        values, energies, valid = jax.vmap(output)(sampled)
        _, end_energy, end_valid = output(solution.y)
        birth, birth_energy, _ = output(y0)
        output_failed = ~jnp.all(valid) | ~end_valid
        failed = (solution.status > RKStatus.EVENT) | output_failed
        lost = (solution.status == RKStatus.EVENT) & ~failed
        first_bad = jnp.min(jnp.where(valid, jnp.inf, times))
        failed_time = jnp.minimum(solution.t, first_bad)

        def hold(carry, item):
            previous, finite_prefix = carry
            value, finite = item
            finite_prefix = finite_prefix & finite
            previous = jnp.where(finite_prefix, value, previous)
            return (previous, finite_prefix), previous

        _, values = jax.lax.scan(hold, (birth, jnp.asarray(True)), (values, valid))
        energy_error = jnp.maximum(jnp.max(jnp.abs(energies/birth_energy - 1)),
                                    jnp.abs(end_energy/birth_energy - 1))
        return (values, jnp.where(lost, solution.t, -1.), jnp.where(failed, failed_time, -1.),
                jnp.where(failed, jnp.inf, energy_error), jnp.stack(solution.stats),
                jnp.where(output_failed, RKStatus.NONFINITE, solution.status))

    return jax.vmap(one)(y0, mu0)


