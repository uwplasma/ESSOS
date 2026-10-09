"""Where the cost of fixed-step field-line tracing actually comes from.

An earlier version of this example claimed the slowdown was XLA graph size
caused by max_steps=10000000000 on the fixed-step path. That was wrong.
Measured with AOT lowering, which separates compilation from execution:

  max_steps          compile      HLO size
  4,096              0.19 s       165,513 chars
  100,000            0.19 s       165,515 chars
  10,000,000,000     0.18 s       165,520 chars

max_steps changes neither compile time nor graph size. The cost is per-step
runtime, and it scales linearly with the number of steps (4 field lines):

  500 steps      0.066 s
  2,000 steps    0.255 s
  8,000 steps    1.034 s
  18,000 steps   2.306 s
  50,000 steps   6.306 s

The same trace with an adaptive PIDController (rtol=atol=1e-7) takes 0.0017 s,
because it takes far fewer, larger steps.

Note that the fixed-step path in Tracing does not pass a stepsize controller at
all, so it silently inherits the diffrax default. Users should be able to select
an explicit or implicit step size.

Run from the repository root:
    python examples/convergence_tests/fixed_step_tracing_cost.py
"""
import time, jax, jax.numpy as jnp, diffrax, json
from diffrax import diffeqsolve, ODETerm, SaveAt
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import near_axis, BiotSavart
from essos.dynamics import FieldLine

nfp = 3
field = near_axis(rc=jnp.array([1, 0.045]), zs=jnp.array([0, -0.045]), etabar=-0.9, nfp=nfp)
current = 17e5 * field.B0 / nfp / 2
curves = CreateEquallySpacedCurves(n_curves=3, order=5, R=field.R0[0],
                                   r=field.R0[0] / 2., n_segments=50, nfp=nfp, stellsym=True)
biot_savart = BiotSavart(Coils(curves=curves, currents=[current] * 3))
term = ODETerm(FieldLine)
tmax = 1e-4
n_save = 200


def measure(n_lines, n_steps, max_steps, controller):
    """Lower and compile ahead of time so compile and run are timed separately."""
    R0 = jnp.linspace(field.R0[0], 1.04 * field.R0[0], n_lines)
    y0 = jnp.stack([R0, jnp.zeros(n_lines), jnp.zeros(n_lines)], axis=-1)
    ts = jnp.linspace(0, tmax, n_save)

    def trace_one(initial_condition):
        return diffeqsolve(term, t0=0., t1=tmax, dt0=tmax / n_steps, y0=initial_condition,
                           solver=diffrax.Dopri5(), args=biot_savart, saveat=SaveAt(ts=ts),
                           throw=False, max_steps=max_steps,
                           stepsize_controller=controller).ys

    traced = jax.jit(jax.vmap(trace_one))
    lowered = traced.lower(y0)
    t0 = time.time(); compiled = lowered.compile(); t_compile = time.time() - t0
    t0 = time.time(); out = compiled(y0); jax.block_until_ready(out); t_run = time.time() - t0
    return t_compile, t_run, len(compiled.as_text())


results = {}

print('compile time and graph size vs max_steps (4 lines, 2000 fixed steps)')
for max_steps in [4096, 100_000, 10_000_000_000]:
    t_compile, t_run, hlo = measure(4, 2000, max_steps, diffrax.ConstantStepSize())
    print(f'  max_steps={max_steps:<15d} compile {t_compile:5.2f} s   run {t_run:6.3f} s   HLO {hlo} chars')
    results[f'max_steps_{max_steps}'] = dict(compile=t_compile, run=t_run, hlo=hlo)

print('\nrun time vs number of fixed steps (4 lines, max_steps generous)')
for n_steps in [500, 2000, 8000, 18000, 50000]:
    t_compile, t_run, _ = measure(4, n_steps, max(2 * n_steps, 4096), diffrax.ConstantStepSize())
    print(f'  {n_steps:6d} steps   compile {t_compile:5.2f} s   run {t_run:7.3f} s')
    results[f'steps_{n_steps}'] = dict(compile=t_compile, run=t_run)

print('\nsame trace with an adaptive controller')
t_compile, t_run, _ = measure(4, 2000, 4096, diffrax.PIDController(rtol=1e-7, atol=1e-7))
print(f'  PIDController 1e-7   compile {t_compile:5.2f} s   run {t_run:7.4f} s')
results['adaptive'] = dict(compile=t_compile, run=t_run)

json.dump(results, open('fixed_step_tracing_cost.json', 'w'), indent=1)
print('\nwrote fixed_step_tracing_cost.json')
