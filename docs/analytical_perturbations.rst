:orphan:

Prescribed analytical Alfvén perturbations
========================================

The existing GC equations accept potentials ``delta A = alpha B0`` and ``Phi``:

.. math::

   \delta\mathbf B=\nabla\alpha\times\mathbf B_0+\alpha\nabla\times\mathbf B_0,
   \qquad\delta\mathbf E=-\nabla\Phi-\partial_t\alpha\,\mathbf B_0.

The reduced action retains the static background magnitude in
``H = m*v_parallel**2/2 + mu*B0 + q*Phi``. Physical magnetic moment is constant
between collisions; wave work changes kinetic energy. Full geometry retains
all covariant field components, including VMEC's radial ``B_s``.
These are imposed waves without growth, saturation or particle feedback.

``AnalyticalAlfvenMode(m,n,omega,amplitude)`` uses a sinusoidal potential in
Boozer angles, with omega in rad/s, amplitude in volts and full-torus integer n.
Its ideal relation is ``alpha=(n-m*iota)*Phi/[omega*(G+iota*I)]``.
The default radial factor ``s**(m/2)`` is smooth on axis; an optional Gaussian
localizes it. ``AlfvenSpectrum`` sums modes. Custom potentials implement
``BoozerPerturbation.potentials(field,regular_position,t)``.

``AnalyticalVmecAlfvenMode`` instead specifies a smooth ideal wave in VMEC
angles. An identical Boozer-angle wave in VMEC requires
``MappedBoozerPerturbation`` and ``VmecBoozerMap.from_booz_xform`` from the same
stellarator-symmetric transform. Mapping derivatives include the chain rule.
Custom Cartesian/VMEC potentials derive from ``GuidingCenterPerturbation``.

Native orbit and collision stepping
----------------------------------

No separate wave integrator is added. ``trace_boozer(...,perturbation=wave)``
and ``Tracing(...,integrator="essos_dopri8",perturbation=wave)`` reuse the
native RK stages, error control, dense saves and event detection. The latter
uses physical ``mu`` as a state component instead of fixing birth energy.
``essos_rk45`` selects native Dopri5. Compatible Boozer/VMEC GPU orbit adapters
retain the private field evaluators and evaluate each stage at its physical
time. Real analytical potential derivatives avoid repeated automatic
differentiation; arbitrary compatible potentials retain the AD fallback.

For native collisions, given a field, particles and ``BackgroundSpecies``:

.. code-block:: python

   from essos.split import trace_split

   result = trace_split(field, particles, bath,
       tmax=1e-4, timestep=1e-7, n_save=101,
       perturbation=wave, integrator="essos_dopri8",
       adaptive=True, orbit_adaptive=True,
       adjoint="accepted")

The default moment controller uses the perturbed RHS, including its actual
time and speed change. Collision kicks use current kinetic speed/pitch and
update physical mu; their coefficients and spatial diffusion use the static
background B0, consistently with this reduced model. Spatial diffusion remains
enabled by default. Boozer requires the base solver's Cartesian
``spatial_geometry`` embedding, or an explicit ``spatial_diffusion=False``
for a velocity-space-only diagnostic.

``execution="independent"`` retains the persistent GPU forward loop and
private field adapters. Wave reverse uses the checkpointed JAX implementation
on the selected device; the specialized unperturbed GPU reverse is not used.
Accepted-map AD freezes controller/rejection/event decisions while
differentiating the accepted orbit and collision maps, including wave
parameters. It does not differentiate a discontinuous hard loss count.

``regularize_axis=False`` reproduces a radially uniform paper harmonic, which
is singular at the axis for m>0. Nonzero waves stop at ``axis_stop_s=1e-8``
by default; smooth modes and zero amplitude require no inner cutoff.
Collisionless Boozer tracing reports ``axis_stop_times`` separately from
LCFS exits and failures. Split tracing reports ``SplitStatus.EVENT`` and its
final public flux distinguishes the inner cutoff from the LCFS. Event states
are held for later saves. Cutoff sensitivity remains a convergence check.

Verification and reproducible benchmarks
---------------------------------------

The 19 focused tests cover ideal fields, induction, wave/action work,
phase-space conservation, unchanged zero-wave dynamics, coordinate mapping,
axis events, native stage times, GPU-kernel CPU interpretation and accepted
wave-parameter gradients through collision kicks. They use small CPU cases.

.. code-block:: bash

   JAX_PLATFORMS=cpu python -B -m pytest -q -p no:cacheprovider tests/test_perturbations.py
   python -m examples.particle_tracing.vmec_alfven_cpu_smoke
   python -m examples.benchmarks.alfven_gc_verification

The four-trajectory VMEC smoke example checks constant mu and permits physical
wave work. The analytic-equilibrium verification checks induction against an
exact trajectory, Liouville conservation and wave-frame exchange identities.
It is inspired by ORBIT conservation tests, not a reproduction of their
JT-60U figures. Both scripts have a 120-second wall limit.

The Paul drivers require ``booz_xform_jax`` for the coordinate transform.
The benchmark retains the published Table 2 QA endpoints in TOML input
files and traces shared marker proposals in Boozer and VMEC, each with/without
the mapped wave. It reports physical-volume weights, LCFS exits, inner stops,
unresolved markers and sampling intervals. All four runs use native Dopri8.
Prepare the commands without launching a simulation:

.. code-block:: bash

   python -m examples.benchmarks.paul_boozer_vmec_benchmark --dry-run \
     --particles 100 --platform cpu --wall-seconds 600

Remove ``--dry-run`` to execute. ``--wout`` selects a common QA input.
Boozer is QS-filtered while VMEC retains the full equilibrium; these are
different physical backgrounds. The exact published equilibrium/runtime
remains unconfirmed. Agreement of a small loss sample alone does not validate
it, and no old result or timing is presented as a measurement of this checkout.

All added scripts default to ``../artifacts/analytical_alfven/`` outside the
checkout, accept ``--output``, and generate plots plus diagnostics when run.
Saved results can be redrawn without tracing using their report-only options,
``paul_boozer_vmec_loss_comparison``, or
``plot_alfven_results --kind {smoke,verification,loss} --data <result-folder>``.
No generated benchmark arrays or images are included in the branch.
