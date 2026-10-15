Thermal-helium core residence
============================

Install ``pip install 'essos[ash]'`` for the optional SOLVAX dependency.
``essos.ash`` accepts explicit local trace-helium transport coefficients,
physical cell volumes and a source. It solves the conservative steady equation
``K n = M S`` and its transpose ``K.T T = M c``. The source-weighted occupation
is ``(M c).T n / sum(M S) = (M S).T T / sum(M S)``. For ``c=1`` this is mean
core residence; weighted occupation can describe a frozen core-dilution cost.
This standard forward/backward identity is exact for the stated discrete model.

Use ``jax.config.update('jax_enable_x64', True)`` before preparing arrays for the
strict residual and balance tolerances. Face arrays D [m^2/s], V [m/s] and
W=dV/dr [m^2] have n+1 entries; cell volumes [m^3], source [m^-3/s] and optional
dimensionless cost have n. The first radial edge is zero and reflects.
``boundary_kappa_m_s`` is mandatory: positive finite values impose
``Gamma=kappa*n`` and positive infinity absorbs. The half-cell transport and
boundary resistances are both included. A closed boundary, nonpositive D,
invalid values, unresolved inward barriers or unqualified solves yield
``valid=False`` and NaN physical outputs. Shape errors raise ``ValueError``.
``check_helium_residence(result)`` raises on an invalid result outside JIT.

The functions retain native JAX arrays for JIT and differentiation through all
provided coefficients, geometry measures, source, cost and boundary inputs.
File loading is host-only. Differentiation cannot recover dependence omitted
by a coefficient provider, including a frozen electric field or ambipolar root.
Pivot and residual checks do not provide a condition-number error bound;
refine the radial mesh and kinetic response and test the local transport closure.
No artificial diffusivity floor is applied.

Run the manufactured numerical check::

    python examples/particle_tracing/helium_ash_removal.py --manufactured

For an actual equilibrium use an explicit exported response::

    python examples/particle_tracing/helium_ash_removal.py \
      --wout path/to/wout.nc --coefficients path/to/helium_response.json

The JSON has ``coefficients`` matching the evaluator's argument names and
``metadata`` describing ``equilibrium_sha256``, ``profiles_id``, ``source_id``,
``collision_model``, ``electric_field`` (prescribed profile or root branch),
``turbulent_closure``, ``boundary_model``, ``volume_origin``,
``coefficient_status`` and ``trace_validity``. JSON null kappa means absorbing.
The example checks the WOUT hash and radius, reports assumptions and numerical
diagnostics, and does not infer missing transport physics. Volumes and W must
be obtained consistently from that equilibrium; a hash alone cannot verify
quadrature. Do not extrapolate an unmodeled outer region. A direct optional
DKX calculation can export this same bundle; DKX is never imported by this
module and full kinetic preparation/qualification remains the provider's work.

``trace_response`` tests at least three distinct density-gradient probes of
``Gamma/n=V-D*d(log n)/dr``. It flags non-affine or nonpositive-D responses.
``convection_from_marker_drift`` implements ``V=A-D'-D*d(log W)/dr`` away from
the axis. A Maxwellian-bath orbit response is not automatically a full-FP
trace-impurity response; match collision physics before comparing coefficients.

``restricted_residence`` uses a common positive horizon and +infinity for
physical survivors. It returns a weighted empirical survival integral,
escaped weight, effective sample size and an independent-marker standard error.
The restricted mean does not estimate an unobserved survival tail. Any numerical
failure invalidates the estimate rather than becoming a confined/censored
particle. Hard exit events are validation observables, not smooth optimization
objectives. Alpha energy retention, wall interception and pumped exhaust require
separate models and diagnostics; core exit does not establish pumping.
