Thermal-helium core residence
============================

Install ``pip install 'essos[ash]'`` and enable JAX float64. ``essos.ash``
solves the local trace-helium closure ``Gamma=-D*dn/dr+V*n`` using supplied
transport coefficients, geometry measures, a source and a boundary condition.
The forward and backward equations are ``K n = M S`` and ``K.T T = M c``.
Source-weighted occupation is ``(M c).T n / sum(M S) = (M S).T T / sum(M S)``;
``c=1`` gives mean core residence. Recycling and pumping require separate models.

Inputs and checks
-----------------

Face arrays D [m^2/s], V [m/s] and W=dV/dr [m^2] have n+1 entries. Cell volumes
[m^3], source [m^-3/s] and optional dimensionless cost have n entries. The radial
mesh starts at zero with a reflecting axis. Positive finite ``boundary_kappa_m_s``
imposes ``Gamma=kappa*n``, including half-cell resistance; +infinity absorbs.
D must be positive. Volumes and W must use the same physical radial coordinate.

Invalid inputs or failed discrete residual, positivity, duality or balance checks
return ``valid=False`` and NaN physical outputs. Shape errors raise ``ValueError``;
``check_helium_residence`` raises on an invalid result outside JIT. These checks
require float64 and do not establish radial convergence or kinetic validity.
Gradients include supplied coefficient, geometry, source, cost and boundary
dependence; omitted responses, such as an ambipolar root, remain omitted.

Example
-------

Run the manufactured cylinder or supply an exported kinetic response::

    python examples/particle_tracing/helium_ash_removal.py --manufactured
    python examples/particle_tracing/helium_ash_removal.py \
      --wout path/to/wout.nc --coefficients path/to/helium_response.json

The JSON contains ``coefficients`` matching the evaluator arguments and
``metadata`` with ``equilibrium_sha256``, ``profiles_id``, ``source_id``,
``collision_model``, ``electric_field``, ``turbulent_closure``, ``boundary_model``,
``volume_origin``, ``coefficient_status`` and ``trace_validity``. Null kappa means
absorbing. The example checks the WOUT hash and modeled radius. The provider
must supply consistent volumes, profiles and coefficients over that domain;
a hash does not verify quadrature, and WOUT alone cannot supply transport.

Other diagnostics
-----------------

``trace_response`` fits ``Gamma/n=V-D*d(log n)/dr`` using at least three distinct
probes with matched backgrounds and electric field. ``convection_from_marker_drift``
uses ``V=A-D'-D*d(log W)/dr`` away from the singular axis.

``restricted_residence`` estimates ``E[min(T,H)]`` at a common positive horizon,
using +infinity for survivors. Any numerical failure invalidates the estimate.
Its standard error assumes independent markers with fixed weights; the
unobserved tail and correlated-marker uncertainty are not estimated.
