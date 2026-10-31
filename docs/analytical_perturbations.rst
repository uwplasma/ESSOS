Prescribed Alfvén perturbations
===============================

ESSOS follows collisionless guiding centres in a static equilibrium with imposed,
time-dependent shear-Alfvén potentials, in Boozer coordinates (``trace_boozer``) or in
the native coordinates of any field such as VMEC (``Tracing``). The wave is prescribed:
there is no growth, saturation or response to the particles.

Equations
---------

The reduced model uses :math:`\delta\mathbf A=\alpha\mathbf B_0` and a potential
:math:`\Phi`, so :math:`\delta\mathbf B=\nabla\times(\alpha\mathbf B_0)` and
:math:`\delta\mathbf E=-\nabla\Phi-\partial_t\alpha\,\mathbf B_0` (``alpha`` in m,
``Phi`` in V, frequencies in rad/s). The Hamiltonian keeps the equilibrium magnitude,

.. math::

   H=\tfrac12 Mv_\parallel^2+\mu B_0+q\Phi,\qquad \dot\mu=0,

and with :math:`\mathbf B^*=\mathbf B_0+\tfrac{Mv_\parallel}{q}\nabla\times\mathbf b_0
+\delta\mathbf B`, :math:`\mathbf E^*=\mathbf E_0+\delta\mathbf E-\tfrac{\mu}{q}\nabla B_0`,

.. math::

   \dot{\mathbf R}=\frac{v_\parallel\mathbf B^*+\mathbf E^*\times\mathbf b_0}{\mathbf b_0\cdot\mathbf B^*},\qquad
   \dot v_\parallel=\frac{q}{M}\frac{\mathbf B^*\cdot\mathbf E^*}{\mathbf b_0\cdot\mathbf B^*}.

In the existing curvature-drift form the corrections are magnetic flutter, inductive
acceleration and the parallel-force term that cancels the extra curvature contraction
with :math:`\delta\mathbf B`. Every Runge-Kutta stage uses its physical time. Kinetic energy
changes through wave work while :math:`\mu` is constant, so the perturbed native trace uses
the fixed-:math:`\mu` ``GuidingCenterMu`` state. The native equations keep all covariant
components of :math:`\mathbf B_0`, including VMEC's radial one; the Boozer model uses
:math:`\mathbf B_{\rm cov}=I\,d\theta+G\,d\zeta` and omits its radial component
(:math:`K=0`).

Modes
-----

``essos.perturbations.AnalyticalAlfvenMode`` has

.. math::

   \Phi=\hat\Phi f(s)\sin(m\theta_B-n\zeta_B+\omega t+\varphi),\qquad
   \alpha=\frac{n-m\iota}{\omega(G+\iota I)}\Phi,

which enforces :math:`\mathbf B_0\cdot\delta\mathbf E=0` (:math:`n` includes the field-period
factor). By default :math:`f=s^{m/2}`, smooth on axis; ``radial_center``/``radial_width`` give a
Gaussian instead. ``AlfvenSpectrum`` superposes modes. ``AnalyticalVmecAlfvenMode`` defines the
harmonic in VMEC angles, and ``MappedBoozerPerturbation`` with ``VmecBoozerMap`` (from
``booz_xform_jax``, stellarator symmetry) evaluates the identical Boozer wave in VMEC, with the
chain rule of the map. A new mode, including a computed one, only implements ``potentials``.

The radially uniform harmonic of Paul et al. (``regularize_axis=False``) is singular on axis for
:math:`m>0`. Tracing stops there at ``axis_stop_s`` (default :math:`10^{-8}`) and reports the stops
(``axis_stop_times``; status 6 in ``Tracing``) apart from LCFS exits and failures;
``loss_fractions(include_axis=True)`` counts both, as the paper does. Perturbed native tracing
supports collisionless guiding centres inside the equilibrium.

Verification
------------

``tests/test_perturbations.py`` and ``tests/test_native_gc_perturbations.py`` check the model
against independent laws (after `Bierwage, White and Matsuyama (2022)
<https://arxiv.org/abs/2205.11280>`_, on analytic fields): :math:`E_\parallel=0`; the
rotating-frame invariant :math:`H-(\omega/n)P_\zeta` for two coupled harmonics, with a control
showing it detects a missing inductive term; Liouville's theorem for the time-dependent
phase-space density; the exact trajectory in a uniform inductive field; the work identity
:math:`dH/dt=q\,\mathbf v\cdot\mathbf E` in a sheared nonorthogonal chart; coordinate
covariance; gradients with respect to the amplitude; and the zero-amplitude limit.

Paul alpha-loss comparison
--------------------------

``examples/particle_tracing/paul_alfven_benchmark.py`` follows `Paul, Mynick and
Bhattacharjee (2023), Table 2 <https://arxiv.org/html/2306.12531v2#S6.T2>`_: vacuum QA,
:math:`m=15`, :math:`n=7`, 399000 rad/s, edge :math:`m\hat\alpha/a=10^{-3}`, 3.52 MeV, 1 ms,
in a Boozer background without QS-breaking harmonics and in the full VMEC equilibrium
(scaled to :math:`a=1.7` m, :math:`B_{00}=5.86` T), with the same markers weighted to uniform
physical volume. It needs ``pip install -e '.[booz]'``:

.. code-block:: bash

   python examples/particle_tracing/paul_alfven_benchmark.py --particles 100 --historical --plot paul.png

``--historical`` applies the sign and alpha-mass conventions of the 2023 SIMSOPT adapter. The
table is this 100-marker run with the bundled ``wout_LandremanPaul2021_QA_lowres.nc`` (tens of
minutes on one CPU core); the unperturbed curves of the two backgrounds coincide in the figure.
Both waves lose 13 markers through the LCFS and eight at the inner stop.

.. list-table:: Weighted losses at 1 ms (100 markers; paper used 5000)
   :header-rows: 1

   * - Background
     - Case
     - ESSOS
     - Approx. 95% sampling interval
     - Paper
   * - Boozer / Perfect QS
     - Unperturbed
     - 12.25%
     - 7.16-20.16%
     - 6.6%
   * - Boozer / Perfect QS
     - Wave
     - 21.62%
     - 14.65-30.71%
     - 16.2%
   * - VMEC / full equilibrium
     - Unperturbed
     - 12.25%
     - 7.16-20.16%
     - 8.3%
   * - VMEC / full equilibrium
     - Wave
     - 21.56%
     - 14.59-30.65%
     - 19.4%

.. image:: _static/analytical_perturbations/paul_loss_100.png
   :alt: Loss curves for Boozer and VMEC backgrounds without and with the wave

Crosses mark the published endpoints. The exact published equilibrium and runtime are unknown, and
the intervals contain sampling noise only (no equilibrium-resolution or cutoff convergence). This is
a consistency check, not a quantitative reproduction.
