FAX modes in ESSOS
=================

This checkout extends the prepared Alfvén/native-AD snapshot. It adds import
and reconstruction modules, reuses the existing GC corrections and native
steppers, and leaves the original worktrees in place. The FAX operator itself
is supplied by a separate FAX installation; it is not copied into ESSOS.

``FaxMode.from_result(problem,result,index,equilibrium=path,
interpretation="potential_shape")`` checks a positive eigenvalue, the
frequency/omega-squared conversion and eigenpair residual. A saved packet
preserves complex coefficients, full-torus harmonic labels, metadata and the
equilibrium SHA256; loading checks that same equilibrium. Producers must still
use the stated ESSOS Boozer coordinate convention.

``packet.as_physical_potential(potential_volts=V,frequency_sign=1)`` reconstructs
``Phi=Re sum C_mn(s) exp(i*(m*theta-n*zeta+omega*t))``. Natural cubic splines
interpolate the real and imaginary coefficients, retaining their relative
phases. ``alpha=(n-m*iota)*Phi/[omega*(G+iota*I)]`` is applied per harmonic.
Thus both magnetic flutter and electric/inductive terms enter the existing
GC RHS. The background magnitude remains in the reduced Hamiltonian.

The eigenproblem determines relative profiles and frequency, not saturated
mode amplitude. ``potential_volts`` or a magnetic-field normalization is an
explicit input. The optional ``as_perturbation`` path interpolates regular-axis
envelopes after removing ``s**(|m|/2)``; annular samples require an explicit C2
taper. That is a different extension policy from physical-profile interpolation.

``MappedBoozerPerturbation`` transfers a wave to native VMEC/Cartesian
coordinates through a differentiable map. ``perturbed_fields``,
``physical_components`` and ``radial_field_rms`` expose field diagnostics.
The base ``trace_split(...,perturbation=wave,adjoint="accepted")`` interface
also accepts reconstructed modes and differentiates their accepted maps.
Packet import/export and spline setup use NumPy outside JIT. Differentiation
acts on the resulting JAX wave parameters and tracing maps; a file round trip
does not carry a gradient back through an upstream FAX eigensolve.
The native table selector has an explicit gather derivative rule: its
bit-preserving value operation otherwise drops radial profile derivatives
when computed potentials are differentiated inside the GPU adapter. A CPU
interpretation test compares persistent-kernel spline orbits with the portable
solver and checks first/second derivatives of the selector.

Small independent verification
------------------------------

Install/use FAX with ``fax.vorticity`` available (the prepared
``fax-physical-potential`` checkout supplies this API). No installation or
download is needed to read an existing ESSOS mode packet.

.. code-block:: bash

   JAX_PLATFORMS=cpu python -B -m pytest -q -p no:cacheprovider \
     tests/test_fax_adapter.py tests/test_mode_reconstruction.py
   python -m examples.benchmarks.fax_shear_alfven_benchmark

The bridge tests include a small full physical-potential FAX matrix solve
against independent uniform-plasma dispersion, interchange and native tracing
of two markers with different energies. They also check phase preservation,
AE3D angular-frequency units, spline values, Maxwell identities, a smooth axis,
and a reconstructed-wave amplitude gradient through collision kicks. Live FAX
checks skip when the optional API is unavailable.

The separate shear-Alfvén script solves small constant FAX radial blocks,
reconstructs the wave and compares native adaptive GC trajectories and first
exits against closed-form results. Its absorbing slab wall is an imposed
diagnostic boundary, not a reactor loss model. It has a 120-second CPU cap.

QA calculation and magnetic sections
------------------------------------

The archive helper retrieves only pinned ZIP members with explicit
``--download`` or reads a local ZIP via ``--archive``. No example downloads
inputs automatically. All paths below default outside the checkout.

.. code-block:: bash

   python -m examples.benchmarks.ae3d_firm3d_archive --download
   python -m examples.fieldline_tracing.fax_qa_poincare

The second command builds full FAX physical-potential matrices on the archived
QA equilibrium using ``fax.vorticity``, complex harmonics and regular-axis
boundary conditions, solves them, imports one eigenmode and plots equilibrium
and perturbed sections. ``--radial-points``, ``--angular-points``, ``--lines``,
``--turns`` and ``--steps-per-turn`` control resolution; ``--wall-seconds``
bounds the process. Defaults are a coarse CPU demonstration. Target-frequency
mode selection is explicit and does not fit a published eigenfrequency.
Field lines use frozen-time RK4 in toroidal angle; particle trajectories use
the native adaptive time integrator.

``examples.particle_tracing.fax_mode_particles --compute
--interpretation potential_shape --frequency-sign 1`` instead computes a small
mode from a selected ``--wout`` and maps it into VMEC for two short native GC
trajectories. ``--mode`` reuses an exported packet without a new FAX solve.

``examples.benchmarks.fax_ae3d_reconstruction --compute`` compares a native
physical-potential solve against archived AE3D profiles, retaining both
frequency and globally aligned profile errors. Its default reconstruction
control imports identical coefficients/frequency and tests only the bridge.
``examples.fieldline_tracing.compare_fax_qa_poincare`` compares the new QA FAX
packet with the archived AE3D potential at a common magnetic-energy
normalization. ``--fax-mode FILE --ae3d-mode FILE`` instead compares supplied
matrix-solution packets. AE3D-formula reconstructions must be labelled
as such; they are not original supplied AE3D matrices. Full native QA mode
convergence and the earlier operator/index discrepancy remain upstream
validation questions; reconstruction agreement alone does not resolve them.

Native particle and collision comparisons
----------------------------------------

The frozen 35-harmonic mode and its hash-checked normalization metadata are
versioned in ``examples/benchmarks/references/fax_qa_35/`` and loaded by default.
A fresh checkout needs no mode ZIP, file transfer or FAX installation for tracing.
The archive command below retrieves the required equilibrium separately.
The default study uses 8192 generated births, a ``1e-4`` equilibrium cutoff
(25 retained magnetic harmonics for this QA equilibrium), and all six cases:

.. code-block:: bash

   python -m examples.benchmarks.ae3d_firm3d_archive --download
   XLA_PYTHON_CLIENT_PREALLOCATE=false python -u -m examples.benchmarks.fax_qa_loss_collisions --run \
     --particles 8192 --birth-source volume --seed 42 --case all --tmax 0.01 \
     --platform cuda --execution independent --representation fourier \
     --mode-tolerance 1e-4 --moment-tolerance 0.01 --collision-tolerance 1e-3 \
     --tolerance 1e-9 --collision-dtmax 1e-6 --wall-seconds 21600 \
     --output ../artifacts/fax_integration/qa_8192_six_cases_gpu

The cases are unperturbed/perturbed collisionless, unperturbed/perturbed with
speed/pitch collisions, and unperturbed/perturbed with those collisions plus
spatial diffusion. All use native adaptive Dopri8 and the base split collision
controller; the initial timestep is not fixed. CUDA defaults to independent
execution, and CPU defaults to shared execution. Individual ``--case`` values
allow separate jobs; the two added cases end in ``_collisions_spatial``.
The wave voltage and initial phase come from the saved magnetic comparison.
The bath defaults to electrons/deuterium, each at ``4.21e20 m^-3`` and
``17.8 keV``. Spatial cases read the same archive's Cartesian surface
embedding. Its toroidal shift is converted from ``phi=zeta+p`` to ESSOS's
``phi=zeta-nu`` convention. ``--geometry-mode-tolerance`` applies a relative
cutoff independently to R, Z and toroidal-shift amplitudes. The default ``1e-6``
retains 345 geometry harmonics for this QA case; ``0`` retains all 4489 nonzero
modes. This is separate from the ``1e-4`` magnetic cutoff (25 harmonics) and
the 35-harmonic wave. A 1280-point audit against the full geometry found a
maximum 56 micrometre position error and 0.069% local inverse-map Jacobian error
at the default geometry cutoff; this is not a loss-convergence measurement.
Native geometry callbacks share exact value/Jacobian evaluations. Spatial
Fourier sums use blocks of at most 128 harmonics, bounding compiler intermediates
even with the full spectrum. Spatial kicks use the existing
Cartesian displacement and nonlinear coordinate inversion in ``trace_split``.

The generated proposal points are uniform in flux coordinates, with physical
Boozer-volume weights and isotropic pitch at 3.52 MeV. They are a new source,
not the archive's centrally concentrated fusion-birth population. Particle
proposals and collision keys have stable prefixes as count increases. All six
cases share the same births and weights. ``--birth-s-min``/``--birth-s-max``
restrict the source region. The archive option retains its 5000-marker limit;
generated cohorts support 8192 markers. To reuse archived births, select
``--birth-source archive --particles 5000`` or a smaller archive prefix.

The moment budget, collision-kick tolerance and orbit tolerance are separate.
``--collision-dtmax`` bounds adaptive collision intervals, while ``--timestep``
sets the initial guess. Native independent execution uses direct Fourier
equilibrium fields; the optional angular table representation requires shared
execution. ``--mode-tolerance`` controls equilibrium harmonic truncation
(default ``1e-4``); it leaves the saved FAX wave unchanged. The summary records
the cutoff and retained equilibrium count. Check field and orbit convergence
before relaxing this cutoff for production. Collisionless cases skip the
collision moment predictor while retaining adaptive orbit control. The
unperturbed collisionless case uses the dedicated Boozer orbit kernel,
without constructing split collision maps. Its common minimum birth-speed
normalization is at least as strict as the per-marker orbit budget.
``--timing-repeats 3`` records its compilation, first execution, and three
blocked warm execution times separately. Wave tables use indexed vector
kernel reads, preserving the existing spline and potential formulas. Radial
rows and individual harmonics are not expanded into scalar selection chains.
Every case records solver compilation and first blocked execution separately;
``wall_seconds`` additionally includes preparation, post-processing and saving.
The native moment predictor scalarizes speed and pitch before differentiation,
avoiding higher spatial field derivatives multiplied by zero velocity tangents.
It retains the same first derivatives, diagonal Hessian and moment budget.
Plotting writes a bold-font ``fax_qa_loss_fraction.png`` at 300 dpi,
plus SVG/PDF, an accompanying loss/energy figure, and ``loss_fraction.csv``
(time in seconds and volume-weighted fractions from zero to one). Each
completed case saves marker indices, weights, states and termination times in
its own NPZ before the next case starts. Regenerate figures and CSV, including
from partial runs, with ``--report-only --output PATH``. Missing cases are not
treated as completed.
After interruption, rerun the same command and output directory with
``--resume``. Births, wave coefficients and physical settings are validated
before completed cases are reused. A geometry cutoff can change while only
non-spatial cases have completed; completed spatial cases require its original
value. Axis-stopped or failed markers retain their unresolved classification.
Shared execution is still the native adaptive solver on the selected GPU;
independent execution uses the vectorized table adapter and shared exact
value/Jacobian evaluations. The blocked full-spectrum geometry callback passes
GPU value/Jacobian parity, but the complete spatial-diffusion CUDA solver still
exceeds the bounded compilation checks. Production spatial loss runs remain
unvalidated; the isolated callback result does not establish solver performance.

``examples.benchmarks.ae3d_firm3d_loss --run --particles 5000
--case both --platform cuda --execution independent`` separately traces the
archived AE3D potential, not a newly calculated FAX mode. Natural cubic is
the default; ``--interpolation linear`` recovers the archived radial policy.
See :doc:`ae3d_firm3d_benchmark` for comparison limits. These ensemble commands
are prepared examples and have not been run as part of this port.

``--report-only`` redraws completed outputs without tracing. Generated mode
packets, arrays, summaries and figures live under
``../artifacts/fax_integration/`` by default; ``--output`` overrides that path.
Only input specifications and source code are retained in this checkout.
