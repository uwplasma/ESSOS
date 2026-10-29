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

The ten bridge tests include a small full physical-potential FAX matrix solve
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

After generating ``fax_qa_poincare/mode.npz`` and its summary:

.. code-block:: bash

   python -m examples.benchmarks.fax_qa_loss_collisions --run \
     --particles 512 --case all --tmax 0.01 --platform cuda \
     --execution independent --wall-seconds 21600

This compares the same archived births with/without the native FAX wave and
with/without collisions, using native adaptive Dopri8 and the base split
collision controller. The initial timestep is not fixed. Every marker retains
its archived parallel velocity and mu/m, so its birth energy may differ
slightly from the nominal monoenergetic value after field reconstruction.
The wave voltage and initial phase come from the saved magnetic comparison.
The bath defaults to electrons/deuterium, each at ``4.21e20 m^-3`` and
``17.8 keV``. This comparison explicitly disables spatial collision diffusion;
it isolates speed/pitch scattering. Use the base ``trace_split`` API with a
Cartesian geometry embedding for spatial diffusion studies.

For a generated cohort on a larger Linux GPU machine:

.. code-block:: bash

   XLA_PYTHON_CLIENT_PREALLOCATE=false python -u -m examples.benchmarks.fax_qa_loss_collisions --run \
     --particles 2000 --birth-source volume --seed 42 --case all --tmax 0.01 \
     --platform cuda --execution shared --representation fourier \
     --moment-tolerance 0.01 --collision-tolerance 1e-3 --tolerance 1e-9 \
     --mode ../artifacts/fax_integration/qa_saved_mode/mode.npz \
     --wave-summary ../artifacts/fax_integration/qa_saved_mode/wave_summary.json \
     --wall-seconds 21600 --output ../artifacts/fax_integration/qa_2000_four_cases_gpu

The generated proposal points are uniform in flux coordinates, with physical
Boozer-volume weights and isotropic pitch at 3.52 MeV. They are a new source,
not the archive's centrally concentrated fusion-birth population. Particle
proposals and collision keys have stable prefixes as count increases. All four
cases share the same births and weights. ``--birth-s-min``/``--birth-s-max``
restrict the source region. The archive option retains its 5000-marker limit;
generated cohorts may use 8000 or more when requested.

The moment budget, collision-kick tolerance and orbit tolerance are separate.
``--collision-dtmax`` bounds adaptive collision intervals, while ``--timestep``
sets the initial guess. Native independent execution uses direct Fourier
equilibrium fields; the optional angular table representation requires shared
execution. Plotting writes a bold-font ``fax_qa_loss_fraction.png`` at 300 dpi,
plus SVG/PDF and an accompanying loss/energy figure. Report-only plotting
works on saved partial cases without treating missing cases as completed.
Shared execution is still the native adaptive solver on the selected GPU;
it avoids scalarizing a large tabulated mode into a persistent kernel. The
local 158-harmonic independent-kernel job was killed before a perturbed result
was saved, so its large-mode resource requirements remain unverified.

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
