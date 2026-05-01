VMEC Extender CLI
=================

ESSOS provides the ``essos-vmec-extender`` command for early VMEC exterior-field
workflows. The command builds an ESSOS-compatible field from ``vmec_jax`` and
``virtual_casing_jax`` and then runs validation, grid export, or field-line
tracing.

The preferred file-based route passes both a VMEC input and a matching wout
file:

.. code-block:: console

   essos-vmec-extender validate \
     --input input.vmec \
     --wout wout.nc \
     --coils coils.json \
     --src-nphi 64 --src-ntheta 64 \
     --digits 8 \
     --out results/validation.json

Validation metrics
------------------

``validate`` reports:

* VMEC surface ``B dot n`` mean, RMS, and max;
* internal/external branch identity on the LCFS;
* coil plus internal-branch normal-field cancellation when coils are supplied;
* external-branch versus coil normal-field parity when coils are supplied;
* requested and effective fixed-schedule levels;
* runtime and source-grid metadata.

The coil metrics are diagnostics unless the wout/coils pair is known to be a
matched free-boundary benchmark with validated units and sign conventions.

Grid export
-----------

.. code-block:: console

   essos-vmec-extender grid \
     --input input.vmec \
     --wout wout.nc \
     --coils coils.json \
     --R 0.8:2.0:128 \
     --phi 0:1.57079632679:64 \
     --Z=-0.8:0.8:128 \
     --out results/extended_field.nc

The output grid stores ``BR``, ``Bphi``, ``BZ``, and ``absB`` on an
``(R, physical phi, Z)`` tensor product grid. Metadata records the internal
virtual-casing sign convention:

.. code-block:: text

   B_total_out = B_coils + B_internal^VC

For ``NFP > 1``, the effective schedule levels may differ from requested
levels because ``virtual_casing_jax`` rounds toroidal schedule sizes up to a
multiple of ``NFP`` to preserve field-period covariance.

Trace
-----

.. code-block:: console

   essos-vmec-extender trace \
     --input input.vmec \
     --wout wout.nc \
     --coils coils.json \
     --seeds seeds.json \
     --nturns 200 \
     --phis 0,1.57079632679 \
     --out results/trace.npz \
     --plot results/poincare.pdf

Seed files may be JSON lists of Cartesian points:

.. code-block:: json

   [[1.8, 0.0, 0.0], [1.85, 0.0, 0.02]]

or dictionaries with ``xyz`` or ``R_phi_Z`` keys.

Benchmark Smoke
---------------

The script ``benchmarks/vmec_extender_cli_benchmark.py`` runs the bundled
low-resolution VMEC/coils example through ``validate``, ``grid``, and ``trace``
and writes a compact JSON report. It is meant to track CLI/runtime regressions
and validation metrics in a reproducible small case; matched free-boundary
physics benchmarks still require dedicated external-code comparisons.
