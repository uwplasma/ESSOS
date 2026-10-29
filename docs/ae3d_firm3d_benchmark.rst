Archived AE3D/FIRM3D loss comparison
==================================

``examples/benchmarks/references/ae3d_firm3d_qa.json`` specifies the public
Zenodo 21299908 packet, member offsets, CRCs and SHA256 digests. The archive
reader bounds every HTTP range, rejects a full-file response and verifies
each member before use. It does not execute archived scripts.

The selected QA reference has 5000 archived alpha markers and a 10 ms
duration. The published classification tables give 26 unperturbed LCFS losses
and 159 for ``QA_35harmonics_scale0_001``. The importer retains all 35 signed
potential profiles, harmonic labels and their supplied amplitude. Frequency
is ``sqrt(eigenvalue)*1000 rad/s``; no extra ``2*pi`` is introduced.

.. code-block:: bash

   python -m examples.benchmarks.ae3d_firm3d_archive --download
   python -m examples.benchmarks.ae3d_firm3d_loss
   python -m examples.benchmarks.ae3d_firm3d_loss --run --particles 512 \
     --case both --platform cuda --execution independent --wall-seconds 21600

Preparation checks the packet, birth fields and reference counts and plots
the inputs. Explicit ``--run`` enables native adaptive Dopri8 tracing.
``--particles 5000 --tmax 0.01`` selects the complete reference cohort;
a smaller prefix is a computational check, not the full loss benchmark.
Current birth energy is reconstructed from archived v_parallel and mu/m.
No particle is silently discarded after an unresolved numerical failure.

Natural cubic interpolation in s is used by default for smooth derivatives.
``--interpolation linear`` reproduces the archive's piecewise-linear
potential profiles. The two options solve different interpolated fields.
The ESSOS equilibrium interpolation also differs from archived FIRM3D;
the preparation report measures birth-field discrepancies. Tolerance,
interpolation and resolution convergence remain necessary for loss agreement.

This validates particle tracing/reconstruction from an archived eigenmode.
It does not validate FAX matrix construction. The full FAX calculation and
field comparisons are separate commands in :doc:`fax_modes`. Historical GPU
results were not copied or presented as measurements of the new checkout.
