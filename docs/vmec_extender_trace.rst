Tracing VMEC Exterior Fields
============================

``essos.vmec_extender.VmecExtendedField`` adapts a
``virtual_casing_jax.VirtualCasingExteriorField`` to the field interface used by
ESSOS tracing routines. It exposes Cartesian ``B``, ``B_covariant``,
``B_contravariant``, ``AbsB``, ``dB_by_dX``, and ``curl_B`` methods.

Python example
--------------

.. code-block:: python

   import vmec_jax
   from virtual_casing_jax import ExteriorFieldConfig
   from essos.vmec_extender import build_vmec_extended_field
   from essos.dynamics import Tracing

   run = vmec_jax.run_fixed_boundary("input.vmec")
   field = build_vmec_extended_field(
       vmec_state=run.state,
       vmec_static=run.static,
       indata=run.indata,
       coil_field=coil_field,
       config=ExteriorFieldConfig(digits=8),
   )

   tracing = Tracing(
       field=field,
       model="FieldLineAdaptative",
       initial_conditions=initial_xyz,
       maxtime=1000,
       times_to_trace=6000,
   )
   tracing.poincare_plot()

Limitations
-----------

Poincare plots and hard wall-hit events are diagnostics. They should not be
presented as differentiable objectives unless a smooth surrogate is explicitly
implemented and tested.
