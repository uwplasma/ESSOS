Near-axis external field jets
=============================

ESSOS consumes the surface-free field-jet targets produced by pyQSC_JAX.
The dependency direction is one way: pyQSC_JAX computes the near-axis
equilibrium and ESSOS evaluates the coil field.

Target construction
-------------------

A vacuum target uses the total near-axis field:

.. code-block:: python

   import pyqsc_jax as qsc
   from essos.field_jet import near_axis_field_jet_target

   solution = qsc.Qsc(
       rc=[1.0, 0.155, 0.0102],
       zs=[0.0, 0.154, 0.0111],
       nfp=2,
       etabar=0.64,
       B2c=-0.00322,
       order="r2",
   )
   target = near_axis_field_jet_target(solution)

For finite plasma pressure or current, provide the formal minor radius. ESSOS
then matches coils to the external vacuum field, not to the total MHD field.
The user-facing finite-beta example is a pressure-only stellarator with
nonzero torsion, not a finite-current tokamak-like channel:

.. code-block:: python

   solution = qsc.solve_configuration("plasma_stellarator", nphi=61)
   assert solution.inputs.I2 == 0
   assert solution.inputs.p2 != 0
   target = near_axis_field_jet_target(
       solution,
       formal_radius=0.15,
   )

The radius fixes the asymptotic current/flux normalization; no finite-radius
surface is constructed. The legacy
``pyqsc_jax.near_axis.near_axis`` adapter is also accepted.

Normalized objective
--------------------

``normalized_field_jet_residuals`` evaluates

.. math::

   r_0 &= (B_{\mathrm{coils}}-B_c)/B_{\mathrm{ref}},\\
   r_1 &= L_{\mathrm{ref}}\,
          \operatorname{STF}_2(\nabla B_{\mathrm{coils}}-D_c)
          /B_{\mathrm{ref}},\\
   r_2 &= L_{\mathrm{ref}}^2\,
          \operatorname{STF}_3(\nabla\nabla B_{\mathrm{coils}}-H_c)
          /B_{\mathrm{ref}}.

The returned blocks have 3, 5, and 7 components per axis point.
``loss_field_jet_coils_near_axis`` forms a weighted sum of their mean squares.
It is smooth and differentiable with respect to coil shapes, coil currents,
axis Fourier coefficients, and near-axis physics inputs.

ESSOS obtains the coil Hessian from ``MagneticField.d2B_by_dXdX`` using nested
JAX differentiation. pyQSC_JAX owns the STF projection and component ordering,
so the tensor convention is not duplicated between packages.

Validation provenance
---------------------

The initial integration was developed against:

* pyQSC_JAX commit ``c7277fc``;
* ESSOS base commit ``240fe64``; and
* the merged near-axis extraction PR commit ``448356f``.

These identifiers document the tested state and are not runtime dependency
pins.

The focused integration suite additionally checks that the pressure-only
database case has ``|iota| > 0.4``, RMS torsion above
``0.5 m^-1``, angle-dependent plasma field, zero enclosed current, and
nonzero field/Hessian subtraction. Both direct optimization scripts are run
from a clean working directory.
