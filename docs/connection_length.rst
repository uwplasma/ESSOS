Connection Length
=================

Connection-length and wall-hit workflows should use the VMEC extender field as
an ESSOS field object and keep discontinuous events separate from smooth
optimization objectives.

Planned validation checks:

* compare Poincare point clouds against exported-grid field-line tools;
* compare connection lengths against FIELDLINES/TORLINES/FLARE where available;
* report interpolation and ODE tolerances with every comparison;
* keep deterministic seed sets for CPU CI smoke tests.

This page is a placeholder for the connection-length gallery that will be
populated once matched benchmark cases are added.
