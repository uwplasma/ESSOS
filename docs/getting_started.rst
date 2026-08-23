Getting started
=====

.. _installation:

Installation
------------

To use ESSOS, there is no need to install it.
You can simply clone the repository and install it using the following command:

.. code-block:: console

    $ git clone https://github.com/uwplasma/ESSOS.git
    $ cd ESSOS
    $ pip install .

Run an example
--------------

To run the one of the examples, use the following command:

.. code-block:: console

    python examples/trace_fieldlines_coils.py

More examples are in the `examples` folder.

VMEC equilibria with VMEX
-------------------------

`VMEX <https://github.com/uwplasma/VMEX>`_ (``pip install vmex``) solves the
VMEC equilibrium ESSOS traces through, and neither package imports the other:
coils leave ESSOS as a Biot-Savart field tabulated onto a cylindrical grid,
which VMEX's free-boundary solver consumes, and the equilibrium comes back as
a wout file, which ``essos.fields.Vmec`` reads.

.. code-block:: console

    python examples/simple_examples/equilibrium_from_coils_vmex.py
