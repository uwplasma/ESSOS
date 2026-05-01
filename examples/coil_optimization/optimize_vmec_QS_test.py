import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.losses import custom_loss
from essos.mhd import VmecJAXBoundary

#  In this exmple, `scipy.optimize.least_squares` is used, but any other optimizer, e.g. from 
#  `scipy.optimize.minimize` or `jaxopt`, can be used as well and may even be preferable.
from scipy.optimize import least_squares

input_filepath = os.path.join(os.path.dirname(__name__), "input_files")
vmec_input = os.path.join(input_filepath, 'input.QA_test')

vmec=VmecJAXBoundary.from_vmec_input(vmec_input, performance_mode=True, solver_mode="default", jit_forces="auto")
vmec.iota()
