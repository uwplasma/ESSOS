import os
from essos.dynamics import GuidingCenter, Lorentz, FieldLine
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
import jax.numpy as jnp
from essos.coils import Curves, Coils
from simsopt import load


output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
json_file = os.path.join(os.path.dirname(__file__), 'input', 'biot_savart_opt.json')
bs = load(json_file)

