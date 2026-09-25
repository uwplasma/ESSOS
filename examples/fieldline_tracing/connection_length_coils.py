import os
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.coils import Coils
from essos.dynamics import connection_length

# Connection length and strike points outside the Landreman-Paul QA plasma,
# with a circular-cross-section toroidal wall of minor radius a_wall.
R_axis, a_wall, max_length = 1.0, 0.35, 200.0
R0 = jnp.linspace(1.20, 1.33, 40)
json_file = os.path.join(os.path.dirname(__file__), '..', 'input_files', 'ESSOS_biot_savart_LandremanPaulQA.json')
field = BiotSavart(Coils.from_json(json_file))


def wall(xyz):
    return a_wall - jnp.hypot(jnp.hypot(xyz[0], xyz[1]) - R_axis, xyz[2])


seeds = jnp.stack([R0, jnp.zeros_like(R0), jnp.zeros_like(R0)], axis=1)
time0 = time()
result = connection_length(field, seeds, wall, max_length=max_length)
print(f"Connection lengths took {time() - time0:.1f} s; "
      f"{int(result['hit'].all(axis=1).sum())}/{len(R0)} lines hit the wall both ways")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(R0, result["connection_length"], "o-")
ax1.set_xlabel("R at phi=0, Z=0 [m]")
ax1.set_ylabel(f"L_c [m] (capped at {2 * max_length:.0f})")
points = result["strike_points"][result["hit"]]
ax2.scatter(jnp.degrees(jnp.arctan2(points[:, 1], points[:, 0])), points[:, 2], s=8)
ax2.set_xlabel("strike toroidal angle [deg]")
ax2.set_ylabel("strike Z [m]")
plt.tight_layout()
plt.show()
