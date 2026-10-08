"""Guiding-center tracing through an interpolated coil + permanent-magnet field.

A stellarator-symmetric shell of point dipoles (a stand-in for a permanent-
magnet array) is added to the Landreman-Paul QA coils. Every evaluation of
the direct field sums over all coil segments and dipoles; the interpolated
field reads a 4x4x4 spline stencil, so tracing cost no longer grows with the
number of magnets. Usage: python trace_particles_interpolated_dipoles.py [ndipoles] [nparticles] [grid] [tmax]
"""
import os
import sys
from time import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util, vmap
from essos.coils import Coils
from essos.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, ONE_EV
from essos.dynamics import Particles, Tracing
from essos.fields import BiotSavart, CombinedField, InterpolatedField, MagneticField
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

args = sys.argv[1:] + ["4000", "32", "48", "2e-5"][len(sys.argv) - 1:]
ndipoles, nparticles, grid, tmax, energy = int(args[0]), int(args[1]), int(args[2]), float(args[3]), 5000 * ONE_EV
input_dir = os.path.join(os.path.dirname(__file__), '..', 'input_files')
coils = Coils.from_json(os.path.join(input_dir, 'ESSOS_biot_savart_LandremanPaulQA.json'))
wout = SurfaceRZFourier.from_wout_file(os.path.join(input_dir, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc'))
boundary = SurfaceRZFourier(wout.rc / wout.rc[0], wout.zs / wout.rc[0], wout.nfp, wout.mpol, wout.ntor, ntheta=64, nphi=64)
classifier = SurfaceClassifier(boundary, h=0.03)


@tree_util.register_pytree_node_class
class Dipoles(MagneticField):
    """Point dipoles: B = 1e-7 sum (3 r (m.r)/r^5 - m/r^3)."""
    def __init__(self, positions, moments):
        self.positions, self.moments = positions, moments

    def B(self, point):
        r = point - self.positions
        d = jnp.linalg.norm(r, axis=1, keepdims=True)
        return 1e-7 * jnp.sum(3 * r * jnp.sum(self.moments * r, 1, keepdims=True) / d**5 - self.moments / d**3, 0)

    def sqrtg(self, point):
        return 1.

    def tree_flatten(self):
        return (self.positions, self.moments), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# Magnets 0.15 m outside the plasma boundary, along its normal, copied to all 2*nfp symmetric images.
nfp, rng = coils.nfp, np.random.default_rng(0)
shell = SurfaceRZFourier(boundary.rc, boundary.zs, nfp, boundary.mpol, boundary.ntor, ntheta=256, nphi=256)
normal, pos = np.array(shell.unitnormal).reshape(-1, 3), np.array(shell.gamma).reshape(-1, 3)
normal *= np.sign(normal[0] @ pos[0])  # outward: the first point is on the outboard midplane
pos = pos + 0.15 * normal
ph = np.arctan2(pos[:, 1], pos[:, 0])
keep = rng.permutation(np.flatnonzero((ph > 0) & (ph < np.pi / nfp)))[:ndipoles // (2 * nfp)]
pos, normal = pos[keep], normal[keep]
mom = 2e3 / max(ndipoles, 1) * (normal + 0.3 * rng.normal(size=pos.shape))
# Stellarator symmetry: x -> (x, -y, -z), and the moment (like the coil current) m -> (-mx, my, mz).
pos, mom = np.r_[pos, pos * [1, -1, -1]], np.r_[mom, mom * [-1, 1, 1]]
angles = 2 * np.pi * np.arange(nfp) / nfp
rotations = [np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) for a in angles]
pos, mom = (np.concatenate([x @ r.T for r in rotations]) for x in (pos, mom))
field = CombinedField(BiotSavart(coils), Dipoles(jnp.array(pos), jnp.array(mom)))

time0 = time()
interp = InterpolatedField(field, R=(0.58, 1.33), Z=(-0.42, 0.42), nr=grid, nz=grid, nphi=2 * grid, nfp=nfp,
                           stellsym=True)
interp.coefficients.block_until_ready()
print(f"{len(pos)} dipoles; tabulated {grid}x{grid}x{grid + 1} nodes in {time() - time0:.2f} s "
      f"on {jax.devices()[0].platform}")

points = jnp.array(boundary.gamma.reshape(-1, 3)[::4])
for name, f in (("direct", field), ("interpolated", interp)):
    gc = jax.jit(vmap(f.gc_quantities))
    gc(points)[2].block_until_ready()
    time0 = time()
    gc(points)[2].block_until_ready()
    print(f"gc_quantities at {len(points)} points, {name}: {1e3 * (time() - time0):.2f} ms")
B0, B1 = vmap(field.B)(points), vmap(interp.B)(points)
print(f"max |B_interp - B|/|B| = {float(jnp.max(jnp.linalg.norm(B1 - B0, axis=1) / jnp.linalg.norm(B0, axis=1))):.2e}")

R0 = jnp.linspace(1.16, 1.26, nparticles)
particles = Particles(initial_xyz=jnp.stack([R0, 0 * R0, 0 * R0], 1), mass=ALPHA_PARTICLE_MASS,
                      charge=ALPHA_PARTICLE_CHARGE, energy=energy)
traces = {}
for name, f in (("direct", field), ("interpolated", interp)):
    time0 = time()
    traces[name] = tracing = Tracing(field=f, model='GuidingCenterAdaptative', particles=particles, maxtime=tmax,
                                     times_to_trace=200, atol=1e-8, rtol=1e-8,
                                     condition=lambda t, y, args, **kwargs: classifier.evaluate_xyz(y[:3]))
    tracing.trajectories.block_until_ready()
    tracing.lost_times = tracing.loss_fraction_BioSavart(classifier)[2]
    print(f"GC tracing of {nparticles} particles, {name}: {time() - time0:.2f} s, "
          f"loss fraction {float(jnp.mean(tracing.lost_times > 0)):.3f}")
lost = (traces["direct"].lost_times > 0) | (traces["interpolated"].lost_times > 0)
final = [traces[name].trajectories[~lost, -1, :3] for name in ("direct", "interpolated")]
deviation = jnp.linalg.norm(final[0] - final[1], axis=1)
print(f"final-position deviation of the {int((~lost).sum())} confined orbits after {tmax:.0e} s: "
      f"median {float(jnp.median(deviation)):.2e} m, max {float(jnp.max(deviation)):.2e} m")
