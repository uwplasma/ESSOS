"""Small forward split-collision example; no Diffrax SDE solve or plotting."""

import argparse

import jax.numpy as jnp
import numpy as np

from essos.background_species import BackgroundSpecies
from essos.constants import ELEMENTARY_CHARGE, PROTON_MASS
from essos.dynamics import Particles
from essos.fields import ExternalField
from essos.split import SplitStatus, trace_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("GuidingCenter", "FullOrbit"), default="GuidingCenter")
    parser.add_argument("--integrator", choices=("essos_rk45", "essos_dopri8"), default="essos_dopri8")
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--spatial-diffusion", action="store_true")
    parser.add_argument("--collision-controller", choices=("conservative", "paper", "candidate"), default="conservative")
    args = parser.parse_args()
    if args.particles < 1:
        parser.error("--particles must be positive")
    field = ExternalField(lambda points: jnp.broadcast_to(jnp.array([0., 0., 1.]), points.shape))
    particles = Particles(jnp.zeros((args.particles, 3)), mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE, energy=5000*ELEMENTARY_CHARGE,
        initial_vparallel_over_v=jnp.linspace(-.8, .8, args.particles), field=field)
    species = BackgroundSpecies(1, jnp.array([1.]), jnp.array([1.]),
                                jnp.array([1e20]), jnp.array([1000.]))
    result = trace_split(field, particles, species, model=args.model, integrator=args.integrator,
        tmax=1e-6, timestep=1e-7, n_save=11, orbit=not args.collision_only,
        spatial_diffusion=args.spatial_diffusion, collision_controller=args.collision_controller)
    result.y.block_until_ready()
    print("Outcomes:", {SplitStatus(int(k)).name: int(v)
                         for k, v in zip(*np.unique(result.status, return_counts=True))})
    print("Mean accepted split intervals:", float(np.mean(result.stats.accepted)))
    print("Mean rejected split trials:", float(np.mean(result.stats.rejected)))
    print("Mean orbit trials:", float(np.mean(result.stats.orbit_steps)))
    print("Peak stored Brownian intervals:", int(np.max(result.stats.peak_pending)))
    print("Maximum accepted collision error:", float(np.max(result.stats.max_accepted_error)))
    if np.any(np.asarray(result.status) > SplitStatus.EVENT):
        raise RuntimeError("Incomplete particle solve; inspect result.status")


if __name__ == "__main__":
    main()
