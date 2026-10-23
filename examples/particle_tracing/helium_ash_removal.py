"""Helium residence from exported coefficients or a manufactured cylinder.

    python examples/particle_tracing/helium_ash_removal.py --manufactured
    python examples/particle_tracing/helium_ash_removal.py \
        --wout path/to/wout.nc --coefficients path/to/helium_response.json

Supply JSON ``coefficients`` matching evaluate_helium_residence kwargs and
``metadata`` with _REQUIRED_METADATA keys. Null kappa means absorbing.
Coefficients must cover the modeled radius; WOUT supplies geometry identity,
not missing transport or profiles. Manufactured mode tests the pipeline only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from essos.ash import evaluate_helium_residence, check_helium_residence

_REQUIRED_METADATA = (
    "equilibrium_sha256", "profiles_id", "source_id", "collision_model",
    "electric_field", "turbulent_closure", "boundary_model", "volume_origin",
    "coefficient_status", "trace_validity",
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_response(path, wout):
    """Host-only input audit; no file operations occur in the native evaluator."""
    from netCDF4 import Dataset
    bundle = json.loads(path.read_text())
    coefficients, metadata = bundle["coefficients"], bundle["metadata"]
    missing = set(_REQUIRED_METADATA)-metadata.keys()
    if missing:
        raise ValueError(f"Missing physical/provenance metadata: {sorted(missing)}")
    if any(metadata[key] is None or metadata[key] == "" for key in _REQUIRED_METADATA):
        raise ValueError("Metadata must explicitly describe assumptions or say 'not evaluated'")
    if metadata["equilibrium_sha256"] != _hash(wout):
        raise ValueError("Coefficient equilibrium hash does not match the supplied WOUT")
    if coefficients["boundary_kappa_m_s"] is None:
        coefficients["boundary_kappa_m_s"] = np.inf
    with Dataset(wout) as data:
        minor_radius = float(np.asarray(data.variables["Aminor_p"][:]).squeeze())
    outer_radius = float(coefficients["radial_edges_m"][-1])
    if not np.isfinite(minor_radius) or minor_radius <= 0 or outer_radius > minor_radius*(1+1.e-10):
        raise ValueError("Require a positive WOUT minor radius and modeled r <= Aminor_p")
    metadata = dict(metadata, wout=str(wout.resolve()), coefficient_sha256=_hash(path),
                    radial_coordinate="r=Aminor_p*sqrt(s)",
                    modeled_boundary_radius_m=outer_radius,
                    modeled_boundary_s=(outer_radius/minor_radius)**2)
    return coefficients, metadata


def manufactured_response(cells=128):
    e = np.linspace(0, 1, cells+1)
    return dict(radial_edges_m=e, cell_volume_m3=np.diff(e**2),
                volume_derivative_m2=2*e, diffusion_m2_s=np.ones_like(e),
                convection_m_s=np.zeros_like(e), source_m3_s=np.ones(cells),
                boundary_kappa_m_s=np.inf), dict(
                    scope="Manufactured cylinder only; no stellarator evaluated",
                    profiles_id="constant unit source", source_id="unit-volume cylinder",
                    collision_model="not evaluated", electric_field="not evaluated",
                    turbulent_closure="not evaluated", boundary_model="absorbing at r=1 m",
                    volume_origin="analytic W=2r, total volume 1 m^3",
                    coefficient_status="manufactured D=1 m^2/s, V=0",
                    trace_validity="not evaluated", modeled_boundary_radius_m=1.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--coefficients", type=Path)
    choice.add_argument("--manufactured", action="store_true")
    parser.add_argument("--wout", type=Path)
    parser.add_argument("--output", type=Path, help="Write the same compact report as JSON")
    args = parser.parse_args(argv)
    if args.coefficients:
        if args.wout is None:
            parser.error("--coefficients requires --wout")
        coefficients, metadata = load_response(args.coefficients, args.wout)
    else:
        if args.wout is not None:
            parser.error("--manufactured has no WOUT; use explicit exported coefficients")
        coefficients, metadata = manufactured_response()
    result = check_helium_residence(evaluate_helium_residence(**coefficients))
    fields = ("source_rate_s", "inventory", "cost", "mean_residence_s", "mean_occupation_s",
              "boundary_particle_rate_s", "forward_residual", "backward_residual",
              "duality_relative_error", "balance_relative_error", "minimum_relative_pivot", "maximum_peclet")
    report = dict(metadata=metadata, result={name: float(getattr(result, name)) for name in fields},
                  numerical_status="discrete checks passed; radial convergence not evaluated",
                  alpha_energy_retention="not evaluated", thermal_core_passage="coefficient-model estimate",
                  wall_interception="not evaluated", pumped_fraction="not modeled",
                  thermal_orbit_verification="not evaluated")
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return report


if __name__ == "__main__":
    main()
