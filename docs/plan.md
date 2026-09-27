# README and documentation plan

A follow-up will expand the README in the VMEX style: one short section per capability, each with one compact figure from `docs/make_readme_figures.py` and two or three sentences with measured numbers. Planned sections:

- **Tracing through the magnetic axis:** VMEC guiding centers in the axis-regular chart, with an orbit that passes s = 0.
- **Beyond the LCFS to a wall (with #83):** the exterior field from coils or a VMEX `VmecExtender`, wall strikes and re-entry.
- **Speed and parallelism:** fused guiding-center evaluation (3.3-4.1x on the examples), device sharding over CPUs and GPUs with `devices=`, and compile reuse (#86). A scaling figure of particles per second against the number of CPU and GPU devices.
- **Collisional tracing:** Monte Carlo collisions against background species with radial profiles, with slowing down checked against Stix.
- **Connection length to a wall:** `connection_length`.
- **Combined fields:** `CombinedField`, coils plus another field.
- **Coil optimization:** the augmented Lagrangian with force objectives (Lorentz force, surface distance, curvature), with gradients checked against finite differences.

The SIMSOPT comparison, installation, first steps and examples table stay.

## Documentation plan

Once the open PRs are merged, `docs/` (readthedocs) will be refactored into a complete, built reference:

- **Tutorials:** installation, first traces, coil design, optimization.
- **Theory and physical models:** fields (Biot-Savart, VMEC, near-axis), guiding-center and full-orbit equations, collision operators, the exterior field and wall.
- **Algorithms:** integrators and events, the axis-regular chart, the fused evaluation, sharding and compile reuse, the augmented Lagrangian, differentiation through orbits.
- **API reference:** generated from the docstrings.
- **Examples gallery:** the scripts in `examples/` with their figures.

