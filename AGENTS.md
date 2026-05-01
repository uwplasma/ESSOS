# AGENTS.md

When modifying VMEC-extender or BMW code:

1. Do not reimplement VMEC field construction; use `vmec_jax` public APIs.
2. Treat `phi` vs `zeta` conventions as high-risk. Add tests for every change touching toroidal coordinates.
3. Treat virtual-casing `internal` vs `external` branch signs as high-risk. Run branch/sign physics tests.
4. Keep shapes static in JIT-heavy functions.
5. Preserve x64 support and avoid silently downcasting physics arrays.
6. All public APIs need numerical tests, docs, and examples.
7. Poincare plots and wall hits are diagnostics unless smoothed; do not claim differentiability for discontinuous events.
8. Every PR touching this code must run physics validation, gradient checks, docs build, and coverage gates.
