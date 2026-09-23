"""Adopt a finished optimization made with the same objective code into the archive.

python adopt.py SOURCE_DIR RUN_DIR 'NAME = value' ...

The example is executed up to its optimization with the given inputs, so CONFIG is the current
problem definition. The stored optimum is re-evaluated with the current code; the checkpoint is
written only if its recomputed cost matches the last recorded one. Termination status was not
recorded then and is inferred from the evaluation count, which the metadata states.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, D)
os.chdir(D)
src = open(f"{D}/optimize_coils_and_nearaxis_finite_beta.py").read()
head_end = src.index('"""', src.index('"""') + 3) + 3  # Never edit the module docstring.
head, src = src[:head_end], src[head_end:]
for edit in list(sys.argv[3:]) + [f'OUTPUT_DIR = Path("{out}")', "SHOW_PLOTS = False"]:
    name = edit.split("=")[0].strip()
    src, count = re.subn(rf"(^|; ){re.escape(name)} = [^\n;#]*", lambda m: m.group(1) + edit + "  ", src,
                         count=1, flags=re.M)
    assert count == 1, name
src = head + src
src = src[:src.index('""" Optimization """')]
namespace = {"__file__": f"{D}/optimize_coils_and_nearaxis_finite_beta.py", "__name__": "adopt"}
exec(compile(src, namespace["__file__"], "exec"), namespace)

import jax
import jax.numpy as jnp

saved = np.load(source / "optimized_dofs.npz")
initial, optimized, history = saved["initial"], saved["optimized"], saved["cost_history"]
assert np.allclose(initial, namespace["initial_dofs"], rtol=1e-12, atol=0), "different starting coils"
residuals = jax.jit(namespace["residuals"])
value = np.asarray(residuals(jnp.asarray(optimized)))
jacobian = np.asarray(jax.jit(jax.jacfwd(namespace["residuals"]))(jnp.asarray(optimized)))
cost = 0.5 * float(value @ value)
# The optimum returned by least_squares is the best accepted point, the minimum of the history.
recorded = float(np.min(history))
relative = abs(cost - recorded) / recorded
print(f"recomputed cost {cost:.10e}, recorded minimum {recorded:.10e}, relative difference {relative:.2e}")
assert relative < 1e-6, "the stored optimum does not reproduce with the current code"
max_nfev = namespace["MAX_FUNCTION_EVALUATIONS"]
optimization = dict(
    status=0 if len(history) >= max_nfev else None, nfev=len(history), cost=cost,
    gradient_inf_norm=float(np.max(np.abs(jacobian.T @ value))), recomputed_cost_relative_difference=relative,
    message=("maximum evaluations reached (inferred from the count)" if len(history) >= max_nfev else
             "stopped by a tolerance before the budget; status and message were not recorded"),
    seconds=None, adopted_from=str(source.name), note="optimized with ESSOS 89a087c / pyQSC_JAX 5ebd9b5 on 2026-09-21")
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "optimized_dofs.npz", initial=initial, optimized=optimized, cost_history=history,
         config=json.dumps(namespace["CONFIG"], default=float), config_hash=namespace["CONFIG_HASH"],
         optimization=json.dumps(optimization))
print(json.dumps(optimization, indent=1))
