"""Native JAX Dormand--Prince 5(4) and 8(7) forward orbit integration.

This module has no Diffrax dependency. The driver supports fixed or adaptive
steps, dense output, scalar boolean/level-set events and bounded work.
It deliberately has no custom adjoint yet: reverse-mode differentiation of the
adaptive while loop is a separate implementation phase.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.interpreters import partial_eval
import numpy as np

from .rk_dense import DOPRI8_OUTPUT
from .rk_dense_high import HIGHER_ORDER


class RKStatus(IntEnum):
    SUCCESS = 0
    EVENT = 1
    MAX_STEPS = 2
    MIN_STEP = 3
    NONFINITE = 4
    INVALID_INPUT = 5


class RKStep(NamedTuple):
    y: jax.Array
    error: jax.Array
    stages: jax.Array


class _RKTrial(NamedTuple):
    """Forward-driver stages k=h*f, with the unscaled FSAL value separate."""
    y: jax.Array
    error: jax.Array
    increments: jax.Array
    f1: jax.Array


class RKStats(NamedTuple):
    steps: jax.Array
    accepted: jax.Array
    rejected: jax.Array


class RKSolution(NamedTuple):
    ts: jax.Array
    ys: jax.Array
    t: jax.Array
    y: jax.Array
    status: jax.Array
    event_mask: object
    stats: RKStats
    step_ts: jax.Array


# Dormand--Prince stages, with Shampine's embedded error estimator (1986).
# These constants also allow direct comparison with Diffrax's Dopri5.
_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.)
_A = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0., 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_ERROR = (35 / 384 - 1951 / 21600, 0., 500 / 1113 - 22642 / 50085,
          125 / 192 - 451 / 720, -2187 / 6784 + 12231 / 42400,
          11 / 84 - 649 / 6300, -1 / 60)
_MID = (6025192743 / 30085553152 / 2, 0., 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2)

# Prince--Dormand (1981) 8(7), including the final FSAL derivative.
_C8 = (1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200,
       5490023248/9719169821, 13/20, 1201146811/1299019798, 1., 1., 1.)
_A8 = (
    (1/18,),
    (1/48, 1/16),
    (1/32, 0., 3/32),
    (5/16, 0., -75/64, 75/64),
    (3/80, 0., 0., 3/16, 3/20),
    (29443841/614563906, 0., 0., 77736538/692538347, -28693883/1125000000, 23124283/1800000000),
    (16016141/946692911, 0., 0., 61564180/158732637, 22789713/633445777,
     545815736/2771057229, -180193667/1043307555),
    (39632708/573591083, 0., 0., -433636366/683701615, -421739975/2616292301,
     100302831/723423059, 790204164/839813087, 800635310/3783071287),
    (246121993/1340847787, 0., 0., -37695042795/15268766246, -309121744/1061227803,
     -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789),
    (-1028468189/846180014, 0., 0., 8478235783/508512852, 1311729495/1432422823,
     -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649,
     -45442868181/3398467696, 3065993473/597172653),
    (185892177/718116043, 0., 0., -3185094517/667107341, -477755414/1098053517,
     -703635378/230739211, 5731566787/1027545527, 5232866602/850066563,
     -4093664535/808688257, 3962137247/1805957418, 65686358/487910083),
    (403863854/491063109, 0., 0., -5068492393/434740067, -411421997/543043805,
     652783627/914296604, 11173962825/925320556, -13158990841/6184727034,
     3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0.),
    (14005451/335480064, 0., 0., 0., 0., -59238493/1068277825, 181606767/758867731,
     561292985/797845732, -1041891430/1371343529, 760417239/1151165299,
     118820643/751138087, -528747749/2220607170, 1/4),
)
_ERROR8 = (14005451/335480064 - 13451932/455176623, 0., 0., 0., 0.,
           -59238493/1068277825 + 808719846/976000145,
           181606767/758867731 - 1757004468/5645159321,
           561292985/797845732 - 656045339/265891186,
           -1041891430/1371343529 + 3867574721/1518517206,
           760417239/1151165299 - 465885868/322736535,
           118820643/751138087 - 53011238/667516719, -528747749/2220607170 - 2/45, 1/4, 0.)
# Bogacki--Shampine (1990) continuous extension, in descending powers, followed
# by an overall factor s. These mathematical coefficients match Dopri8.
_DENSE8 = (
    (-6.3448349392860401388, 22.1396504998094068976, -30.0610568289666450593,
     19.9990069333683970610, -6.6910181737837595697, 1.),
    (0., 0., 0., 0., 0., 0.),
    (0., 0., 0., 0., 0., 0.),
    (0., 0., 0., 0., 0., 0.),
    (0., 0., 0., 0., 0., 0.),
    (-39.6107919852202505218, 116.4422149550342161651, -121.4999627731334642623,
     52.2273532792945524050, -7.6142658045872677172, 0.),
    (20.3761213808791436958, -67.1451318825957197185, 83.1721004639847717481,
     -46.8919164181093621583, 10.7281392630428866124, 0.),
    (7.3347098826795362023, -16.5672243527496524646, 9.5724507555993664382,
     -.1890893225010595467, .5526637063753648783, 0.),
    (32.8801774352459155182, -89.9916014847245016028, 87.8406057677205645007,
     -35.7075975946222072821, 4.2186562625665153803, 0.),
    (-10.1588990526426760954, 22.6237489648532849093, -17.4152107770762969005,
     6.2736448083240352160, -.6627209125361597559, 0.),
    (-12.5401268098782561200, 32.2362340167355370113, -28.5903289514790976966,
     10.3160881272450748458, -1.2636789001135462218, 0.),
    (29.5553001484516038033, -82.1020315488359848644, 81.6630950584341412934,
     -34.7650769866611817349, 5.4106037898590422230, 0.),
    (-41.7923486424390588923, 116.2662185791119533462, -114.9375291377009418170,
     47.7457971078225540396, -7.0321379067945741781, 0.),
    (20.3006925822100825485, -53.9020777466385396792, 50.2558364226176017553,
     -19.0082099341608028453, 2.3537586759714983486, 0.),
)


def _rk_step(rhs, t, y, h, args, f0, nodes, rows, error_weights):
    # Keep the differentiable numerical map independent of retries/controller
    # state. Zero tableau entries create no dependencies in the accepted map.
    stages = [rhs(t, y, args) if f0 is None else f0]
    for c, weights in zip(nodes, rows):
        trial = y + h * sum(a * k for a, k in zip(weights, stages) if a != 0)
        stages.append(rhs(t + c * h, trial, args))
    # For both FSAL tableaus the final stage is evaluated at the solution.
    error = h * sum(a * k for a, k in zip(error_weights, stages) if a != 0)
    return RKStep(trial, error, jnp.stack(stages))


def rk45_step(rhs, t, y, h, args=None, *, f0=None):
    """One DP5(4) trial step. ``stages`` contains derivatives, not h*f.

    Pass the last derivative of the previous *accepted* step as ``f0`` for
    FSAL reuse. Rejected trials must leave that cached derivative unchanged.
    """
    return _rk_step(rhs, t, y, h, args, f0, _C, _A, _ERROR)


def dopri8_step(rhs, t, y, h, args=None, *, f0=None):
    """One DP8(7) trial step, with the same FSAL contract as rk45_step."""
    return _rk_step(rhs, t, y, h, args, f0, _C8, _A8, _ERROR8)


def _trial_step(rhs, t, y, h, args, f0, nodes, rows, error_weights):
    """The same explicit RK formula, storing increments for the forward driver.

    A static stage loop supports ordinary JAX forward/reverse derivatives.
    Only k is carried; the final FSAL evaluation is outside the loop and is
    kept unscaled, so cache reuse and h=0 derivatives need no division by h.
    Both particle models use this map without a physics-specific branch.
    """
    f0 = rhs(t, y, args) if f0 is None else f0
    n = len(error_weights)
    coefficients = np.zeros((n - 2, n - 1))
    for i, row in enumerate(rows[:-1]):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    times = jnp.asarray(nodes[:-1], dtype=y.dtype)
    endpoint_weights = jnp.asarray(rows[-1], dtype=y.dtype)
    bank = jnp.concatenate([(h * f0)[None], jnp.zeros((n - 2,) + y.shape, y.dtype)])

    def stage(i, bank):
        point = y + jnp.tensordot(coefficients[i - 1], bank, axes=1)
        increment = h * rhs(t + times[i - 1] * h, point, args)
        return lax.dynamic_update_index_in_dim(bank, increment, i, axis=0)

    bank = lax.fori_loop(1, n - 1, stage, bank)
    endpoint = y + jnp.tensordot(endpoint_weights, bank, axes=1)
    final = rhs(t + h, endpoint, args)
    error = jnp.tensordot(jnp.asarray(error_weights[:-1], dtype=y.dtype), bank, axes=1)
    if error_weights[-1] != 0:
        error = error + error_weights[-1] * (h * final)
    increments = jnp.concatenate([bank, (h * final)[None]])
    return _RKTrial(endpoint, error, increments, final)


def rk45_interpolate(y0, h, step, fraction):
    """Quartic dense output on an accepted step, for 0 <= fraction <= 1."""
    return _rk45_dense(y0, step.y, h * step.stages, fraction)


def _rk45_dense(y0, y1, k, fraction):
    delta = y1 - y0
    midpoint = jnp.tensordot(jnp.asarray(_MID, dtype=y0.dtype), k, axes=1)
    # Interpolate the endpoints, endpoint derivatives, and Shampine midpoint.
    a = 2 * (k[-1] - k[0]) - 8 * delta + 16 * midpoint
    b = 5 * k[0] - 3 * k[-1] + 14 * delta - 32 * midpoint
    c = k[-1] - 4 * k[0] - 5 * delta + 16 * midpoint
    s = fraction
    value = y0 + s * (k[0] + s * (c + s * (b + s * a)))
    return jnp.where(s == 0, y0, jnp.where(s == 1, y1, value))


def dopri8_interpolate(y0, h, step, fraction):
    """Dopri8 continuous extension on a trial step, for 0 <= fraction <= 1."""
    return _dopri8_dense(y0, step.y, h * step.stages, fraction)


def _dopri8_dense(y0, y1, k, fraction):
    coefficients = jnp.asarray(_DENSE8, dtype=y0.dtype)
    weights = jax.vmap(jnp.polyval, in_axes=(0, None))(coefficients, fraction) * fraction
    value = y0 + jnp.tensordot(weights, k, axes=1)
    return jnp.where(fraction == 0, y0, jnp.where(fraction == 1, y1, value))


def _dopri8_output_coefficients(increments):
    """Prepare the optional save polynomial once, reusing the trial stages."""
    return jnp.asarray(DOPRI8_OUTPUT, dtype=increments.dtype).T @ increments


def _higher_order_output_coefficients(rhs, t, y, h, increments, args, method):
    """Output-only RK stages: continuous order five (DP5) or seven (PD8).

    The original increments are read-only. The added derivatives are never
    used for the endpoint, embedded error, controller, or FSAL cache. A static
    stage loop gives this map ordinary JAX forward/reverse derivatives. Keep
    its storage local to the current step and reuse the polynomial for saves.
    """
    nodes, rows, output = HIGHER_ORDER[method]
    original, extra = len(increments), len(nodes)
    coefficients = np.zeros((extra, original + extra))
    for i, row in enumerate(rows):
        coefficients[i, :len(row)] = row
    coefficients = jnp.asarray(coefficients, dtype=y.dtype)
    nodes = jnp.asarray(nodes, dtype=y.dtype)
    bank = jnp.pad(increments, ((0, extra), (0, 0)))

    def stage(i, bank):
        state = y + jnp.tensordot(coefficients[i], bank, axes=1)
        increment = h * rhs(t + nodes[i]*h, state, args)
        return lax.dynamic_update_index_in_dim(bank, increment, original+i, axis=0)

    bank = lax.fori_loop(0, extra, stage, bank)
    return jnp.asarray(output, dtype=y.dtype).T @ bank


def _polynomial_dense(y0, y1, coefficients, fraction):
    value = coefficients[0]
    for coefficient in coefficients[1:]:
        value = coefficient + fraction * value
    value = y0 + fraction * value
    return jnp.where(fraction == 0, y0, jnp.where(fraction == 1, y1, value))


def _concrete_check(value, predicate, message):
    if not isinstance(value, jax.core.Tracer) and not predicate(np.asarray(value)):
        raise ValueError(message)


def _prepare_rhs(rhs, t0, y0, args):
    """Separate solve-invariant work from the time/state-dependent RHS.

    Time and the whole state are unknown to this JAX partial evaluation.
    Parameter-only expressions are evaluated once per solve and passed to
    every stage. They remain JAX values with ordinary parameter derivatives;
    nothing is stopped, converted to NumPy, or stored on the field object.
    Both orbit models use the same preparation and the original RHS.

    Keep this use of JAX's partial-evaluation interface isolated here. The
    resulting setup/stage graphs execute ordinary JAX primitives, including
    when enclosed by jit, vmap, JVP or VJP.
    """
    graph = jax.make_jaxpr(lambda t, y: rhs(t, y, args))(t0, y0)
    setup, stage, _, _ = partial_eval.partial_eval_jaxpr_nounits(
        graph, [True, True], instantiate=True)
    residuals = jax.core.eval_jaxpr(setup.jaxpr, setup.consts)

    def evaluate(t, y, unused_args):
        return jax.core.eval_jaxpr(stage.jaxpr, stage.consts, *residuals, t, y)[0]

    return evaluate


def _first_hit(hits):
    """Report one event, using leaf order to break simultaneous-event ties."""
    previous = jnp.asarray(False)
    result = []
    for hit in hits:
        result.append(hit & ~previous)
        previous = previous | hit
    return tuple(result)


def solve_rk(rhs, y0, args=None, *, t0=0., t1, dt0, ts=(), adaptive=True,
               rtol=1e-9, atol=1e-9, event=None, root_find=True,
               max_steps=1_000_000, dtmin=0., dtmax=np.inf, record_steps=False, method="dopri5",
               error_norm=None, controller="i", pcoeff=.4, icoeff=.3, dense_output="standard"):
    """Integrate an array state forward in time with bounded adaptive retries.

    ``ts`` is an ordered save grid, independent of the integration steps.
    Unreached saves are filled with infinity. ``t``/``y`` give the last accepted
    state, including a refined event crossing. Level-set events fire on a sign
    change (or an initial zero); boolean events fire when true. Event callables
    have signature ``event(t, y, args)`` and may form a PyTree.
    The earliest event is reported; ties use PyTree leaf order.
    An initial event leaves all sampled saves empty, while returning t0/y0 as
    the terminal state, matching Diffrax's event/save convention.

    By default a scaled RMS embedded error controls adaptive acceptance. Pass
    ``error_norm(y0, y1, embedded_error) -> scalar`` to supply an already scaled
    error (acceptance <= 1), for example ``essos.orbit_error.OrbitErrorNorm``.
    It affects acceptance and step selection only; it does not alter RK stages.
    ``controller="i"`` keeps the original integral controller. ``"pi"`` uses
    ``factor = .9 * error**(-(icoeff+pcoeff)/q) * previous_error**(pcoeff/q)``,
    where q is the embedded error order and history changes only on acceptance.
    PI defaults to p=.4, i=.3, with accepted growth bounded to [1, 10] and
    rejected growth to [.2, .9], for comparison with ESSOS's Diffrax PI settings.
    Fixed stepping
    uses ``dt0`` (bounded by dtmax and t1), and still supplies dense output/events.
    ``max_steps`` counts accepted and rejected trials. Failures return a status;
    no inaccurate step is forced through a minimum step-size limit.

    By default memory holds only the save grid and current step. Opt into an
    infinity-padded accepted-time history with ``record_steps=True``; its first
    ``stats.accepted + 1`` entries are valid. This is diagnostic data for future
    accepted-step replay, not an implemented AD rule.

    Concrete invalid inputs raise ValueError. Invalid dynamic values under JIT
    return INVALID_INPUT. Non-finite states, derivatives or event values are
    rejected, with NONFINITE returned when no smaller trial can proceed.

    ``method`` selects ``"dopri5"`` or ``"dopri8"``. The convenience entry points
    ``solve_rk45`` and ``solve_dopri8`` fix this choice.
    ``dense_output="optimized"`` selects a coefficient-only Dopri8 save
    polynomial with smaller order-six/seven B-series error norm. It reuses all
    stages and prepares six coefficient vectors once per trial. RK endpoints,
    acceptance and event-root interpolation are unaffected. Formal dense order
    remains five; error reduction is problem-dependent. Dopri5 keeps its
    standard quartic for either setting.
    ``dense_output="higher_order"`` adds two output-only RHS evaluations for
    Dopri5 (continuous order five) or three for Dopri8 (continuous order seven).
    They run once per accepted step containing an interior save, and share
    their polynomial across all saves in that step. Events retain the standard
    interpolant. A non-finite output evaluation returns NONFINITE without
    retrying or changing the accepted orbit step. These maps support ordinary
    AD; the adaptive driver still has no reverse-mode replay rule.
    """
    if method == "dopri5":
        nodes, rows, weights, dense, error_order = _C, _A, _ERROR, _rk45_dense, 5
    elif method == "dopri8":
        nodes, rows, weights, dense, error_order = _C8, _A8, _ERROR8, _dopri8_dense, 8
    else:
        raise ValueError("method must be 'dopri5' or 'dopri8'")
    if controller not in ("i", "pi"):
        raise ValueError("controller must be 'i' or 'pi'")
    if dense_output not in ("standard", "optimized", "higher_order"):
        raise ValueError("dense_output must be 'standard', 'optimized' or 'higher_order'")
    if controller == "pi":
        for value, name, positive in ((pcoeff, "pcoeff", False), (icoeff, "icoeff", True)):
            if jnp.asarray(value).ndim != 0:
                raise ValueError(f"{name} must be scalar")
            _concrete_check(value, lambda x: np.isfinite(x) & ((x > 0) if positive else (x >= 0)),
                            f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    def interpolate(y, h, trial, fraction):
        return dense(y, trial.y, trial.increments, fraction)
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    y0 = jnp.asarray(y0, dtype=jnp.result_type(y0, float))
    if y0.ndim != 1 or not y0.size:
        raise ValueError("y0 must be a nonempty one-dimensional array")
    if not jnp.issubdtype(y0.dtype, jnp.floating):
        raise ValueError("y0 must contain real values")
    ts = jnp.asarray(ts, dtype=y0.dtype)
    if ts.ndim != 1:
        raise ValueError("ts must be one-dimensional")
    for value, name in ((t0, "t0"), (t1, "t1"), (dt0, "dt0"), (dtmin, "dtmin"), (dtmax, "dtmax")):
        if jnp.asarray(value).ndim != 0:
            raise ValueError(f"{name} must be scalar")
    for value, name in ((dt0, "dt0"), (dtmax, "dtmax")):
        _concrete_check(value, lambda x: np.all(x > 0), f"{name} must be positive")
    _concrete_check(dt0, np.isfinite, "dt0 must be finite")
    _concrete_check(t0, np.isfinite, "t0 must be finite")
    _concrete_check(t1, np.isfinite, "t1 must be finite")
    _concrete_check(dtmin, lambda x: np.isfinite(x) & (x >= 0), "dtmin must be finite and nonnegative")
    _concrete_check(rtol, lambda x: np.all(np.isfinite(x) & (x >= 0)), "rtol must be finite and nonnegative")
    _concrete_check(atol, lambda x: np.all(np.isfinite(x) & (x > 0)), "atol must be finite and positive")
    _concrete_check(jnp.asarray(t1) - t0, lambda x: np.isfinite(x) & (x >= 0), "t1 must be finite and >= t0")
    _concrete_check(ts, lambda x: np.all(np.isfinite(x)) and np.all(np.diff(x) >= 0), "ts must be finite and ordered")
    _concrete_check(jnp.asarray(dtmax) - dtmin, lambda x: x >= 0, "dtmax must be >= dtmin")
    _concrete_check(jnp.asarray(dt0) - dtmin, lambda x: x >= 0, "dt0 must be >= dtmin")
    _concrete_check((ts >= t0) & (ts <= t1), np.all, "ts must lie in [t0, t1]")
    t0, t1, h0 = (jnp.asarray(v, dtype=y0.dtype) for v in (t0, t1, dt0))
    # Comparisons made above may be tracers, so retain runtime validation too.
    # For ordinary Tracing calls these constants are folded out by XLA.
    valid_input = (jnp.isfinite(t0) & jnp.isfinite(t1) & (t1 >= t0) &
                   jnp.isfinite(h0) & (h0 > 0) & jnp.isfinite(dtmin) & (dtmin >= 0) &
                   (dtmax > 0) & (dtmax >= dtmin) & (h0 >= dtmin) &
                   jnp.all(jnp.isfinite(jnp.asarray(rtol)) & (jnp.asarray(rtol) >= 0)) &
                   jnp.all(jnp.isfinite(jnp.asarray(atol)) & (jnp.asarray(atol) > 0)) &
                   jnp.all(jnp.isfinite(ts) & (ts >= t0) & (ts <= t1)) &
                   jnp.all(jnp.diff(ts) >= 0))
    if controller == "pi":
        valid_input = valid_input & jnp.isfinite(pcoeff) & (pcoeff >= 0) & jnp.isfinite(icoeff) & (icoeff > 0)
    functions, structure = tree_util.tree_flatten(event)

    def events(t, y):
        values = tuple(jnp.asarray(f(t, y, args)) for f in functions)
        if any(v.shape != () or (v.dtype != jnp.bool_ and not jnp.issubdtype(v.dtype, jnp.floating)) for v in values):
            raise ValueError("events must return scalar booleans or floating-point level sets")
        return values

    def finite_events(values):
        return jnp.all(jnp.asarray(tuple(jnp.isfinite(v) for v in values), dtype=bool))

    values0 = events(t0, y0)
    initial_hits = _first_hit(tuple(v if v.dtype == jnp.bool_ else v == 0 for v in values0))
    hit0 = jnp.any(jnp.asarray(initial_hits, dtype=bool))
    status0 = jnp.where(valid_input,
                       jnp.where(jnp.isfinite(y0).all() & finite_events(values0),
                                 jnp.where(hit0, RKStatus.EVENT, RKStatus.SUCCESS), RKStatus.NONFINITE),
                       RKStatus.INVALID_INPUT)
    rhs = _prepare_rhs(rhs, t0, y0, args)
    state = dict(t=t0, y=y0, h=jnp.minimum(h0, dtmax), f=rhs(t0, y0, args),
                 values=values0, hits=initial_hits, status=status0,
                 ys=jnp.full((len(ts),) + y0.shape, jnp.inf, dtype=y0.dtype),
                 saved_ts=jnp.full(ts.shape, jnp.inf, dtype=ts.dtype),
                 index=jnp.array(0), steps=jnp.array(0), accepted=jnp.array(0), rejected=jnp.array(0),
                 history=jnp.full((max_steps + 1 if record_steps else 0,), jnp.inf, dtype=y0.dtype))
    if adaptive and controller == "pi":
        state["previous_error"] = jnp.ones((), dtype=y0.dtype)
    if record_steps:
        state["history"] = state["history"].at[0].set(t0)

    def save_initial(s):
        if not len(ts):
            return s
        def body(s):
            i = s["index"]
            return dict(s, ys=s["ys"].at[i].set(y0), saved_ts=s["saved_ts"].at[i].set(t0), index=i + 1)
        return lax.while_loop(lambda s: ~hit0 & (s["index"] < len(ts)) &
                              (ts[jnp.minimum(s["index"], len(ts) - 1)] == t0), body, s)

    state = save_initial(state)

    def advance(s):
        t, y = s["t"], s["y"]
        remaining = t1 - t
        if adaptive:
            h = jnp.minimum(s["h"], remaining)
            # Snap a roundoff-sized remainder instead of taking a tiny extra step.
            h = jnp.where(remaining - h <= 8 * jnp.finfo(y.dtype).eps *
                          jnp.maximum(jnp.abs(t1), jnp.abs(t)), remaining, h)
            end_t = jnp.where(h == remaining, t1, t + h)
        else:
            # Derive fixed times from the step index: repeatedly adding h0 can
            # leave a tiny remainder even when the interval is exactly N*h0.
            end_t = jnp.minimum(t0 + (s["accepted"] + 1) * jnp.minimum(h0, dtmax), t1)
            h = end_t - t
        trial = _trial_step(rhs, t, y, h, args, s["f"], nodes, rows, weights)
        finite = (jnp.isfinite(trial.y).all() & jnp.isfinite(trial.increments).all() &
                  jnp.isfinite(trial.f1).all() & jnp.isfinite(trial.error).all())
        if adaptive:
            if error_norm is None:
                scale = jnp.asarray(atol) + jnp.asarray(rtol) * jnp.maximum(jnp.abs(y), jnp.abs(trial.y))
                error = jnp.sqrt(jnp.mean(jnp.square(trial.error / scale)))
            else:
                error = jnp.asarray(error_norm(y, trial.y, trial.error))
                if error.ndim != 0:
                    raise ValueError("error_norm must return a scalar")
            finite = finite & jnp.isfinite(error) & (error >= 0)
        else:
            error = jnp.zeros((), dtype=y.dtype)
        new_values = events(end_t, trial.y)
        finite = finite & finite_events(new_values)
        accept = finite & ((error <= 1) | (not adaptive)) & (end_t > t)
        triggered = tuple((new if old.dtype == jnp.bool_ else ((old > 0) != (new > 0)) | (new == 0)) & accept
                          for old, new in zip(s["values"], new_values))
        fractions = []
        valid_roots = []
        for f, old, new, hit in zip(functions, s["values"], new_values, triggered):
            if root_find and old.dtype != jnp.bool_:
                def bisect(c):
                    lo, hi, iteration, valid = c
                    mid = (lo + hi) / 2
                    value = f(t + mid * h, interpolate(y, h, trial, mid), args)
                    same = (value > 0) == (old > 0)
                    return (jnp.where(same, mid, lo), jnp.where(same, hi, mid),
                            iteration + 1, valid & jnp.isfinite(value))
                # A while-loop avoids root work on ordinary steps under vmap.
                lo, hi, _, valid_root = lax.while_loop(lambda c: hit & c[3] & (c[2] < 52), bisect,
                                           (jnp.zeros_like(t), jnp.ones_like(t), jnp.array(0), jnp.asarray(True)))
                fraction = (lo + hi) / 2
            else:
                fraction = jnp.ones_like(t)
                valid_root = jnp.asarray(True)
            fractions.append(jnp.where(hit, fraction, jnp.inf))
            valid_roots.append(valid_root)
        finite = finite & jnp.all(jnp.asarray(valid_roots, dtype=bool))
        accept = accept & finite
        first = jnp.min(jnp.asarray(fractions)) if fractions else jnp.asarray(jnp.inf, dtype=t.dtype)
        has_event = jnp.isfinite(first) & accept
        fraction = jnp.minimum(first, 1.)
        stop_t = jnp.where(has_event, t + fraction * h, end_t)
        stop_y = interpolate(y, h, trial, fraction)
        hits = _first_hit(tuple(hit & accept & (fraction_i == first) for hit, fraction_i in zip(triggered, fractions)))
        if adaptive and controller == "pi":
            # Work in log space so zero errors and clipped factors stay finite.
            safe_error = jnp.maximum(jnp.where(finite, error, 1.), jnp.finfo(y.dtype).tiny)
            log_factor = (jnp.log(.9) - (icoeff + pcoeff) / error_order * jnp.log(safe_error)
                          + pcoeff / error_order * jnp.log(s["previous_error"]))
            factor = jnp.exp(jnp.clip(log_factor, jnp.log(jnp.where(accept, 1., .2)),
                                     jnp.log(jnp.where(accept, 10., .9))))
            proposed = h * jnp.where(finite, factor, .2)
        else:
            # Preserve the original I-controller, including its growth limits.
            factor = jnp.where(finite, jnp.clip(.9 * jnp.maximum(error, 1e-16)**(-1/error_order), .2, 5.), .2)
            proposed = h * jnp.where(accept, factor, jnp.minimum(factor, .9)) if adaptive else h0
        next_h = jnp.clip(proposed, dtmin, dtmax)
        stuck = (end_t <= t) | (~accept & ((h <= dtmin) | (t + next_h <= t)))
        if not adaptive:
            stuck = stuck | ~finite
        status = jnp.where(has_event, RKStatus.EVENT,
                           jnp.where(stuck, jnp.where(finite, RKStatus.MIN_STEP, RKStatus.NONFINITE), RKStatus.SUCCESS))
        updated = dict(s, t=jnp.where(accept, stop_t, t), y=jnp.where(accept, stop_y, y), h=next_h,
                       f=jnp.where(accept, trial.f1, s["f"]),
                       values=tuple(jnp.where(accept, new, old) for old, new in zip(s["values"], new_values)),
                       hits=hits, status=status, steps=s["steps"] + 1,
                       accepted=s["accepted"] + accept.astype(int), rejected=s["rejected"] + (~accept).astype(int))
        if adaptive and controller == "pi":
            # Zero errors use neutral history; rejected trials retain the last
            # accepted error, including after a non-finite trial.
            history_error = jnp.where(error >= jnp.finfo(y.dtype).tiny, safe_error, 1.)
            updated["previous_error"] = jnp.where(accept, history_error, s["previous_error"])
        if record_steps:
            updated["history"] = lax.cond(accept, lambda history: history.at[updated["accepted"]].set(stop_t),
                                          lambda history: history, s["history"])
        if len(ts):
            output_finite = jnp.asarray(True)
            if dense_output == "higher_order":
                degree = 5 if method == "dopri5" else 7
                next_save = ts[jnp.minimum(s["index"], len(ts)-1)]
                needed = (accept & (s["index"] < len(ts)) & (next_save > t) &
                          (next_save < end_t) & (next_save <= stop_t))
                def prepare_output(carry):
                    coefficients = _higher_order_output_coefficients(
                        rhs, t, y, h, trial.increments, args, method)
                    return (jnp.asarray(False), coefficients, jnp.all(jnp.isfinite(coefficients)))
                # A single-iteration while skips output work under vmap when
                # no particle needs it. vmap(cond) evaluates both branches.
                _, coefficients, output_finite = lax.while_loop(lambda carry: carry[0], prepare_output,
                    (needed, jnp.zeros((degree,) + y.shape, dtype=y.dtype), output_finite))
                updated["status"] = jnp.where(output_finite, updated["status"], RKStatus.NONFINITE)
                def save_value(fraction):
                    return _polynomial_dense(y, trial.y, coefficients, fraction)
            elif dense_output == "optimized" and method == "dopri8":
                coefficients = _dopri8_output_coefficients(trial.increments)
                def save_value(fraction):
                    return _polynomial_dense(y, trial.y, coefficients, fraction)
            else:
                def save_value(fraction):
                    return interpolate(y, h, trial, fraction)
            def save(s):
                i = s["index"]
                value = save_value((ts[i] - t) / h)
                if dense_output == "higher_order":
                    # The event terminal state belongs to the event interpolant.
                    value = jnp.where(has_event & (ts[i] == stop_t), stop_y, value)
                return dict(s, ys=s["ys"].at[i].set(value), saved_ts=s["saved_ts"].at[i].set(ts[i]), index=i + 1)
            updated = lax.while_loop(lambda s: accept & output_finite & (s["index"] < len(ts)) &
                                     (ts[jnp.minimum(s["index"], len(ts) - 1)] <= stop_t), save, updated)
        return updated

    state = lax.while_loop(lambda s: (s["t"] < t1) & (s["status"] == RKStatus.SUCCESS) &
                           (s["steps"] < max_steps), advance, state)
    status = jnp.where((state["t"] < t1) & (state["status"] == RKStatus.SUCCESS), RKStatus.MAX_STEPS, state["status"])
    return RKSolution(state["saved_ts"], state["ys"], state["t"], state["y"], status,
                      tree_util.tree_unflatten(structure, state["hits"]),
                      RKStats(state["steps"], state["accepted"], state["rejected"]), state["history"])


def solve_rk45(rhs, y0, args=None, **kwargs):
    """Dormand--Prince 5(4); see solve_rk for integration options and results."""
    return solve_rk(rhs, y0, args, method="dopri5", **kwargs)


def solve_dopri8(rhs, y0, args=None, **kwargs):
    """Dormand--Prince 8(7); see solve_rk for integration options and results."""
    return solve_rk(rhs, y0, args, method="dopri8", **kwargs)
