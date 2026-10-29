"""ESSOS Dopri5/8 forward orbit solvers and private GPU field adapters.

Fixed/adaptive integration, physical error budgets, standard dense output and
events share one implementation. Independent GPU particle loops retain the
same formulas. Diffrax is used only by the existing-Tracing result adapter;
native numerical integration supports an optional accepted-step reverse rule.
"""
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax, tree_util
# Some JAX versions omit this helper from the public partial_eval facade.
from jax._src.interpreters import partial_eval
from jax.extend.core import Literal
import numpy as np

# Physical error budgets

def orbit_block_norm(error, velocity_components):
    """Maximum of position and velocity Euclidean norms after scaling."""
    return jnp.maximum(jnp.linalg.norm(error[:3]), jnp.linalg.norm(error[3:3 + velocity_components]))


@dataclass(frozen=True, eq=False)
class OrbitErrorNorm:
    """Embedded orbit error in physical position and velocity blocks.

    ``position_map``, when supplied, maps an entire integration state to its
    Cartesian position in metres. Comparing the embedded endpoints handles
    periodic coordinates and regular axis charts without dividing by radius.
    Its result is a local embedded estimate, not a global trajectory bound.

    ``velocity_components`` is one for guiding centers and three for full
    orbits; any remaining state (constant mu, failure flag) is excluded.
    A chart angle that affects position must be included by ``position_map``.
    """

    rtol: object
    atol: object
    speed_scale: object
    length_scale: object = 1.
    velocity_components: int = 1
    position_map: Callable | None = None

    def __post_init__(self):
        if self.velocity_components not in (1, 3):
            raise ValueError("velocity_components must be 1 or 3")
        for name in ("rtol", "atol", "speed_scale", "length_scale"):
            value = getattr(self, name)
            if np.ndim(value) != 0:
                raise ValueError(f"orbit {name} must be scalar")
            if not isinstance(value, jax.core.Tracer):
                value = np.asarray(value)
                valid = np.isfinite(value) & (value >= 0 if name == "rtol" else value > 0)
                if not bool(valid):
                    raise ValueError(f"orbit {name} must be finite and {'nonnegative' if name == 'rtol' else 'positive'}")

    def __call__(self, y0, y1, error):
        del y0  # Reference scales do not depend on the coordinate origin.
        if self.position_map is None:
            position_error = error[:3]
        else:
            position_error = self.position_map(y1) - self.position_map(y1 - error)
        velocity_error = error[3:3 + self.velocity_components]
        budget = jnp.asarray(self.atol) + jnp.asarray(self.rtol)
        normalized = jnp.concatenate([position_error / (budget * self.length_scale),
                                      velocity_error / (budget * self.speed_scale)])
        value = orbit_block_norm(normalized, self.velocity_components)
        valid = ((self.length_scale > 0) & jnp.isfinite(self.length_scale) &
                 (self.speed_scale > 0) & jnp.isfinite(self.speed_scale) &
                 (self.rtol >= 0) & (self.atol > 0) & jnp.isfinite(budget) &
                 jnp.isfinite(y1).all() & jnp.isfinite(error).all() & jnp.isfinite(value))
        return jnp.where(valid, value, jnp.inf)

# Runge-Kutta stages and portable driver

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
    next_dt: object = None


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


def _trial_unrolled(rhs, t, y, h, args, f0, nodes, rows, weights):
    """Explicit component arithmetic exposes independent operations to XLA."""
    increments = [tuple(h * value for value in f0)]
    for node, row in zip(nodes, rows):
        point = tuple(value + sum(a*k[i] for a,k in zip(row,increments) if a)
                      for i,value in enumerate(y))
        final = tuple(rhs(t+node*h, jnp.stack(point), args))
        increments.append(tuple(h*value for value in final))
    error = tuple(sum(a*k[i] for a,k in zip(weights,increments) if a) for i in range(len(y)))
    return _RKTrial(jnp.stack(point), jnp.stack(error),
                    jnp.stack(tuple(jnp.stack(k) for k in increments)), jnp.stack(final))


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
               error_norm=None, controller="i", pcoeff=.4, icoeff=.3, stage_execution="unrolled",
               adjoint="none", adjoint_checkpoint_steps=16, adjoint_max_accepted_steps=None,
               adjoint_parameter_accumulation="segment", _recorder=None):
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
    error (acceptance <= 1), for example ``essos.rk.OrbitErrorNorm``.
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
    accepted-step inspection. ``adjoint="accepted"`` separately enables the
    checkpointed reverse rule in ``essos.native_ad``. Its accepted times, event
    fractions and controller decisions are frozen; stages and dense output
    retain their physical derivatives.

    Concrete invalid inputs raise ValueError. Invalid dynamic values under JIT
    return INVALID_INPUT. Non-finite states, derivatives or event values are
    rejected, with NONFINITE returned when no smaller trial can proceed.

    ``method`` selects ``"dopri5"`` or ``"dopri8"``. The convenience entry points
    ``solve_rk45`` and ``solve_dopri8`` fix this choice.
    """
    if adjoint not in ('none','accepted'):
        raise ValueError("adjoint must be 'none' or 'accepted'")
    if adjoint == 'accepted':
        options=locals().copy()
        from .native_ad import solve_accepted
        for name in ('rhs','y0','args','adjoint','_recorder'):
            options.pop(name)
        return solve_accepted(rhs,y0,args,**options)
    if stage_execution not in ("loop", "unrolled"):
        raise ValueError("stage_execution must be 'loop' or 'unrolled'")
    trial_step = _trial_step if stage_execution == "loop" else _trial_unrolled
    if method == "dopri5":
        nodes, rows, weights, dense, error_order = _C, _A, _ERROR, _rk45_dense, 5
    elif method == "dopri8":
        nodes, rows, weights, dense, error_order = _C8, _A8, _ERROR8, _dopri8_dense, 8
    else:
        raise ValueError("method must be 'dopri5' or 'dopri8'")
    if controller not in ("i", "pi"):
        raise ValueError("controller must be 'i' or 'pi'")
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
    y0 = jnp.asarray(y0)
    y0 = y0.astype(jnp.result_type(y0, float))
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
    initial_saves = state["index"]
    if _recorder is not None:
        state = _recorder.initialize(state)

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
        trial = trial_step(rhs, t, y, h, args, s["f"], nodes, rows, weights)
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
            def save(s):
                i = s["index"]
                value = interpolate(y, h, trial, (ts[i] - t) / h)
                return dict(s, ys=s["ys"].at[i].set(value), saved_ts=s["saved_ts"].at[i].set(ts[i]), index=i + 1)
            updated = lax.while_loop(lambda s: accept & (s["index"] < len(ts)) &
                                     (ts[jnp.minimum(s["index"], len(ts) - 1)] <= stop_t), save, updated)
        if _recorder is not None:
            updated = _recorder.record(s, updated, t, y, h, fraction, accept)
        return updated

    state = lax.while_loop(lambda s: (s["t"] < t1) & (s["status"] == RKStatus.SUCCESS) &
                           (s["steps"] < max_steps), advance, state)
    status = jnp.where((state["t"] < t1) & (state["status"] == RKStatus.SUCCESS), RKStatus.MAX_STEPS, state["status"])
    solution = RKSolution(state["saved_ts"], state["ys"], state["t"], state["y"], status,
                      tree_util.tree_unflatten(structure, state["hits"]),
                      RKStats(state["steps"], state["accepted"], state["rejected"]), state["history"], state["h"])
    return solution if _recorder is None else _recorder.finish(state, solution, initial_saves)


def solve_rk45(rhs, y0, args=None, **kwargs):
    """Dormand--Prince 5(4); see solve_rk for integration options and results."""
    return solve_rk(rhs, y0, args, method="dopri5", **kwargs)


def solve_dopri8(rhs, y0, args=None, **kwargs):
    """Dormand--Prince 8(7); see solve_rk for integration options and results."""
    return solve_rk(rhs, y0, args, method="dopri8", **kwargs)

# GPU field access and JAX compatibility

def _padded_spline(knots, coefficients):
    """Power-of-two kernel view of the same piecewise cubics.

    End extrapolation is expressed about the last knot. Recentring and a
    different reduction order can introduce ordinary floating-point changes.
    This setup is outside jit and intended for fixed, nondifferentiated fields.
    """
    knots, coefficients = np.asarray(knots), np.asarray(coefficients)
    count = len(knots)
    size = 1 << (count-1).bit_length()
    width = 1 << (coefficients.shape[1]-1).bit_length()
    grid = np.full(size, np.inf)
    grid[:count] = knots
    values = np.zeros((size, width, 4))
    values[:count-1, :coefficients.shape[1]] = coefficients
    a, b, c, d = np.moveaxis(coefficients[-1], -1, 0)
    h = knots[-1]-knots[-2]
    shifted = np.stack((a, b+3*a*h, c+2*b*h+3*a*h*h, ((a*h+b)*h+c)*h+d), axis=-1)
    values[count-1:, :coefficients.shape[1]] = shifted
    return jnp.asarray(grid), jnp.asarray(values)


def _select_one(value, index, axis):
    """Bit-preserving scalar/slice selection on a kernel-local array."""
    shape = [1]*value.ndim
    shape[axis] = value.shape[axis]
    mask = jnp.arange(value.shape[axis]).reshape(shape) == index
    dtype = {8: jnp.uint64, 4: jnp.uint32, 2: jnp.uint16, 1: jnp.uint8}[value.dtype.itemsize]
    bits = jax.lax.bitcast_convert_type(value, dtype)
    selected = jnp.max(jnp.where(mask, bits, jnp.zeros_like(bits)), axis=axis, keepdims=True)
    return jax.lax.bitcast_convert_type(selected, value.dtype)


def _kernel_is_finite(value):
    """Exact IEEE finite test using primitives supported by older Pallas.

    JAX 0.7.2's Triton backend cannot lower lax.is_finite. Inspecting the
    exponent preserves NaN/infinity rejection without changing the RHS guard.
    Integer bit operations also avoid assumptions about fast float comparisons.
    """
    if value.dtype == jnp.float64:
        unsigned, exponent = jnp.uint64, 0x7ff0000000000000
    elif value.dtype == jnp.float32:
        unsigned, exponent = jnp.uint32, 0x7f800000
    else:
        raise TypeError("The fused trial finite guard supports float32/float64")
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    mask = jnp.asarray(exponent, dtype=unsigned)
    return jax.lax.bitwise_and(bits, mask) != mask


class _KernelField:
    """Kernel-local Ref adapter; evaluation uses the original field methods."""
    table = sine_coef = None
    def modB_derivatives(self, *args):
        from .boozer import BoozerField
        return BoozerField.modB_derivatives(self, *args)

    def __init__(self, refs, psi0):
        radial, b, flux, profiles, xm, xn = refs
        self.r_knots, self.b_coef = radial[...], b
        self.s_knots, self.profile_coef = flux[...], profiles
        self.xm, self.xn, self.psi0 = xm[...], xn[...], psi0

    def profiles(self, s):
        from .boozer import BoozerField
        value, slope = BoozerField.profiles(self, s)
        return tuple(value[i] for i in range(3)), tuple(slope[i] for i in range(3))


def _kernel_rhs(model, y0, moment, mass, charge):
    """Lower the original checked RHS; adapt unsupported local-array indexing."""
    from .boozer import guiding_center_rhs, _chart

    def physical(y):
        value = guiding_center_rhs(model, y, moment, mass, charge)
        _, r, theta, *_ = _chart(y)
        B = model.modB_derivatives(r, theta, y[2])[0]
        return jnp.where(jnp.isfinite(B) & (B > 0), value, jnp.nan)

    original = jax.make_jaxpr(physical)(y0)
    flat = jax.make_jaxpr(lambda y: _evaluate_inline(original.jaxpr, original.consts, y)[0])(y0)

    def evaluate(y):
        graph = flat.jaxpr
        env = dict(zip(graph.constvars, flat.consts))
        env.update(zip(graph.invars, (y,)))
        def read(v):
            return v.val if isinstance(v, Literal) else env[v]
        for equation in graph.eqns:
            values = [read(v) for v in equation.invars]
            name = equation.primitive.name
            if name == "slice":
                value = values[0]
                for axis, (start, end, size) in enumerate(zip(equation.params["start_indices"],
                        equation.params["limit_indices"], value.shape)):
                    if start != 0 or end != size:
                        if end-start != 1:
                            raise ValueError("Unsupported kernel-local RHS slice")
                        value = _select_one(value, start, axis)
                result = [value]
            elif name == "dynamic_slice":
                value = values[0]
                for axis, (index, size, wanted) in enumerate(zip(values[1:], value.shape,
                        equation.params["slice_sizes"])):
                    if size != wanted:
                        if wanted != 1:
                            raise ValueError("Unsupported kernel-local dynamic RHS slice")
                        value = _select_one(value, jnp.clip(index, 0, size-1), axis)
                result = [value]
            elif name == "concatenate" and len(values) > 2:
                value = jnp.zeros((len(values),), values[0].dtype)
                for i, v in enumerate(values):
                    value = jnp.where(jnp.arange(len(values)) == i, jnp.reshape(v, ()), value)
                result = [value]
            elif name == "le_to":
                # Positive finite knots followed by +inf; NaN queries sort last.
                result = [jnp.less_equal(values[0], values[1]) | jnp.isnan(values[1])]
            elif name == "is_finite":
                result = [_kernel_is_finite(values[0])]
            else:
                result = _bind_primitive(equation.primitive, values, equation.params)
                if not equation.primitive.multiple_results:
                    result = [result]
            env.update(zip(equation.outvars, result))
        return read(graph.outvars[0])
    return evaluate



def _bind_primitive(primitive, values, parameters):
    """Bind an equation using either supported JAX get_bind_params ABI.

    Older JAX returns (subfunctions, keyword_parameters); newer JAX returns
    only keyword_parameters. Keep legacy subfunctions for opaque AD/call
    primitives rather than discarding them while adapting ordinary equations.
    This Python dispatch runs during graph preparation, not each RK stage.
    """
    binding = primitive.get_bind_params(parameters)
    if isinstance(binding, tuple):
        subfunctions, binding = binding
        return primitive.bind(*subfunctions, *values, **binding)
    return primitive.bind(*values, **binding)


def _evaluate_inline(graph, constants, *arguments):
    """Expose jit boundaries without entering custom AD/control-flow calls."""
    from jax.extend.core import Literal
    environment = dict(zip(graph.constvars, constants))
    environment.update(zip(graph.invars, arguments))

    def read(var):
        return var.val if isinstance(var, Literal) else environment[var]

    for equation in graph.eqns:
        values = [read(var) for var in equation.invars]
        if equation.primitive.name in ("jit", "pjit"):
            nested = equation.params["jaxpr"]
            result = _evaluate_inline(nested.jaxpr, nested.consts, *values)
        else:
            result = _bind_primitive(equation.primitive, values, equation.params)
            if not equation.primitive.multiple_results:
                result = [result]
        environment.update(zip(equation.outvars, result))
    return [read(var) for var in graph.outvars]

# Private Biot-Savart GPU view

def _cross(a,b):
    return a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _pack(values):
    result = jnp.zeros((len(values),),values[0].dtype)
    index = jnp.arange(len(values))
    for i,value in enumerate(values):
        result = jnp.where(index==i,value,result)
    return result


def _coil_geometry(field):
    gamma,tangent,current = field.coils.gamma,field.coils.gamma_dash,field.coils.currents
    nc,nq,_ = gamma.shape
    pc,pq = (1<<(size-1).bit_length() for size in (nc,nq))
    weight = current*1e-7
    arrays = tuple(jnp.pad(value[:,:,i].T,((0,pq-nq),(0,pc-nc))).reshape(-1)
        for value in (gamma,tangent*weight[:,None,None]) for i in range(3))
    return arrays,nc,nq


def _coil_field_values(refs,point,nc,nq,jacobian):
    gamma,tangent = tuple(ref[...] for ref in refs[:3]),tuple(ref[...] for ref in refs[3:6])
    pc = 1<<(nc-1).bit_length()
    index = jnp.arange(gamma[0].size)
    mask = (index//pc<nq)&(index%pc<nc)
    r = tuple(point[i]-gamma[i] for i in range(3))
    radius2 = jnp.where(mask,_dot(r,r),1.)
    inverse3 = 1./jnp.sqrt(radius2)**3
    contribution = tuple(jnp.where(mask,value*inverse3,0.) for value in _cross(tangent,r))
    mean = lambda value:jnp.sum(value)/nq
    B = tuple(mean(value) for value in contribution)
    if not jacobian:
        return B,None
    tx,ty,tz = (mean(jnp.where(mask,value*inverse3,0.)) for value in tangent)
    zero = jnp.asarray(0.,point[0].dtype)
    skew = ((zero,-tz,ty),(tz,zero,-tx),(-ty,tx,zero))
    J = tuple(tuple(skew[i][j]-3*mean(contribution[i]*r[j]/radius2)
                    for j in range(3)) for i in range(3))
    return B,J


def _coil_rhs_values(refs,y,nc,nq,model,energy,mass,charge,moment=None):
    scalar = lambda i:_select_one(y,i,0).reshape(())
    point = tuple(scalar(i) for i in range(3))
    B,J = _coil_field_values(refs,point,nc,nq,model=='gc')
    if model=='lorentz':
        velocity = tuple(scalar(i) for i in range(3,6))
        force = _cross(velocity,B)
        return _pack((*velocity,*(charge/mass*value for value in force),
                          jnp.asarray(0.,y.dtype),jnp.asarray(0.,y.dtype)))
    magnitude = jnp.sqrt(_dot(B,B))
    grad = tuple(sum(J[i][j]*B[i]/magnitude for i in range(3)) for j in range(3))
    curl = (J[2][1]-J[1][2],J[0][2]-J[2][0],J[1][0]-J[0][1])
    bg = _cross(B,grad)
    curl_b = tuple(curl[i]/magnitude+bg[i]/magnitude**2 for i in range(3))
    bx = _cross(B,curl_b)
    curvature = tuple(-value/magnitude for value in bx)
    vpar = scalar(3)
    moment = (energy-mass*vpar**2/2)/magnitude if moment is None else moment
    star = tuple(B[i]+vpar*mass/charge*curl_b[i] for i in range(3))
    force = tuple(moment*grad[i]+mass*vpar**2*curvature[i] for i in range(3))
    denominator = _dot(B,star)
    perpendicular = _cross(B,force)
    velocity = tuple(vpar*B[i]/magnitude+perpendicular[i]/denominator/charge for i in range(3))
    acceleration = -_dot(star,force)/denominator*magnitude/mass
    return _pack((*velocity,acceleration))



def _prepare_coils_gpu(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,*,
                         model="gc",method="dopri8",adaptive=True,event=None,
                         length_scale=1.,interpret=False):
    if model not in ("gc","lorentz"):
        raise ValueError("model must be gc or lorentz")
    arrays,nc,nq=_coil_geometry(field)
    scalar=lambda y,i:_select_one(y,i,0).reshape(())
    def factory(refs,point,invariant):
        evaluate=lambda y:_coil_rhs_values(refs,y,nc,nq,model,energy,mass,charge)
        def estimate(endpoint,error,speed):
            position=jnp.sqrt(sum(scalar(error,i)**2 for i in range(3)))/length_scale
            velocity=(jnp.abs(scalar(error,3)) if model=="gc" else
                      jnp.sqrt(sum(scalar(error,i)**2 for i in range(3,6))))/speed
            return jnp.maximum(position,velocity)
        level=(lambda y:scalar(y,0)*0+1.) if event is None else event
        return evaluate,estimate,level
    return _prepare_gpu_solver(arrays,factory,speed,times,dt0,rtol,atol,max_steps,
        state_size=4 if model=="gc" else 8,method=method,adaptive=adaptive,num_warps=2,
        interpret=interpret)

# Private VMEC GPU view

@dataclass
class _VmecData:
    arrays: tuple
    tables: dict


def _pad_vmec(value, shape):
    result = np.zeros(shape, dtype=np.float64)
    result[tuple(slice(0, n) for n in value.shape)] = value
    return jnp.asarray(result)


def _prepare_vmec_data(field):
    """Prepare the same regularized radial values/slopes for a concrete field."""
    from .fields import VMEC_WOUT_PARTNERS
    grids = tuple(np.asarray(x) for x in (field.s_full_grid, field.s_half_grid))
    widths = tuple(1 << (len(x)-1).bit_length() for x in (field.xm_nyq, field.xm))
    radial_width = 1 << (field.ns-1).bit_length()
    arrays = []
    for grid in grids:
        values = np.full(radial_width, np.inf)
        values[:len(grid)] = grid
        arrays.append(jnp.asarray(values))
    for width, modes in zip(widths, ((field.xm_nyq, field.xn_nyq), (field.xm, field.xn))):
        arrays.extend(_pad_vmec(np.asarray(value), (width,)) for value in modes)
    powers = []
    for xm, width, covariant in ((field.xm_nyq, widths[0], False),
                                 (field.xm_nyq, widths[0], True),
                                 (field.xm, widths[1], False)):
        m = np.asarray(xm).astype(int)
        k = np.minimum(m, 2+m % 2)
        if covariant:
            k = np.where(m > 0, np.maximum(k-2, -1), 0)
        powers.append(k)
        arrays.append(_pad_vmec(k, (width,)))
    definitions = dict(field._NYQUIST)
    definitions.update(rmnc=(False, {}, True), zmns=(False, {}, False))
    tables = {}
    for name, (half_grid, options, cosine) in definitions.items():
        geometry = name in ('rmnc', 'zmns')
        power_index = 2 if geometry else 1 if name == 'bsubsmns' else 0
        k = powers[power_index]
        width = widths[1 if geometry else 0]
        grid = grids[int(half_grid)]
        pieces = []
        for partner, is_cosine in ((name, cosine), (VMEC_WOUT_PARTNERS[name], not cosine)):
            table = getattr(field, partner)
            if table is None:
                continue
            table = np.asarray(table)[1:] if half_grid else np.asarray(table)
            scaled = table / np.where(grid > 0, grid, 1.)[:, None]**(k/2)
            if not half_grid:
                m = np.asarray(field.xm if geometry else field.xm_nyq).astype(int)
                scaled[0] = np.where(m > 0, 2*scaled[1]-scaled[2], scaled[0])
                if name == 'bsubsmns':
                    axis = field._bsubs_axis_m1(partner != name)
                    if axis is not None:
                        scaled[0] = np.where(m == 1, np.asarray(axis), scaled[0])
            slope = np.gradient(scaled, axis=0)
            slope[:2] = scaled[1]-scaled[0]
            index = len(arrays)
            arrays.extend((_pad_vmec(scaled, (radial_width, width)),
                           _pad_vmec(slope, (radial_width, width))))
            pieces.append((index, index+1, int(half_grid), len(grid), 6+power_index,
                           4 if geometry else 2, is_cosine))
        tables[name] = tuple(pieces)
    return _VmecData(tuple(arrays), tables)




def _vmec_scalar(value,index):
    return _select_one(value,index,0).reshape(())


def _vmec_chart(y):
    u, w = _vmec_scalar(y, 0), _vmec_scalar(y, 1)
    s = u*u+w*w
    theta = jnp.mod(jnp.arctan2(w, u)+_vmec_scalar(y, 4), 2*jnp.pi)
    return s, jnp.where(_kernel_is_finite(s), theta, s), _vmec_scalar(y, 2)


def _vmec_radial(refs, s, metadata):
    values, slopes, grid_index, count, power_index, _, _ = metadata
    grid, k = refs[grid_index], refs[power_index][...]
    ds = grid[1]-grid[0]
    i = jnp.clip(jnp.floor((s-grid[0])/ds).astype(jnp.int32), 0, count-2)
    u = (s-grid[i])/ds
    a, b = refs[values][i, :], refs[values][i+1, :]
    da, db = refs[slopes][i, :], refs[slopes][i+1, :]

    def basis(value):
        t = jnp.minimum(value, 1.)
        return ((1+2*t)*(1-t)**2, t*(1-t)**2, t*t*(3-2*t),
                t*t*(t-1), jnp.maximum(value-1., 0.))

    h, dh = jax.jvp(basis, (u,), (jnp.ones_like(u),))
    interpolated = h[0]*a+h[1]*da+h[2]*b+h[3]*db+h[4]*db
    derivative = (dh[0]*a+dh[1]*da+dh[2]*b+dh[3]*db+dh[4]*db)/ds

    def power(value):
        q = jnp.sqrt(jnp.maximum(value, jnp.finfo(value.dtype).tiny))
        return jnp.where(k == -1, 1/q, jnp.where(k == 0, 1.,
            jnp.where(k == 1, q, jnp.where(k == 2, q*q, q*q*q))))

    factor, dfactor = jax.jvp(power, (s,), (jnp.ones_like(s),))
    return factor*interpolated, dfactor*interpolated+factor*derivative


def _vmec_series(refs, point, pieces):
    s, theta, phi = point
    total = None
    for metadata in pieces:
        mode_index, cosine = metadata[-2:]
        m, n = refs[mode_index][...], refs[mode_index+1][...]
        angle = m*theta-n*phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        f, df = _vmec_radial(refs, s, metadata)
        if cosine:
            value = (jnp.sum(f*cos), (jnp.sum(df*cos), -jnp.sum((m*f)*sin), jnp.sum((n*f)*sin)))
        else:
            value = (jnp.sum(f*sin), (jnp.sum(df*sin), jnp.sum((m*f)*cos), -jnp.sum((n*f)*cos)))
        total = value if total is None else (total[0]+value[0], tuple(a+b for a, b in zip(total[1], value[1])))
    return total


def _vmec_quantities(refs, point, tables):
    values = {name: _vmec_series(refs, point, tables[name]) for name in
              ('bmnc', 'gmnc', 'bsubsmns', 'bsubumnc', 'bsubvmnc', 'bsupumnc', 'bsupvmnc')}
    magnitude, gradient = values['bmnc']
    sqrtg = values['gmnc'][0]
    covariant = tuple(values[name][0] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc'))
    zero = magnitude*0
    contravariant = zero, values['bsupumnc'][0], values['bsupvmnc'][0]
    jac = tuple(values[name][1] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc'))
    curl = tuple(value/sqrtg for value in
                 (jac[2][1]-jac[1][2], jac[0][2]-jac[2][0], jac[1][0]-jac[0][1]))
    correction = _cross(covariant, gradient)
    curl_b = tuple(curl[i]/magnitude+correction[i]/magnitude**2/sqrtg for i in range(3))
    curvature = tuple(-value*sqrtg/magnitude for value in _cross(contravariant, curl_b))
    return covariant, contravariant, magnitude, gradient, curl_b, curvature, sqrtg


def _vmec_rhs_values(refs, y, tables, mass, charge, energy, moment=None):
    point = _vmec_chart(y)
    covariant, contravariant, magnitude, gradient, curl_b, curvature, sqrtg = _vmec_quantities(refs, point, tables)
    vpar = _vmec_scalar(y, 3)
    moment = (energy-mass*vpar**2/2)/magnitude if moment is None else moment
    star = tuple(contravariant[i]+vpar*mass/charge*curl_b[i] for i in range(3))
    force = tuple(moment*gradient[i]+mass*vpar**2*curvature[i] for i in range(3))
    denominator = _dot(covariant, star)
    perpendicular = _cross(covariant, force)
    velocity = tuple(vpar*contravariant[i]/magnitude+perpendicular[i]/denominator/charge/sqrtg
                     for i in range(3))
    acceleration = -_dot(star, force)/denominator*magnitude/mass
    u, w, s = _vmec_scalar(y, 0), _vmec_scalar(y, 1), point[0]
    half = jnp.where(s > 0, .5/jnp.where(s > 0, s, 1.), 0.)
    c = s/(s+1e-2)
    zero = jnp.asarray(0., y.dtype)
    return _pack((u*half*velocity[0]-(1-c)*w*velocity[1],
                 w*half*velocity[0]+(1-c)*u*velocity[1], velocity[2], acceleration,
                 c*velocity[1], zero, zero, zero))


def _vmec_xyz_values(refs, y, tables):
    point = _vmec_chart(y)
    R = _vmec_series(refs, point, tables['rmnc'])[0]
    Z = _vmec_series(refs, point, tables['zmns'])[0]
    return R*jnp.cos(point[2]), R*jnp.sin(point[2]), Z

def _prepare_vmec_gpu(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,*,
                         method="dopri8",adaptive=True,length_scale=1.,interpret=False):
    prepared=_prepare_vmec_data(field)
    def factory(refs,point,invariant):
        evaluate=lambda y:_vmec_rhs_values(refs,y,prepared.tables,mass,charge,energy)
        def estimate(endpoint,error,speed):
            high,low=_vmec_xyz_values(refs,endpoint,prepared.tables),_vmec_xyz_values(refs,endpoint-error,prepared.tables)
            position=jnp.sqrt(sum((a-b)**2 for a,b in zip(high,low)))/length_scale
            return jnp.maximum(position,jnp.abs(_vmec_scalar(error,3))/speed)
        return evaluate,estimate,lambda y:1.-_vmec_scalar(y,0)**2-_vmec_scalar(y,1)**2
    # Avoid cross-warp reductions for short spectra; keep parallelism for large ones.
    warps = 1 if max(prepared.arrays[2].size, prepared.arrays[4].size) <= 128 else 4
    return _prepare_gpu_solver(prepared.arrays,factory,speed,times,dt0,rtol,atol,max_steps,
        state_size=8,method=method,adaptive=adaptive,num_warps=warps,interpret=interpret)

# Independent GPU particle loops

def _prepare_gpu_solver(arrays, factory, speed, times, dt0, rtol, atol, max_steps, *,
                       state_size=4, method="dopri8", adaptive=True, num_warps=1, interpret=False):
    """Prepare a CUDA solver outside jit; return solve(y0, invariant).

    Fixed and adaptive Dopri5/8 share FSAL stages and standard dense output.
    Adaptive acceptance uses the adapter's error blocks and I control. Save
    times do not constrain steps. Each particle stops independently; there is
    no particle chunking or compaction. Numerical equivalence is subject to
    ordinary floating-point changes from a different compiler execution path.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    if method not in ("dopri5", "dopri8"):
        raise ValueError("The fused orbit method must be dopri5 or dopri8")
    if not interpret and (jax.default_backend() != "gpu" or "cuda" not in jax.devices()[0].client.platform_version.lower()):
        raise ValueError("The experimental fused orbit solver requires CUDA")
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or times[-1] <= 0 or np.any(np.diff(times) < 0)):
        raise ValueError("Save times must be ordered, start at zero and end at a positive time")
    if not (np.isfinite(dt0) and dt0 > 0 and np.isfinite(speed) and speed > 0 and
            np.isfinite(rtol) and rtol >= 0 and np.isfinite(atol) and atol > 0):
        raise ValueError("Require positive finite dt0, speed, atol and finite rtol >= 0")
    if (isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or
            not 0 < max_steps <= np.iinfo(np.int32).max):
        raise ValueError("max_steps must be a positive int32 work bound")
    rows, weights, order = (_A, _ERROR, 5) if method == "dopri5" else (_A8, _ERROR8, 8)
    stages = len(weights)
    bank_size = 1 << (stages-1).bit_length()
    matrix = np.zeros((bank_size, bank_size))
    for i, row in enumerate(rows, 1):
        matrix[i, :len(row)] = row
    coefficients = jnp.asarray(matrix)
    error_weights = jnp.asarray(tuple(weights)+(0.,)*(bank_size-stages))
    if method == "dopri5":
        dense_matrix = np.zeros((1, bank_size))
        dense_matrix[0, :stages] = np.asarray(_MID)
    else:
        dense_matrix = np.zeros((len(_DENSE8[0]), bank_size))
        dense_matrix[:, :stages] = np.asarray(_DENSE8).T
    dense_coefficients = jnp.asarray(dense_matrix)
    tmax = float(times[-1])
    # Load float64 scalars explicitly. Nested Pallas loops must not infer
    # their device scalar precision from weak Python constant initializers.
    scalars = jnp.asarray([0., dt0, tmax, 8*np.finfo(np.float64).eps*tmax,
                           speed, rtol+atol], jnp.float64)
    inputs = tuple(arrays)+(coefficients, error_weights, dense_coefficients, jnp.asarray(times), scalars)
    nsave = len(times)

    def solve(y0, moment):
        if y0.shape != (state_size,) or y0.dtype != jnp.float64:
            raise ValueError("The fused orbit solver requires a float64 power-of-two state")
        # Validate the birth once with ordinary JAX, outside the persistent
        # loop. Invalid inputs must not spend max_steps rejecting NaN trials.
        initial_level = jnp.asarray(1., y0.dtype)
        initial_status = jnp.where(jnp.isfinite(y0).all() & jnp.isfinite(moment).all() & jnp.isfinite(initial_level),
            jnp.where(initial_level == 0, 1, 0), 4).astype(jnp.int32)

        def kernel(yref, mref, initial_status_ref, *refs):
            (*field_refs, aref, eref, dref, tsref, paramsref,
             ysout, tsout, yout, tout, hout, statsout, statusout) = refs
            point = yref[...]
            zero, initial_h, final_t, roundoff, particle_speed, budget = (paramsref[i] for i in range(6))
            evaluate, estimate, level = factory(field_refs, point, mref[...])

            def finite(value):
                return jnp.min(_kernel_is_finite(value).astype(jnp.int32)) > 0

            def dense(y, endpoint, bank, fraction):
                if method == "dopri5":
                    # Same quartic extension as rk._rk45_dense; the padded
                    # bank's last row is not the final FSAL stage.
                    first = _select_one(bank, 0, 0).reshape(state_size)
                    last = _select_one(bank, stages-1, 0).reshape(state_size)
                    midpoint = jnp.sum(dref[0, :][:, None]*bank, axis=0)
                    delta = endpoint-y
                    a = 2*(last-first)-8*delta+16*midpoint
                    b = 5*first-3*last+14*delta-32*midpoint
                    c = last-4*first-5*delta+16*midpoint
                    value = y+fraction*(first+fraction*(c+fraction*(b+fraction*a)))
                    return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, value))
                weights = jnp.zeros((bank_size,), point.dtype)
                for power in range(dense_matrix.shape[0]):
                    weights = weights*fraction+dref[power, :]
                value = y+jnp.sum((weights*fraction)[:, None]*bank, axis=0)
                return jnp.where(fraction == 0, y, jnp.where(fraction == 1, endpoint, value))

            def clear(i, _):
                ysout[i, :] = jnp.full((state_size,), jnp.inf, point.dtype)
                tsout[i] = tsref[0]+jnp.inf
                return ()

            jax.lax.fori_loop(0, nsave, clear, ())
            initial_level = level(point)
            status = jnp.where(_kernel_is_finite(initial_level),
                jnp.where(initial_level == 0, 1, initial_status_ref[...]), 4).astype(jnp.int32)

            def initial_save(i):
                ysout[i, :] = point
                tsout[i] = zero
                return i+1

            index = jax.lax.while_loop(lambda i: (status == 0) & (i < nsave) &
                (tsref[jnp.minimum(i, nsave-1)] == 0), initial_save, jnp.asarray(0, jnp.int32))
            indexbank = jnp.arange(bank_size)[:, None]
            # Time, state, FSAL derivative, proposal, event value, status,
            # next save index, trials, accepted trials, rejected trials.
            # Evaluate the first derivative in the stage loop so the compiler
            # sees one RHS body. Cache it even when the first trial is rejected.
            initial = (zero, point, jnp.zeros_like(point), initial_h, initial_level, status, index,
                       jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))

            def advance(carry):
                t, y, f0, next_h, old_value, status, index, trials, accepted, rejected = carry
                remaining = final_t-t
                h = jnp.minimum(next_h if adaptive else initial_h, remaining)
                # 0 <= t <= final_t, so max(abs(t), abs(final_t)) is final_t.
                h = jnp.where(remaining-h <= roundoff, remaining, h)
                end_t = jnp.where(h == remaining, final_t, t+h)
                if not adaptive:
                    end_t = jnp.minimum((accepted+1)*initial_h, final_t)
                    h = end_t-t
                bank = jnp.where(indexbank == 0, h*f0[None, :], jnp.zeros((bank_size, state_size), point.dtype))

                def stage(i, stage_carry):
                    bank, _, first = stage_carry
                    yi = y+jnp.sum(aref[i, :][:, None]*bank, axis=0)
                    fi = evaluate(yi)
                    return jnp.where(indexbank == i, h*fi[None, :], bank), fi, jnp.where(i == 0, fi, first)

                bank, f1, f0 = jax.lax.fori_loop((trials != 0).astype(jnp.int32), np.int32(stages), stage, (bank, f0, f0))
                endpoint = y+jnp.sum(aref[stages-1, :][:, None]*bank, axis=0)
                embedded = jnp.sum(eref[...][:, None]*bank, axis=0)
                error = estimate(endpoint, embedded, particle_speed)/budget if adaptive else zero
                new_value = level(endpoint)
                valid = (finite(endpoint) & finite(bank) & finite(f1) & finite(embedded) &
                    _kernel_is_finite(error) & (error >= 0) & _kernel_is_finite(new_value))
                accept = valid & ((error <= 1) | (not adaptive)) & (end_t > t)
                crossing = accept & (((old_value > 0) != (new_value > 0)) | (new_value == 0))
                factor = jnp.where(valid, jnp.clip(.9*jnp.maximum(error, 1e-16)**(-1/order), .2, 5.), .2)
                proposed = h*jnp.where(accept, factor, jnp.minimum(factor, .9))
                stuck = (end_t <= t) | (~accept & ((h <= 0) | (t+proposed <= t)))
                if not adaptive:
                    stuck = stuck | ~valid
                new_status = jnp.where(stuck, jnp.where(valid, 3, 4), 0).astype(jnp.int32)

                def bisect(root_carry):
                    lo, hi, i, valid_root = root_carry
                    mid = (lo+hi)/2
                    value = level(dense(y, endpoint, bank, mid))
                    same = (value > 0) == (old_value > 0)
                    return (jnp.where(same, mid, lo), jnp.where(same, hi, mid), i+1,
                            valid_root & _kernel_is_finite(value))

                lo, hi, _, valid_root = jax.lax.while_loop(lambda c: crossing & c[3] & (c[2] < 52),
                    bisect, (zero, zero+1., jnp.asarray(0, jnp.int32), jnp.asarray(True)))
                fraction = (lo+hi)/2
                stop_t = jnp.where(crossing, t+fraction*h, end_t)
                stop_y = jax.lax.cond(crossing,
                    lambda _: dense(y, endpoint, bank, fraction), lambda _: endpoint, None)
                # A nonfinite interpolant rejects the crossing and retries from
                # the previous state with .2*h, matching the deferred root path.
                root_invalid = crossing & ~valid_root
                accept = accept & ~root_invalid
                proposed = jnp.where(root_invalid, .2*h, proposed)
                root_stuck = (h <= 0) | (t+proposed <= t)
                new_status = jnp.where(crossing & valid_root, 1,
                    jnp.where(root_invalid, jnp.where(root_stuck, 4, 0), new_status)).astype(jnp.int32)

                def save(i):
                    ysout[i, :] = dense(y, endpoint, bank, (tsref[i]-t)/h)
                    tsout[i] = tsref[i]
                    return i+1

                index = jax.lax.while_loop(lambda i: accept & (i < nsave) &
                    (tsref[jnp.minimum(i, nsave-1)] <= stop_t), save, index)
                return (jnp.where(accept, stop_t, t), jnp.where(accept, stop_y, y),
                    jnp.where(accept, f1, f0), proposed, jnp.where(accept, new_value, old_value),
                    new_status, index, trials+1, accepted+accept.astype(jnp.int32),
                    rejected+(~accept).astype(jnp.int32))

            result = jax.lax.while_loop(lambda c: (c[0] < final_t) & (c[5] == 0) &
                (c[7] < max_steps), advance, initial)
            t, y, _, h, _, status, _, trials, accepted, rejected = result
            status = jnp.where((t < final_t) & (status == 0), 2, status)
            yout[...], tout[...], hout[...] = y, t, h
            statsout[0], statsout[1], statsout[2] = trials, accepted, rejected
            statusout[...] = status

        shapes = (jax.ShapeDtypeStruct((nsave, state_size), y0.dtype), jax.ShapeDtypeStruct((nsave,), y0.dtype),
            jax.ShapeDtypeStruct((state_size,), y0.dtype), jax.ShapeDtypeStruct((), y0.dtype),
            jax.ShapeDtypeStruct((), y0.dtype), jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32))
        ys, saved, y, t, h, counts, status = pl.pallas_call(kernel, out_shape=shapes, grid=(),
            interpret=interpret, compiler_params=plt.CompilerParams(num_warps=num_warps))(y0, moment, initial_status, *inputs)
        return RKSolution(saved, ys, t, y, status, status == 1,
            RKStats(counts[0], counts[1], counts[2]), jnp.zeros((0,), y0.dtype), h)

    solve.method = method
    return solve

# Boozer GPU preparation

def _prepare_boozer_gpu(field, mass, charge, speed, times, dt0, rtol, atol, max_steps, *,
                         method="dopri8", adaptive=True, interpret=False):
    if field.table is not None or field.sine_coef is not None:
        raise ValueError("The CUDA adapter requires a fixed cosine Fourier Boozer field")
    radial, b = _padded_spline(field.r_knots, field.b_coef)
    flux, profiles = _padded_spline(field.s_knots, field.profile_coef)
    modes = b.shape[1]
    arrays = (radial, b, flux, profiles, jnp.pad(field.xm, (0,modes-len(field.xm))),
              jnp.pad(field.xn, (0,modes-len(field.xn))))
    scalar = lambda y,i: _select_one(y,i,0).reshape(())
    def factory(refs, point, moment):
        model = _KernelField(refs, float(field.psi0))
        evaluate = _kernel_rhs(model, point, moment, mass, charge)
        def estimate(endpoint, error, speed):
            a,b,c,d = (scalar(error,i) for i in range(4))
            return jnp.maximum(jnp.sqrt(a*a+b*b),jnp.maximum(jnp.abs(c),jnp.abs(d)/speed))
        return evaluate, estimate, lambda y: 1.-scalar(y,0)**2-scalar(y,1)**2
    return _prepare_gpu_solver(arrays,factory,speed,times,dt0,rtol,atol,max_steps,
                              method=method,adaptive=adaptive,interpret=interpret)

# Existing Tracing integration

class _ParticleParameters(NamedTuple):
    mass: float
    charge: float
    energy: float
    total_speed: float


def prepare_solver(field, *, mass, charge, energy, times, dt0, model="gc", method="dopri8",
                   adaptive=True, rtol=1e-9, atol=1e-9, controller="i", pcoeff=.4, icoeff=.3,
                   max_steps=1_000_000, length_scale=1., execution="jax", interpret=False):
    """Prepare ``solve(state) -> RKSolution`` for fixed-field forward tracing.

    Cartesian GC states are (x,y,z,vpar), Lorentz states (x,y,z,vx,vy,vz).
    VMEC GC uses the existing five-component axis-regular chart. VMEC stops
    at s=1. The CUDA path uses independent particle programs and I control;
    the portable path supports I/PI. Use jit(vmap(solve)) for a batch.
    This function prepares fixed fields outside jit. It defines no AD rule.
    """
    import numpy as np
    from .fields import BiotSavart, Vmec
    from .dynamics import GuidingCenter, Lorentz, _axis_regular, _from_axis_regular
    from .electric_field import Electric_field_zero
    if execution not in ("jax", "cuda") or model not in ("gc", "lorentz"):
        raise ValueError("Require execution jax/cuda and model gc/lorentz")
    if not all(np.isfinite(v) for v in (mass,charge,energy,length_scale)) or min(mass,energy,length_scale)<=0 or charge==0:
        raise ValueError("Require positive finite mass, energy, length_scale and nonzero charge")
    speed=float(np.sqrt(2*energy/mass))
    vmec=isinstance(field,Vmec)
    if vmec and model!='gc':
        raise ValueError("VMEC native integration supports guiding centers")
    size=5 if vmec else 4 if model=='gc' else 6
    if execution=='cuda':
        if controller!='i':
            raise ValueError("CUDA execution supports the I controller")
        if vmec:
            prepare_orbit_solver = _prepare_vmec_gpu
        elif isinstance(field,BiotSavart):
            prepare_orbit_solver = _prepare_coils_gpu
        else:
            raise ValueError("CUDA Cartesian fields must be BiotSavart")
        options={} if vmec else dict(model=model)
        kernel=prepare_orbit_solver(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,
            method=method,adaptive=adaptive,length_scale=length_scale,interpret=interpret,**options)
        padded=8 if size in (5,6) else 4
        def solve(y):
            result=kernel(jnp.pad(y,(0,padded-size)),jnp.zeros((),y.dtype))
            return result._replace(ys=result.ys[:,:size],y=result.y[:size])
        return solve
    particle=_ParticleParameters(mass,charge,energy,speed)
    if model=='gc':
        rhs=lambda t,y,args:GuidingCenter(t,y,(field,particle,Electric_field_zero()))
        rhs=_axis_regular(rhs) if vmec else rhs
    else:
        rhs=lambda t,y,args:Lorentz(t,y,(field,particle))
    position=(lambda y:field.to_xyz(_from_axis_regular(y)[:3])) if vmec else None
    norm=OrbitErrorNorm(rtol,atol,speed,length_scale,1 if model=='gc' else 3,position)
    event=(lambda t,y,args:1.-y[0]**2-y[1]**2) if vmec else None
    return lambda y:solve_rk(rhs,y,t1=times[-1],dt0=dt0,ts=times,method=method,adaptive=adaptive,
        rtol=rtol,atol=atol,error_norm=norm,controller=controller,pcoeff=pcoeff,icoeff=icoeff,
        event=event,max_steps=max_steps)

class _NativeOrbitSolution(NamedTuple):
    ys: object
    ts: object
    event_mask: object
    result: object


def _configure_tracing(trace, integrator, model, solver, native_error_control, orbit_length_scale,
              orbit_speed_scale, native_controller, native_pcoeff, native_icoeff):
    if integrator not in ("diffrax", "essos_rk45", "essos_dopri8"):
        raise ValueError("integrator must be 'diffrax', 'essos_rk45' or 'essos_dopri8'")
    if integrator != "diffrax":
        if model not in ('GuidingCenter', 'GuidingCenterAdaptative', 'FullOrbit', 'FullOrbitAdaptative'):
            raise ValueError("native integrators currently support collisionless guiding-center and full-orbit models only")
        if solver is not None:
            raise ValueError("solver selects a Diffrax solver; omit it when using a native integrator")
    trace.integrator = integrator
    if native_error_control not in ("physics", "standard"):
        raise ValueError("native_error_control must be 'physics' or 'standard'")
    trace.native_error_control = native_error_control
    if native_controller not in ("i", "pi"):
        raise ValueError("native_controller must be 'i' or 'pi'")
    trace.native_controller, trace.native_pcoeff, trace.native_icoeff = native_controller, native_pcoeff, native_icoeff
    trace.orbit_length_scale, trace.orbit_speed_scale = orbit_length_scale, orbit_speed_scale


def _trace_error_norm(trace, *, exterior=False, flag=False):
    """Physical blocks only; keep coordinates/diagnostics out of the RK map."""
    from .dynamics import _from_axis_regular
    if trace.native_error_control == "standard":
        return None
    position_map = None
    if trace._axis_regular and not exterior:
        def position_map(y):
            state = y[:-1] if flag else y
            return trace.field.to_xyz(_from_axis_regular(state)[:3])
    return OrbitErrorNorm(
        rtol=trace.rtol, atol=trace.atol, length_scale=trace.orbit_length_scale,
        speed_scale=trace.particles.total_speed if trace.orbit_speed_scale is None else trace.orbit_speed_scale,
        velocity_components=3 if trace.model.startswith("FullOrbit") else 1,
        position_map=position_map)


def _solve_tracing(trace, terms, controller, t0, y0, args, event, throw, exterior, flag):
    import diffrax
    solution = solve_rk(
        terms, y0, args, t0=t0, t1=trace.maxtime, dt0=trace.timestep,
        ts=jnp.clip(trace.times, t0, trace.maxtime), adaptive=controller,
        rtol=trace.rtol, atol=trace.atol, event=event.cond_fn,
        root_find=event.root_finder is not None, max_steps=trace.max_steps,
        method="dopri8" if trace.integrator == "essos_dopri8" else "dopri5",
        error_norm=_trace_error_norm(trace, exterior=exterior, flag=flag),
        controller=trace.native_controller, pcoeff=trace.native_pcoeff, icoeff=trace.native_icoeff,
        adjoint=trace.native_adjoint, adjoint_checkpoint_steps=trace.native_adjoint_checkpoint_steps,
        adjoint_max_accepted_steps=trace.native_adjoint_max_accepted_steps,
        adjoint_parameter_accumulation=trace.native_adjoint_parameter_accumulation)
    ys = solution.ys
    if throw:
        import equinox as eqx
        ys = eqx.error_if(ys, solution.status > RKStatus.EVENT, "native RK solve failed; inspect solve_rk status/stats")
    result = diffrax.RESULTS.where(solution.status == RKStatus.SUCCESS, diffrax.RESULTS.successful,
               diffrax.RESULTS.where(solution.status == RKStatus.EVENT, diffrax.RESULTS.event_occurred,
                                     diffrax.RESULTS.max_steps_reached))
    return _NativeOrbitSolution((ys, solution.y[None]), (solution.ts, solution.t[None]),
                                solution.event_mask, result)

# Boozer tracing integration

def _boozer_error_norm(error, speed, rtol, atol):
    """Independent flux-radius, toroidal-radian and parallel-speed budgets."""
    return jnp.maximum(jnp.linalg.norm(error[:2]), jnp.maximum(
        jnp.abs(error[2]), jnp.abs(error[3])/speed)) / (rtol + atol)


def trace_boozer_orbits(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                  method, adaptive, controller, pcoeff, icoeff, max_steps, *, execution="jax"):
    """Select portable JAX or fixed-field CUDA execution without changing physics.

    CUDA supports fixed steps and adaptive I control for cosine Fourier fields. The portable
    path also supports fixed steps, PI control, sine modes and angle tables.
    The choice is explicit: no backend, field layout or controller is changed.
    """
    if execution not in ("jax", "cuda"):
        raise ValueError("native_execution must be 'jax' or 'cuda'")
    solver = None
    if execution == "cuda":
        if controller != "i":
            raise ValueError("CUDA native orbits require I control")
        prepare_orbit_solver = _prepare_boozer_gpu
        solver = prepare_orbit_solver(field, mass, charge, float(speed), times,
                                      dt0, rtol, atol, max_steps, method=method, adaptive=adaptive)
    return _boozer_batch(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                          method, adaptive, controller, pcoeff, icoeff, max_steps, solver)


@partial(jax.jit, static_argnames=("method", "adaptive", "controller", "max_steps", "orbit_solver"))
def _boozer_batch(field, y0, mu0, mass, charge, speed, times, dt0, rtol, atol,
                   method, adaptive, controller, pcoeff, icoeff, max_steps, orbit_solver=None):
    """One shared native RK solve per orbit, independent of the save grid."""
    from .boozer import _chart, guiding_center_rhs

    def one(y0, mu):
        def magnetic_field(y):
            _, r, theta, *_ = _chart(y)
            return field.modB_derivatives(r, theta, y[2])[0]

        def rhs(t, y, args):
            del t, args
            value = guiding_center_rhs(field, y, mu, mass, charge)
            B = magnetic_field(y)
            return jnp.where(jnp.isfinite(B) & (B > 0), value, jnp.nan)

        def error_norm(y0, y1, error):
            del y0, y1
            # This field supplies |B| and profiles, not a Cartesian embedding.
            # Use a unit flux-radius chart, one radian for zeta and birth speed
            # for v_parallel. No scale grows with angle winding or vanishes at
            # a parallel turning point; these are chart errors, not metres.
            return _boozer_error_norm(error, speed, rtol, atol)

        if orbit_solver is None:
            solution = solve_rk(rhs, y0, t1=times[-1], dt0=dt0, ts=times,
                method=method, adaptive=adaptive, rtol=rtol, atol=atol,
                error_norm=error_norm, controller=controller, pcoeff=pcoeff, icoeff=icoeff,
                max_steps=max_steps, event=lambda t, y, args: 1. - y[0]**2 - y[1]**2)
        else:
            solution = orbit_solver(y0, mu)

        def output(y):
            s, _, theta, *_ = _chart(y)
            B = magnetic_field(y)
            energy = .5*y[3]**2 + mu*B
            state = jnp.array([s, theta, y[2], y[3], jnp.sqrt(jnp.maximum(2*energy, 0.))])
            valid = jnp.isfinite(state).all() & jnp.isfinite(energy) & (energy > 0) & (B > 0)
            return state, energy, valid

        # Hold an orbit at the root-found loss point or last accepted state
        # after failure. The solver itself leaves unreached saves infinite.
        sampled = jnp.where(jnp.isfinite(solution.ys).all(axis=1)[:, None], solution.ys, solution.y)
        values, energies, valid = jax.vmap(output)(sampled)
        _, end_energy, end_valid = output(solution.y)
        birth, birth_energy, _ = output(y0)
        output_failed = ~jnp.all(valid) | ~end_valid
        failed = (solution.status > RKStatus.EVENT) | output_failed
        lost = (solution.status == RKStatus.EVENT) & ~failed
        first_bad = jnp.min(jnp.where(valid, jnp.inf, times))
        failed_time = jnp.minimum(solution.t, first_bad)

        def hold(carry, item):
            previous, finite_prefix = carry
            value, finite = item
            finite_prefix = finite_prefix & finite
            previous = jnp.where(finite_prefix, value, previous)
            return (previous, finite_prefix), previous

        _, values = jax.lax.scan(hold, (birth, jnp.asarray(True)), (values, valid))
        energy_error = jnp.maximum(jnp.max(jnp.abs(energies/birth_energy - 1)),
                                    jnp.abs(end_energy/birth_energy - 1))
        return (values, jnp.where(lost, solution.t, -1.), jnp.where(failed, failed_time, -1.),
                jnp.where(failed, jnp.inf, energy_error), jnp.stack(solution.stats),
                jnp.where(output_failed, RKStatus.NONFINITE, solution.status))

    return jax.vmap(one)(y0, mu0)
