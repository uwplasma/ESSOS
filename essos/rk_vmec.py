"""Private fixed-field VMEC GPU view; public field classes remain untouched."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from essos.fields import VMEC_WOUT_PARTNERS
from .rk_gpu_utils import _select_one, _kernel_is_finite


@dataclass
class Prepared:
    arrays: tuple
    tables: dict


def padded(value, shape):
    result = np.zeros(shape, dtype=np.float64)
    result[tuple(slice(0, n) for n in value.shape)] = value
    return jnp.asarray(result)


def prepare(field):
    """Prepare the same regularized radial values/slopes for a concrete field."""
    grids = tuple(np.asarray(x) for x in (field.s_full_grid, field.s_half_grid))
    widths = tuple(1 << (len(x)-1).bit_length() for x in (field.xm_nyq, field.xm))
    radial_width = 1 << (field.ns-1).bit_length()
    arrays = []
    for grid in grids:
        values = np.full(radial_width, np.inf)
        values[:len(grid)] = grid
        arrays.append(jnp.asarray(values))
    for width, modes in zip(widths, ((field.xm_nyq, field.xn_nyq), (field.xm, field.xn))):
        arrays.extend(padded(np.asarray(value), (width,)) for value in modes)
    powers = []
    for xm, width, covariant in ((field.xm_nyq, widths[0], False),
                                 (field.xm_nyq, widths[0], True),
                                 (field.xm, widths[1], False)):
        m = np.asarray(xm).astype(int)
        k = np.minimum(m, 2+m % 2)
        if covariant:
            k = np.where(m > 0, np.maximum(k-2, -1), 0)
        powers.append(k)
        arrays.append(padded(k, (width,)))
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
            arrays.extend((padded(scaled, (radial_width, width)),
                           padded(slope, (radial_width, width))))
            pieces.append((index, index+1, int(half_grid), len(grid), 6+power_index,
                           4 if geometry else 2, is_cosine))
        tables[name] = tuple(pieces)
    return Prepared(tuple(arrays), tables)


from .rk_coils import cross, dot, pack


def scalar(value,index):
    return _select_one(value,index,0).reshape(())


def chart(y):
    u, w = scalar(y, 0), scalar(y, 1)
    s = u*u+w*w
    theta = jnp.mod(jnp.arctan2(w, u)+scalar(y, 4), 2*jnp.pi)
    return s, jnp.where(_kernel_is_finite(s), theta, s), scalar(y, 2)


def radial(refs, s, metadata):
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


def series(refs, point, pieces):
    s, theta, phi = point
    total = None
    for metadata in pieces:
        mode_index, cosine = metadata[-2:]
        m, n = refs[mode_index][...], refs[mode_index+1][...]
        angle = m*theta-n*phi
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        f, df = radial(refs, s, metadata)
        if cosine:
            value = (jnp.sum(f*cos), (jnp.sum(df*cos), -jnp.sum((m*f)*sin), jnp.sum((n*f)*sin)))
        else:
            value = (jnp.sum(f*sin), (jnp.sum(df*sin), jnp.sum((m*f)*cos), -jnp.sum((n*f)*cos)))
        total = value if total is None else (total[0]+value[0], tuple(a+b for a, b in zip(total[1], value[1])))
    return total


def quantities(refs, point, tables):
    values = {name: series(refs, point, tables[name]) for name in
              ('bmnc', 'gmnc', 'bsubsmns', 'bsubumnc', 'bsubvmnc', 'bsupumnc', 'bsupvmnc')}
    magnitude, gradient = values['bmnc']
    sqrtg = values['gmnc'][0]
    covariant = tuple(values[name][0] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc'))
    zero = magnitude*0
    contravariant = zero, values['bsupumnc'][0], values['bsupvmnc'][0]
    jac = tuple(values[name][1] for name in ('bsubsmns', 'bsubumnc', 'bsubvmnc'))
    curl = tuple(value/sqrtg for value in
                 (jac[2][1]-jac[1][2], jac[0][2]-jac[2][0], jac[1][0]-jac[0][1]))
    correction = cross(covariant, gradient)
    curl_b = tuple(curl[i]/magnitude+correction[i]/magnitude**2/sqrtg for i in range(3))
    curvature = tuple(-value*sqrtg/magnitude for value in cross(contravariant, curl_b))
    return covariant, contravariant, magnitude, gradient, curl_b, curvature, sqrtg


def rhs_values(refs, y, tables, mass, charge, energy):
    point = chart(y)
    covariant, contravariant, magnitude, gradient, curl_b, curvature, sqrtg = quantities(refs, point, tables)
    vpar = scalar(y, 3)
    moment = (energy-mass*vpar**2/2)/magnitude
    star = tuple(contravariant[i]+vpar*mass/charge*curl_b[i] for i in range(3))
    force = tuple(moment*gradient[i]+mass*vpar**2*curvature[i] for i in range(3))
    denominator = dot(covariant, star)
    perpendicular = cross(covariant, force)
    velocity = tuple(vpar*contravariant[i]/magnitude+perpendicular[i]/denominator/charge/sqrtg
                     for i in range(3))
    acceleration = -dot(star, force)/denominator*magnitude/mass
    u, w, s = scalar(y, 0), scalar(y, 1), point[0]
    half = jnp.where(s > 0, .5/jnp.where(s > 0, s, 1.), 0.)
    c = s/(s+1e-2)
    zero = jnp.asarray(0., y.dtype)
    return pack((u*half*velocity[0]-(1-c)*w*velocity[1],
                 w*half*velocity[0]+(1-c)*u*velocity[1], velocity[2], acceleration,
                 c*velocity[1], zero, zero, zero))


def xyz_values(refs, y, tables):
    point = chart(y)
    R = series(refs, point, tables['rmnc'])[0]
    Z = series(refs, point, tables['zmns'])[0]
    return R*jnp.cos(point[2]), R*jnp.sin(point[2]), Z

def prepare_orbit_solver(field,mass,charge,energy,speed,times,dt0,rtol,atol,max_steps,*,
                         method="dopri8",adaptive=True,length_scale=1.,interpret=False):
    from .rk_gpu import prepare_gpu_solver
    prepared=prepare(field)
    def factory(refs,point,invariant):
        evaluate=lambda y:rhs_values(refs,y,prepared.tables,mass,charge,energy)
        def estimate(endpoint,error,speed):
            high,low=xyz_values(refs,endpoint,prepared.tables),xyz_values(refs,endpoint-error,prepared.tables)
            position=jnp.sqrt(sum((a-b)**2 for a,b in zip(high,low)))/length_scale
            return jnp.maximum(position,jnp.abs(scalar(error,3))/speed)
        return evaluate,estimate,lambda y:1.-scalar(y,0)**2-scalar(y,1)**2
    return prepare_gpu_solver(prepared.arrays,factory,speed,times,dt0,rtol,atol,max_steps,
        state_size=8,method=method,adaptive=adaptive,num_warps=4,interpret=interpret)
