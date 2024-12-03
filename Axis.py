import jax.numpy as jnp

def r_axis(raxiscc, nfp, tor_angle, raxiscs = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate of the magnetic axis, returns the r coordinate
    at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.

    Returns:
        float: The r coordinate of the magnetic axis at the given toroidal angle.
    """
    r = 0
    for m in range(len(raxiscc)):
        r += raxiscc[m] * jnp.cos(nfp * m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r += raxiscs[m] * jnp.sin(nfp * m * tor_angle)
    return r

def z_axis(zaxiscs, nfp, tor_angle, zaxiscc = None):
    """Given the sine (and optionally cosine) Fourier components
    of the z coordinate of the magnetic axis, returns the z coordinate
    at the given toroidal angle.

    Args:
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        zaxiscc (list(float), optional): Cosine Fourier components of the r coordinate. Defaults to None. Defaults to None.

    Returns:
        float: The z coordinate of the magnetic axis at the given toroidal angle.
    """
    z = 0
    for m in range(len(zaxiscs)):
        z += zaxiscs[m] * jnp.sin(nfp * m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z += zaxiscc[m] * jnp.cos(nfp * m * tor_angle)
    return z

def r_axis_prime(raxiscc, nfp, tor_angle, raxiscs = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate of the magnetic axis, returns the derivative
    of the r coordinate at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.

    Returns:
        float: The derivative of the r component of the magnetic axis with respect to the toroidal angle at the given toroidal angle.
    """
    r_ = 0
    for m in range(len(raxiscc)):
        r_ += -nfp * m * raxiscc[m] * jnp.sin(nfp * m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r_ += nfp * m * raxiscs[m] * jnp.cos(nfp * m * tor_angle)
    return r_

def z_axis_prime(zaxiscs, nfp, tor_angle, zaxiscc = None):
    """Given the sine (and optionally cosine) Fourier components
    of the z coordinate of the magnetic axis, returns the derivative
    of the z coordinate at the given toroidal angle.

    Args:
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the z coordinate of the magnetic axis.
        zaxiscc (list(float), optional): Cosine Fourier components of the z coordinate. Defaults to None.

    Returns:
        float: The derivative of the z component of the magnetic axis with respect to the toroidal angle at the given toroidal angle.
    """
    z_ = 0
    for m in range(len(zaxiscs)):
        z_ += nfp * m * zaxiscs[m] * jnp.cos(nfp * m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z_ += -nfp * m * zaxiscc[m] * jnp.sin(nfp * m * tor_angle)
    return z_

def axis_del_phi(raxiscc, zaxiscs, nfp, tor_angle, raxiscs = None, zaxiscc = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate and the sine (and optionally cosine) Fourier
    components of the z coordinate of the magnetic axis, returns the
    tangent vector to the magnetic axis at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (_type_): Toroidal angle at which to evaluate the derivatives of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.
        zaxiscc (list(float), optional): Cosine Fourier components of the z coordinate. Defaults to None.

    Returns:
        3_list(float): The tangent vector of the magnetic axis at the given toroidal angle.
    """
    rax = r_axis(raxiscc, nfp, tor_angle, raxiscs = raxiscs)
    rax_ = r_axis_prime(raxiscc, nfp, tor_angle, raxiscs = raxiscs)
    zax_ = z_axis_prime(zaxiscs, nfp, tor_angle, zaxiscc = zaxiscc)
    sinterm = jnp.sin(tor_angle)
    costerm = jnp.cos(tor_angle)
    return jnp.array([rax_*costerm - rax*sinterm, rax_*sinterm + rax*costerm, zax_])

""" Potentially useful parts from Cyclone for reading in and creating axis formulations
# Feel free to scrap, grab pieces from, or whatever else

def axis_from_vmec_input(file):
    import f90nml
    nml = f90nml.read(file)['indata']
    raxiscc = nml['raxis_cc']
    zaxiscs = -np.array(nml['zaxis_cs'])
    nfp = nml['nfp']
    stellsym = not nml['lasym']
    raxiscs = None
    zaxiscc = None
    if not stellsym:
        raxiscs = -np.array(nml['raxis_cs'])
        zaxiscc = nml['zaxis_cc']
    return nfp, stellsym, raxiscc, zaxiscs, raxiscs, zaxiscc

def axis_from_wout(file):
    from scipy.io import netcdf_file
    f = netcdf_file(file, mmap=False)
    raxiscc = f.variables['raxis_cc'][()]
    zaxiscs = -f.variables['zaxis_cs'][()]
    nfp = f.variables['nfp'][()]
    stellsym = not bool(f.variables['lasym__logical__'][()])
    raxiscs = None
    zaxiscc = None
    if not stellsym:
        raxiscs = -f.variables['raxis_cs'][()]
        zaxiscc = f.variables['zaxis_cc'][()]
    return nfp, stellsym, raxiscc, zaxiscs, raxiscs, zaxiscc

def axis_from_list():
    return None

def axis_from_dict():
    return None

def import_axis(axis):
    if 'input.' in axis:
        nfp, stellsym, raxiscc, zaxiscs, raxiscs, zaxiscc = axis_from_vmec_input(axis)
    elif 'wout' in axis and '.nc' in axis:
        nfp, stellsym, raxiscc, zaxiscs, raxiscs, zaxiscc = axis_from_wout(axis)
    elif axis is None or axis == 'default':
        raise TypeError('Default not implemented yet')
    elif type(axis) == list or type(axis) == dict:
        # Must also specify nfp and stellsym in list/dict
        raise TypeError('User input lists or dictionaries for axis parameters are not presently supported.')
    else:
        raise TypeError('Axis could not be interpreted as VMEC input. file or VMEC wout.nc file.')
    def axis_function(angle):
        r = r_axis(raxiscc, nfp, angle, raxiscs = raxiscs)
        return jnp.array([r * jnp.cos(angle), r * jnp.sin(angle), z_axis(zaxiscs, nfp, angle, zaxiscc = zaxiscc)])
    def normal_vec_function(angle):
        return axis_del_phi(raxiscc, zaxiscs, nfp, angle, raxiscs = raxiscs, zaxiscc = zaxiscc)
    return nfp, stellsym, axis_function, normal_vec_function

"""