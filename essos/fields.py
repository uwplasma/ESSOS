import jax.numpy as jnp
from jax import jit, jacfwd, grad, vmap
from functools import partial

class BiotSavart():
    def __init__(self, coils):
        self.coils = coils
        self.currents = coils.currents[0]
        self.gamma = coils.gamma
        self.gamma_dash = coils.gamma_dash
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        dif_R = (points-self.gamma).T
        dB = jnp.cross(self.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)

class Vmec():
    def __init__(self, wout):
        self.wout = wout
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout)
        self.nfp = self.nc.variables["nfp"][0]
        self.bmnc = jnp.array(self.nc.variables["bmnc"][:])
        self.xm = jnp.array(self.nc.variables["xm"][:])
        self.xn = jnp.array(self.nc.variables["xn"][:])
        self.rmnc = jnp.array(self.nc.variables["rmnc"][:])
        self.zmns = jnp.array(self.nc.variables["zmns"][:])
        self.bsubsmns = jnp.array(self.nc.variables["bsubsmns"][:])
        self.bsubumnc = jnp.array(self.nc.variables["bsubumnc"][:])
        self.bsubvmnc = jnp.array(self.nc.variables["bsubvmnc"][:])
        self.gmnc = jnp.array(self.nc.variables["gmnc"][:])
        self.xm_nyq = jnp.array(self.nc.variables["xm_nyq"][:])
        self.xn_nyq = jnp.array(self.nc.variables["xn_nyq"][:])
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = self.nc.variables["ns"][0]
        self.s_full_grid = jnp.linspace(0, 1, self.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        s, theta, phi = points
        bsubsmns_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row), in_axes=1)(self.bsubsmns[1:, :])
        bsubumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row), in_axes=1)(self.bsubumnc[1:, :])
        bsubvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row), in_axes=1)(self.bsubvmnc[1:, :])
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row), in_axes=1)(self.gmnc[1:])
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row), in_axes=1)(self.zmns)
        d_rmnc_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.rmnc)
        d_zmns_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.zmns)
        
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sinangle_nyq = jnp.sin(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s = jnp.dot(bsubsmns_interp, sinangle_nyq)
        B_sub_theta = jnp.dot(bsubumnc_interp, cosangle_nyq)
        B_sub_phi = jnp.dot(bsubvmnc_interp, cosangle_nyq)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        
        cosangle  = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle  = jnp.sin(self.xm * theta - self.xn * phi)
        msinangle = self.xm * sinangle
        nsinangle = self.xn * sinangle
        mcosangle = self.xm * cosangle
        ncosangle = self.xn * cosangle
        
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        
        R = jnp.dot(rmnc_interp, cosangle)
        d_R_d_theta = jnp.dot(rmnc_interp, -msinangle)
        d_R_d_phi   = jnp.dot(rmnc_interp, nsinangle)
        d_R_d_s     = jnp.dot(d_rmnc_d_s_interp, cosangle)
        
        d_X_d_theta = d_R_d_theta * cosphi
        d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
        d_X_d_s = d_R_d_s * cosphi

        d_Y_d_theta = d_R_d_theta * sinphi
        d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
        d_Y_d_s = d_R_d_s * sinphi
        
        d_Z_d_s = jnp.dot(d_zmns_d_s_interp, sinangle)
        d_Z_d_theta = jnp.dot(zmns_interp, mcosangle)
        d_Z_d_phi = jnp.dot(zmns_interp, -ncosangle)

        grad_s_X = (d_Y_d_theta * d_Z_d_phi - d_Z_d_theta * d_Y_d_phi) / sqrt_g_vmec
        grad_s_Y = (d_Z_d_theta * d_X_d_phi - d_X_d_theta * d_Z_d_phi) / sqrt_g_vmec
        grad_s_Z = (d_X_d_theta * d_Y_d_phi - d_Y_d_theta * d_X_d_phi) / sqrt_g_vmec

        grad_theta_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
        grad_theta_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
        grad_theta_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

        grad_phi_X = (d_Y_d_s * d_Z_d_theta - d_Z_d_s * d_Y_d_theta) / sqrt_g_vmec
        grad_phi_Y = (d_Z_d_s * d_X_d_theta - d_X_d_s * d_Z_d_theta) / sqrt_g_vmec
        grad_phi_Z = (d_X_d_s * d_Y_d_theta - d_Y_d_s * d_X_d_theta) / sqrt_g_vmec
        
        return jnp.array([B_sub_s * grad_s_X + B_sub_theta * grad_theta_X + B_sub_phi * grad_phi_X,
                          B_sub_s * grad_s_Y + B_sub_theta * grad_theta_Y + B_sub_phi * grad_phi_Y,
                          B_sub_s * grad_s_Z + B_sub_theta * grad_theta_Z + B_sub_phi * grad_phi_Z])
        
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        s, theta, phi = points
        bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row), in_axes=1)(self.bmnc[1:, :])
        cos_values = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        return jnp.dot(bmnc_interp, cos_values)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)