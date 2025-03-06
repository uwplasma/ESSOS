import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap
from essos.surfaces import SurfaceRZFourier

class BiotSavart():
    def __init__(self, coils):
        self.coils = coils
        self.currents = coils.currents
        self.gamma = coils.gamma
        self.gamma_dash = coils.gamma_dash
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        dif_R = (jnp.array(points)-self.gamma).T
        dB = jnp.cross(self.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        return self.B(points)
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        return self.B(points)
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points
    
class Vmec():
    def __init__(self, wout, ntheta=30, nphi=30, close=True, range='full torus'):
        self.wout = wout
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout)
        self.nfp = int(self.nc.variables["nfp"][0])
        self.bmnc = jnp.array(self.nc.variables["bmnc"][:])
        self.xm = jnp.array(self.nc.variables["xm"][:])
        self.xn = jnp.array(self.nc.variables["xn"][:])
        self.rmnc = jnp.array(self.nc.variables["rmnc"][:])
        self.zmns = jnp.array(self.nc.variables["zmns"][:])
        self.bsubsmns = jnp.array(self.nc.variables["bsubsmns"][:])
        self.bsubumnc = jnp.array(self.nc.variables["bsubumnc"][:])
        self.bsubvmnc = jnp.array(self.nc.variables["bsubvmnc"][:])
        self.bsupumnc = jnp.array(self.nc.variables["bsupumnc"][:])
        self.bsupvmnc = jnp.array(self.nc.variables["bsupvmnc"][:])
        self.gmnc = jnp.array(self.nc.variables["gmnc"][:])
        self.xm_nyq = jnp.array(self.nc.variables["xm_nyq"][:])
        self.xn_nyq = jnp.array(self.nc.variables["xn_nyq"][:])
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = self.nc.variables["ns"][0]
        self.s_full_grid = jnp.linspace(0, 1, self.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        self.r_axis = self.rmnc[0, 0]
        self.mpol = int(jnp.max(self.xm))
        self.ntor = int(jnp.max(jnp.abs(self.xn)) / self.nfp)
        self._surface = SurfaceRZFourier(self, ntheta=ntheta, nphi=nphi, close=close, range=range)

    @property
    def surface(self):
        return self._surface
        
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        s, theta, phi = points
        bsubsmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.bsubsmns)
        bsubumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubumnc[1:])
        bsubvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsubvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sinangle_nyq = jnp.sin(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s = jnp.dot(bsubsmns_interp, sinangle_nyq)
        B_sub_theta = jnp.dot(bsubumnc_interp, cosangle_nyq)
        B_sub_phi = jnp.dot(bsubvmnc_interp, cosangle_nyq)
        return jnp.array([B_sub_s, B_sub_theta, B_sub_phi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        s, theta, phi = points
        bsupumnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupumnc[1:])
        bsupvmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bsupvmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sup_theta = jnp.dot(bsupumnc_interp, cosangle_nyq)
        B_sup_phi = jnp.dot(bsupvmnc_interp, cosangle_nyq)
        return jnp.array([0*B_sup_theta, B_sup_theta, B_sup_phi])
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        d_rmnc_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.rmnc)
        d_zmns_d_s_interp = vmap(lambda row: grad(lambda s: jnp.interp(s, self.s_full_grid, row))(s), in_axes=1)(self.zmns)
        
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        B_sub_s, B_sub_theta, B_sub_phi = self.B_covariant(points)
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
        bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
        cos_values = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        return jnp.dot(bmnc_interp, cos_values)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        s, theta, phi = points
        rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        cosangle = jnp.cos(self.xm * theta - self.xn * phi)
        sinangle = jnp.sin(self.xm * theta - self.xn * phi)
        R = jnp.dot(rmnc_interp, cosangle)
        Z = jnp.dot(zmns_interp, sinangle)
        X = R * jnp.cos(phi)
        Y = R * jnp.sin(phi)
        return jnp.array([X, Y, Z])
    
    # @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    # def surface_gamma(self, s, ntheta=30, nphi=30):
    #     rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
    #     zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
    #     theta_2d, phi_2d = jnp.meshgrid(
    #         jnp.linspace(0, 2 * jnp.pi, num=ntheta),
    #         jnp.linspace(0, 2 * jnp.pi, num=nphi))
    #     r_coordinate = jnp.zeros((ntheta, nphi))
    #     z_coordinate = jnp.zeros((ntheta, nphi))
    #     angles = jnp.einsum('i,jk->ijk', self.xm, theta_2d) - jnp.einsum('i,jk->ijk', self.xn, phi_2d)
    #     cos_angles = jnp.cos(angles)
    #     sin_angles = jnp.sin(angles)
    #     r_coordinate = jnp.einsum('i,ijk->jk', rmnc_interp, cos_angles)
    #     z_coordinate = jnp.einsum('i,ijk->jk', zmns_interp, sin_angles)
    #     x_coordinate = r_coordinate * jnp.cos(phi_2d)
    #     y_coordinate = r_coordinate * jnp.sin(phi_2d)
    #     return jnp.array([x_coordinate, y_coordinate, z_coordinate])

    # @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    # def surface_gammadash(self, s, ntheta=30, nphi=30):
    #     rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
    #     zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
    #     theta_2d, phi_2d = jnp.meshgrid(
    #         jnp.linspace(0, 2 * jnp.pi, num=ntheta),
    #         jnp.linspace(0, 2 * jnp.pi, num=nphi),
    #         indexing="ij" 
    #     )
    #     angles = jnp.einsum('i,jk->ijk', self.xm, theta_2d) - jnp.einsum('i,jk->ijk', self.xn, phi_2d)
    #     sin_angles = jnp.sin(angles)
    #     cos_angles = jnp.cos(angles)
    #     dX_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.cos(phi_2d)
    #     dY_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.sin(phi_2d)
    #     dZ_dtheta = jnp.einsum('i,ijk,i->jk', self.xm, cos_angles, zmns_interp)
    #     dX_dphi = jnp.einsum('i,ijk,i->jk', self.xn, cos_angles, rmnc_interp) * jnp.cos(phi_2d) - jnp.einsum('i,i,jk->jk', rmnc_interp, jnp.ones_like(rmnc_interp), jnp.sin(phi_2d))
    #     dY_dphi = jnp.einsum('i,ijk,i->jk', self.xn, cos_angles, rmnc_interp) * jnp.sin(phi_2d) + jnp.einsum('i,i,jk->jk', rmnc_interp, jnp.ones_like(rmnc_interp), jnp.cos(phi_2d))
    #     dZ_dphi = jnp.einsum('i,ijk,i->jk', -self.xn, sin_angles, zmns_interp)
    #     return jnp.array([dX_dtheta, dY_dtheta, dZ_dtheta]), jnp.array([dX_dphi, dY_dphi, dZ_dphi])

    # @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    # def surface_normal(self, s, ntheta=30, nphi=30):
    #     gammadash1, gammadash2 = self.surface_gammadash(s, ntheta, nphi)
    #     normal = jnp.cross(gammadash1, gammadash2, axis=0)
    #     return normal #/ jnp.linalg.norm(normal, axis=0)

    # @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    # def surface_AbsB(self, s, ntheta=30, nphi=30):
    #     theta_2d, phi_2d = jnp.meshgrid(
    #         jnp.linspace(0, 2 * jnp.pi, num=ntheta),
    #         jnp.linspace(0, 2 * jnp.pi, num=ntheta))
    #     bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
    #     angles = jnp.einsum('i,jk->ijk', self.xm_nyq, theta_2d) - jnp.einsum('i,jk->ijk', self.xn_nyq, phi_2d)
    #     cos_angles = jnp.cos(angles)
    #     return jnp.einsum('i,ijk->jk', bmnc_interp, cos_angles)
        
    # def plot_surface(self, s, ntheta=50, nphi=50, ax=None, show=True, close=False, axis_equal=True, **kwargs):
    #     if close: raise NotImplementedError("close=True is not implemented, need to have closed surfaces")

    #     import matplotlib.pyplot as plt 
    #     from matplotlib import cm
    #     if ax is None or ax.name != "3d":
    #         fig = plt.figure()
    #         ax = fig.add_subplot(projection='3d')
            
    #     boundary = self.surface_gamma(s, ntheta, nphi)
    #     Bmag = self.surface_AbsB(s, ntheta, nphi)
    #     B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
        
    #     ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.6, facecolors=cm.jet(B_rescaled), linewidth=0, antialiased=False)
    #     ax.set_axis_off()

    #     if axis_equal:
    #         fix_matplotlib_3d(ax)
    #     if show:
    #         plt.show()
    
    # def to_vtk(self, filename, extra_data=None, field=None, ntheta=50, nphi=50):
    #     import numpy as np
    #     from pyevtk.hl import gridToVTK
    #     boundary = np.array(self.surface_gamma(s=1, ntheta=ntheta, nphi=nphi))
    #     Bmag = np.array(self.surface_AbsB(s=1, ntheta=ntheta, nphi=nphi))
    #     x = boundary[0].reshape((1, ntheta, nphi)).copy()
    #     y = boundary[1].reshape((1, ntheta, nphi)).copy()
    #     z = boundary[2].reshape((1, ntheta, nphi)).copy()
    #     Bmag = Bmag.reshape((1, ntheta, nphi)).copy()
    #     pointData = {}
    #     if field is not None:
    #         B_dot_n_over_B = np.array(BdotN_over_B(self, field, ntheta, nphi)).reshape((1, ntheta, nphi)).copy()
    #         pointData["B_dot_n_over_B"] = B_dot_n_over_B
    #         B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary.T)).reshape((1, ntheta, nphi)).copy()
    #         pointData["B_BiotSavart"] = B_BiotSavart
    #     pointData["B_VMEC"]=Bmag
    #     if extra_data is not None:
    #         pointData = {**pointData, **extra_data}
    #     gridToVTK(str(filename), x, y, z, pointData=pointData)
