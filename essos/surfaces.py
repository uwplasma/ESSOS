from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
from .plot import fix_matplotlib_3d

class SurfaceRZFourier:
    def __init__(self, vmec, s=1):
        self.nfp = vmec.nfp
        self.bmnc = vmec.bmnc
        self.xm = vmec.xm
        self.xn = vmec.xn
        self.rmnc = vmec.rmnc
        self.zmns = vmec.zmns
        self.xm_nyq = vmec.xm_nyq
        self.xn_nyq = vmec.xn_nyq
        self.len_xm_nyq = len(self.xm_nyq)
        self.ns = vmec.ns
        self.s_full_grid = vmec.s_full_grid
        self.ds = vmec.ds
        self.s_half_grid = vmec.s_half_grid
        self.r_axis = vmec.r_axis
        self._dofs = jnp.concatenate([self.rmnc[1:, :].ravel(), self.zmns[1:, :].ravel()])
        self.rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        self.zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)

    @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    def surface_gamma(self, s, ntheta=30, nphi=30):
        theta_2d, phi_2d = jnp.meshgrid(
            jnp.linspace(0, 2 * jnp.pi, num=ntheta),
            jnp.linspace(0, 2 * jnp.pi, num=nphi))
        r_coordinate = jnp.zeros((ntheta, nphi))
        z_coordinate = jnp.zeros((ntheta, nphi))
        angles = jnp.einsum('i,jk->ijk', self.xm, theta_2d) - jnp.einsum('i,jk->ijk', self.xn, phi_2d)
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)
        r_coordinate = jnp.einsum('i,ijk->jk', self.rmnc_interp, cos_angles)
        z_coordinate = jnp.einsum('i,ijk->jk', self.zmns_interp, sin_angles)
        x_coordinate = r_coordinate * jnp.cos(phi_2d)
        y_coordinate = r_coordinate * jnp.sin(phi_2d)
        return jnp.array([x_coordinate, y_coordinate, z_coordinate])

    @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    def surface_gammadash(self, s, ntheta=30, nphi=30):
        theta_2d, phi_2d = jnp.meshgrid(
            jnp.linspace(0, 2 * jnp.pi, num=ntheta),
            jnp.linspace(0, 2 * jnp.pi, num=nphi),
            indexing="ij" )
        angles = jnp.einsum('i,jk->ijk', self.xm, theta_2d) - jnp.einsum('i,jk->ijk', self.xn, phi_2d)
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        dX_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, self.rmnc_interp) * jnp.cos(phi_2d)
        dY_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, self.rmnc_interp) * jnp.sin(phi_2d)
        dZ_dtheta = jnp.einsum('i,ijk,i->jk', self.xm, cos_angles, self.zmns_interp)
        dX_dphi = jnp.einsum('i,ijk,i->jk', self.xn, cos_angles, self.rmnc_interp) * jnp.cos(phi_2d) - jnp.einsum('i,i,jk->jk', self.rmnc_interp, jnp.ones_like(self.rmnc_interp), jnp.sin(phi_2d))
        dY_dphi = jnp.einsum('i,ijk,i->jk', self.xn, cos_angles, self.rmnc_interp) * jnp.sin(phi_2d) + jnp.einsum('i,i,jk->jk', self.rmnc_interp, jnp.ones_like(self.rmnc_interp), jnp.cos(phi_2d))
        dZ_dphi = jnp.einsum('i,ijk,i->jk', -self.xn, sin_angles, self.zmns_interp)
        return jnp.array([dX_dtheta, dY_dtheta, dZ_dtheta]), jnp.array([dX_dphi, dY_dphi, dZ_dphi])

    @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    def surface_normal(self, s, ntheta=30, nphi=30):
        gammadash1, gammadash2 = self.surface_gammadash(s, ntheta, nphi)
        normal = jnp.cross(gammadash1, gammadash2, axis=0)
        return normal #/ jnp.linalg.norm(normal, axis=0)

    @partial(jit, static_argnames=['self', 'ntheta', 'nphi'])
    def surface_AbsB(self, s, ntheta=30, nphi=30):
        theta_2d, phi_2d = jnp.meshgrid(
            jnp.linspace(0, 2 * jnp.pi, num=ntheta),
            jnp.linspace(0, 2 * jnp.pi, num=ntheta))
        bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
        angles = jnp.einsum('i,jk->ijk', self.xm_nyq, theta_2d) - jnp.einsum('i,jk->ijk', self.xn_nyq, phi_2d)
        cos_angles = jnp.cos(angles)
        return jnp.einsum('i,ijk->jk', bmnc_interp, cos_angles)
        
    def plot_surface(self, s, ntheta=50, nphi=50, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        if close: raise NotImplementedError("close=True is not implemented, need to have closed surfaces")

        import matplotlib.pyplot as plt 
        from matplotlib import cm
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
        boundary = self.surface_gamma(s, ntheta, nphi)
        Bmag = self.surface_AbsB(s, ntheta, nphi)
        B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
        
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.6, facecolors=cm.jet(B_rescaled), linewidth=0, antialiased=False)
        ax.set_axis_off()

        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
    
    def to_vtk(self, filename, extra_data=None, field=None, ntheta=50, nphi=50):
        import numpy as np
        from pyevtk.hl import gridToVTK
        boundary = np.array(self.surface_gamma(s=1, ntheta=ntheta, nphi=nphi))
        Bmag = np.array(self.surface_AbsB(s=1, ntheta=ntheta, nphi=nphi))
        x = boundary[0].reshape((1, ntheta, nphi)).copy()
        y = boundary[1].reshape((1, ntheta, nphi)).copy()
        z = boundary[2].reshape((1, ntheta, nphi)).copy()
        Bmag = Bmag.reshape((1, ntheta, nphi)).copy()
        pointData = {}
        if field is not None:
            B_dot_n_over_B = np.array(BdotN_over_B(self, field, ntheta, nphi)).reshape((1, ntheta, nphi)).copy()
            pointData["B_dot_n_over_B"] = B_dot_n_over_B
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary.T)).reshape((1, ntheta, nphi)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        pointData["B_VMEC"]=Bmag
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        gridToVTK(str(filename), x, y, z, pointData=pointData)
