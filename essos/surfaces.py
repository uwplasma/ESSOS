from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, devices, device_put
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from .plot import fix_matplotlib_3d

mesh = Mesh(devices(), ("dev",))
sharding = NamedSharding(mesh, PartitionSpec("dev", None))

@partial(jit, static_argnames=['surface','field'])
def B_on_surface(surface, field):
    ntheta = surface.ntheta
    nphi = surface.nphi
    gamma = surface.gamma
    gamma_reshaped = gamma.reshape(nphi * ntheta, 3)
    gamma_sharded = device_put(gamma_reshaped, sharding)
    B_on_surface = jit(vmap(field.B), in_shardings=sharding, out_shardings=sharding)(gamma_sharded)
    B_on_surface = B_on_surface.reshape(nphi, ntheta, 3)
    return B_on_surface

@partial(jit, static_argnames=['surface','field'])
def BdotN(surface, field):
    B_surface = B_on_surface(surface, field)
    B_dot_n = jnp.sum(B_surface * surface.unitnormal, axis=2)
    return B_dot_n

@partial(jit, static_argnames=['surface','field'])
def BdotN_over_B(surface, field):
    B_surface = B_on_surface(surface, field)
    B_dot_n = jnp.sum(B_surface * surface.unitnormal, axis=2)
    return B_dot_n / jnp.linalg.norm(B_surface, axis=2)

class SurfaceRZFourier:
    def __init__(self, vmec, s=1, ntheta=30, nphi=30, close=True, range='full torus'):
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
        self.rmnc_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.rmnc)
        self.zmns_interp = vmap(lambda row: jnp.interp(s, self.s_full_grid, row, left='extrapolate'), in_axes=1)(self.zmns)
        self.bmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.bmnc[1:, :])
        self.mpol = vmec.mpol
        self.ntor = vmec.ntor
        self.ntheta = ntheta
        self.nphi = nphi
        if range == 'full torus': div = 1
        else: div = self.nfp
        if range == 'half period': end_val = 0.5
        else: end_val = 1.0
        
        self.quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=self.ntheta, endpoint=True if close else False)
        self.quadpoints_phi   = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=self.nphi, endpoint=True if close else False)
        
        self.num_dofs = 2 * (self.mpol + 1) * (2 * self.ntor + 1) - self.ntor - (self.ntor + 1)
        shape = (int(jnp.max(self.xm)) + 1, int(jnp.max(self.xn)) + 1)
        self.rc = jnp.zeros(shape)
        self.zs = jnp.zeros(shape)
        for j, xm_this in enumerate(self.xm):
            m = int(self.xm[j])
            n = int(self.xn[j] / self.nfp)
            self.rc = self.rc.at[m, n + self.ntor].set(self.rmnc_interp[j])
            self.zs = self.zs.at[m, n + self.ntor].set(self.zmns_interp[j])
        
        self._dofs = jnp.concatenate((jnp.ravel(self.rc), jnp.ravel(self.zs)))
        
        self._set_gamma()

    @property
    def dofs(self):
        return self._dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        self._dofs = new_dofs
        self.rc = new_dofs[:self.num_dofs].reshape(self.rc.shape)
        self.zs = new_dofs[self.num_dofs:].reshape(self.zs.shape)
        # !! Need to change rmnc and zmns as well
        
    partial(jit, static_argnames=['self'])
    def _set_gamma(self):
        theta_2d, phi_2d = jnp.meshgrid(self.quadpoints_theta, self.quadpoints_phi)
        angles = jnp.einsum('i,jk->ijk', self.xm, theta_2d) - jnp.einsum('i,jk->ijk', self.xn, phi_2d)
        
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        r_coordinate = jnp.einsum('i,ijk->jk', self.rmnc_interp, cos_angles)
        z_coordinate = jnp.einsum('i,ijk->jk', self.zmns_interp, sin_angles)
        gamma = jnp.transpose(jnp.array([r_coordinate * jnp.cos(phi_2d), r_coordinate * jnp.sin(phi_2d), z_coordinate]), (1, 2, 0))

        dX_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, self.rmnc_interp) * jnp.cos(phi_2d)
        dY_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, self.rmnc_interp) * jnp.sin(phi_2d)
        dZ_dtheta = jnp.einsum('i,ijk,i->jk',  self.xm, cos_angles, self.zmns_interp)
        gammadash_theta = 2*jnp.pi*jnp.transpose(jnp.array([dX_dtheta, dY_dtheta, dZ_dtheta]), (1, 2, 0))

        dX_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, self.rmnc_interp) * jnp.cos(phi_2d) - r_coordinate * jnp.sin(phi_2d)
        dY_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, self.rmnc_interp) * jnp.sin(phi_2d) + r_coordinate * jnp.cos(phi_2d)
        dZ_dphi = jnp.einsum('i,ijk,i->jk', -self.xn, cos_angles, self.zmns_interp)
        gammadash_phi = 2*jnp.pi*jnp.transpose(jnp.array([dX_dphi, dY_dphi, dZ_dphi]), (1, 2, 0))

        normal = jnp.cross(gammadash_phi, gammadash_theta, axis=2)
        unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)

        angles_nyq = jnp.einsum('i,jk->ijk', self.xm_nyq, theta_2d) - jnp.einsum('i,jk->ijk', self.xn_nyq, phi_2d)
        AbsB = jnp.einsum('i,ijk->jk', self.bmnc_interp, jnp.cos(angles_nyq))
        
        self._gamma = gamma
        self._gammadash_theta = gammadash_theta
        self._gammadash_phi = gammadash_phi
        self._normal = normal
        self._unitnormal = unitnormal
        self._AbsB = AbsB
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def gammadash_theta(self):
        return self._gammadash_theta
    
    @property
    def gammadash_phi(self):
        return self._gammadash_phi
    
    @property
    def normal(self):
        return self._normal
    
    @property
    def unitnormal(self):
        return self._unitnormal
    
    @property
    def AbsB(self):
        return self._AbsB
        
    def plot(self, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        if close: raise NotImplementedError("close=True is not implemented, need to have closed surfaces")

        import matplotlib.pyplot as plt 
        from matplotlib import cm
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
        boundary = self.gamma
        Bmag = self.AbsB
        B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
        
        ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], alpha=0.6, facecolors=cm.jet(B_rescaled), linewidth=0, antialiased=True, **kwargs)
        # ax.set_axis_off()
        ax.grid(False)

        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
    
    def to_vtk(self, filename, extra_data=None, field=None):
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import gridToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        boundary = np.array(self.gamma)
        Bmag = np.array(self.AbsB)
        x = boundary[:, :, 0].reshape((1, self.nphi, self.ntheta)).copy()
        y = boundary[:, :, 1].reshape((1, self.nphi, self.ntheta)).copy()
        z = boundary[:, :, 2].reshape((1, self.nphi, self.ntheta)).copy()
        Bmag = Bmag.reshape((1, self.nphi, self.ntheta)).copy()
        pointData = {}
        if field is not None:
            B_dot_n_over_B = np.array(BdotN_over_B(self, field)).reshape((1,self. nphi, self.ntheta)).copy()
            pointData["B_dot_n_over_B"] = B_dot_n_over_B
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary)).reshape((1, self.nphi, self.ntheta)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        pointData["B_VMEC"]=Bmag
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        gridToVTK(str(filename), x, y, z, pointData=pointData)
