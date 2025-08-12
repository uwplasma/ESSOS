from functools import partial
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax import jit, vmap, devices, device_put
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from essos.plot import fix_matplotlib_3d
import jaxkd

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

def nested_lists_to_array(ll):
    """
    Convert a ragged list of lists to a 2D jnp array.  Any entries
    that are None are replaced by 0. This routine is useful for
    parsing fortran namelists that include 2D arrays using f90nml.

    Args:
        ll: A list of lists to convert.
    """
    mdim = len(ll)
    ndim = max(len(x) for x in ll)
    arr = jnp.zeros((mdim, ndim))
    for jm, l in enumerate(ll):
        arr = arr.at[jm, :len(l)].set(jnp.array([x if x is not None else 0 for x in l]))
    return arr
    

class SurfaceRZFourier:
    def __init__(self, vmec=None, s=1, ntheta=30, nphi=30, close=True, range_torus='full torus',
                 rc=None, zs=None, nfp=None):
        if rc is not None:
            self.rc = rc
            self.zs = zs
            self.nfp = nfp
            self.mpol = rc.shape[0]
            self.ntor = (rc.shape[1] - 1) // 2
            m1d = jnp.arange(self.mpol)
            n1d = jnp.arange(-self.ntor, self.ntor + 1)
            n2d, m2d = jnp.meshgrid(n1d, m1d)
            self.xm = m2d.flatten()[self.ntor:]
            self.xn = self.nfp*n2d.flatten()[self.ntor:]
            indices = jnp.array([self.xm, self.xn / self.nfp + self.ntor], dtype=int).T
            self.rmnc_interp = self.rc[indices[:, 0], indices[:, 1]]
            self.zmns_interp = self.zs[indices[:, 0], indices[:, 1]]   
        elif isinstance(vmec, str):
            self.input_filename = vmec
            import f90nml
            all_namelists = f90nml.read(vmec)
            nml = all_namelists['indata']
            if 'nfp' in nml:
                self.nfp = nml['nfp']
            else:
                self.nfp = 1
            rc = nested_lists_to_array(nml['rbc'])
            zs = nested_lists_to_array(nml['zbs'])
            rbc_first_n = nml.start_index['rbc'][0]
            rbc_last_n = rbc_first_n + rc.shape[1] - 1
            zbs_first_n = nml.start_index['zbs'][0]
            zbs_last_n = zbs_first_n + zs.shape[1] - 1
            self.ntor = jnp.max(jnp.abs(jnp.array([rbc_first_n, rbc_last_n, zbs_first_n, zbs_last_n], dtype='i')))
            rbc_first_m = nml.start_index['rbc'][1]
            rbc_last_m = rbc_first_m + rc.shape[0] - 1
            zbs_first_m = nml.start_index['zbs'][1]
            zbs_last_m = zbs_first_m + zs.shape[0] - 1
            self.mpol = max(rbc_last_m, zbs_last_m)
            self.rc = jnp.zeros((self.mpol, 2 * self.ntor + 1))
            self.zs = jnp.zeros((self.mpol, 2 * self.ntor + 1))
            m_indices_rc = jnp.arange(rc.shape[0]) + nml.start_index['rbc'][1]
            n_indices_rc = jnp.arange(rc.shape[1]) + nml.start_index['rbc'][0] + self.ntor
            self.rc = self.rc.at[m_indices_rc[:, None], n_indices_rc].set(rc)
            m_indices_zs = jnp.arange(zs.shape[0]) + nml.start_index['zbs'][1]
            n_indices_zs = jnp.arange(zs.shape[1]) + nml.start_index['zbs'][0] + self.ntor
            self.zs = self.zs.at[m_indices_zs[:, None], n_indices_zs].set(zs)
            m1d = jnp.arange(self.mpol)
            n1d = jnp.arange(-self.ntor, self.ntor + 1)
            n2d, m2d = jnp.meshgrid(n1d, m1d)
            self.xm = m2d.flatten()[self.ntor:]
            self.xn = self.nfp*n2d.flatten()[self.ntor:]
            indices = jnp.array([self.xm, self.xn / self.nfp + self.ntor], dtype=int).T
            self.rmnc_interp = self.rc[indices[:, 0], indices[:, 1]]
            self.zmns_interp = self.zs[indices[:, 0], indices[:, 1]]              
        else:
            try:
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
                self.num_dofs = 2 * (self.mpol + 1) * (2 * self.ntor + 1) - self.ntor - (self.ntor + 1)
                shape = (int(jnp.max(self.xm)) + 1, int(jnp.max(self.xn)) + 1)
                self.rc = jnp.zeros(shape)
                self.zs = jnp.zeros(shape)
                indices = jnp.array([self.xm, self.xn / self.nfp + self.ntor], dtype=int).T
                self.rc = self.rc.at[indices[:, 0], indices[:, 1]].set(self.rmnc_interp)
                self.zs = self.zs.at[indices[:, 0], indices[:, 1]].set(self.zmns_interp)
            except:
                raise ValueError("vmec must be a Vmec object or a string pointing to a VMEC input file.")
        self.ntheta = ntheta
        self.nphi = nphi
        self.range_torus = range_torus
        if range_torus == 'full torus': div = 1
        else: div = self.nfp
        if range_torus == 'half period': end_val = 0.5
        else: end_val = 1.0
        self.quadpoints_theta = jnp.linspace(0, 2 * jnp.pi, num=self.ntheta, endpoint=True if close else False)
        self.quadpoints_phi   = jnp.linspace(0, 2 * jnp.pi * end_val / div, num=self.nphi, endpoint=True if close else False)
        self.theta_2d, self.phi_2d = jnp.meshgrid(self.quadpoints_theta, self.quadpoints_phi)
        self.num_dofs_rc = len(jnp.ravel(self.rc)[self.ntor:])
        self.num_dofs_zs = len(jnp.ravel(self.zs)[self.ntor:])
        self._dofs = jnp.concatenate((jnp.ravel(self.rc)[self.ntor:], jnp.ravel(self.zs)[self.ntor:]))
        
        self.angles =  jnp.einsum('i,jk->ijk', self.xm, self.theta_2d) - jnp.einsum('i,jk->ijk', self.xn, self.phi_2d)
    
        (self._gamma, self._gammadash_theta, self._gammadash_phi,
            self._normal, self._unitnormal) = self._set_gamma(self.rmnc_interp, self.zmns_interp)
        
        if hasattr(self, 'bmnc'):
            self._AbsB = self._set_AbsB()

    @property
    def dofs(self):
        return self._dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        self._dofs = new_dofs
        self.rc = jnp.concatenate((jnp.zeros(self.ntor),new_dofs[:self.num_dofs_rc])).reshape(self.rc.shape)
        self.zs = jnp.concatenate((jnp.zeros(self.ntor),new_dofs[self.num_dofs_rc:])).reshape(self.zs.shape)
        indices = jnp.array([self.xm, self.xn / self.nfp + self.ntor], dtype=int).T
        self.rmnc_interp = self.rc[indices[:, 0], indices[:, 1]]
        self.zmns_interp = self.zs[indices[:, 0], indices[:, 1]]
        (self._gamma, self._gammadash_theta, self._gammadash_phi,
            self._normal, self._unitnormal) = self._set_gamma(self.rmnc_interp, self.zmns_interp)
        # if hasattr(self, 'bmnc'):
        #     self._AbsB = self._set_AbsB()
        
    @partial(jit, static_argnames=['self'])
    def _set_gamma(self, rmnc_interp, zmns_interp):
        phi_2d = self.phi_2d
        angles = self.angles
        
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        r_coordinate = jnp.einsum('i,ijk->jk', rmnc_interp, cos_angles)
        z_coordinate = jnp.einsum('i,ijk->jk', zmns_interp, sin_angles)
        gamma = jnp.transpose(jnp.array([r_coordinate * jnp.cos(phi_2d), r_coordinate * jnp.sin(phi_2d), z_coordinate]), (1, 2, 0))

        dX_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.cos(phi_2d)
        dY_dtheta = jnp.einsum('i,ijk,i->jk', -self.xm, sin_angles, rmnc_interp) * jnp.sin(phi_2d)
        dZ_dtheta = jnp.einsum('i,ijk,i->jk',  self.xm, cos_angles, zmns_interp)
        gammadash_theta = 2*jnp.pi*jnp.transpose(jnp.array([dX_dtheta, dY_dtheta, dZ_dtheta]), (1, 2, 0))

        dX_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, rmnc_interp) * jnp.cos(phi_2d) - r_coordinate * jnp.sin(phi_2d)
        dY_dphi = jnp.einsum('i,ijk,i->jk',  self.xn, sin_angles, rmnc_interp) * jnp.sin(phi_2d) + r_coordinate * jnp.cos(phi_2d)
        dZ_dphi = jnp.einsum('i,ijk,i->jk', -self.xn, cos_angles, zmns_interp)
        gammadash_phi = 2*jnp.pi*jnp.transpose(jnp.array([dX_dphi, dY_dphi, dZ_dphi]), (1, 2, 0))

        normal = jnp.cross(gammadash_phi, gammadash_theta, axis=2)
        unitnormal = normal / jnp.linalg.norm(normal, axis=2, keepdims=True)
        
        return (gamma, gammadash_theta, gammadash_phi, normal, unitnormal)
    
    @partial(jit, static_argnames=['self'])
    def _set_AbsB(self):
        angles_nyq = jnp.einsum('i,jk->ijk', self.xm_nyq, self.theta_2d) - jnp.einsum('i,jk->ijk', self.xn_nyq, self.phi_2d)
        AbsB = jnp.einsum('i,ijk->jk', self.bmnc_interp, jnp.cos(angles_nyq))
        return AbsB
    
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
    
    @property
    def x(self):
        return self.dofs

    @x.setter
    def x(self, new_dofs):
        self.dofs = new_dofs
        
    def plot(self, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        if close: raise NotImplementedError("Call close=True when instantiating the VMEC/SurfaceRZFourier object.")
        
        kwargs.setdefault('alpha', 0.6)

        import matplotlib.pyplot as plt 
        from matplotlib import cm
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
        boundary = self.gamma
        
        if hasattr(self, 'bmnc'):
            Bmag = self.AbsB
            B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
            ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], facecolors=cm.jet(B_rescaled), linewidth=0, antialiased=True, **kwargs)
        else:
            ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], linewidth=0, antialiased=True, **kwargs)
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
        if hasattr(self, 'bmnc'):
            Bmag = np.array(self.AbsB)
            Bmag = Bmag.reshape((1, self.nphi, self.ntheta)).copy()
        x = boundary[:, :, 0].reshape((1, self.nphi, self.ntheta)).copy()
        y = boundary[:, :, 1].reshape((1, self.nphi, self.ntheta)).copy()
        z = boundary[:, :, 2].reshape((1, self.nphi, self.ntheta)).copy()
        pointData = {}
        if field is not None:
            B_dot_n_over_B = np.array(BdotN_over_B(self, field)).reshape((1,self. nphi, self.ntheta)).copy()
            pointData["B_dot_n_over_B"] = B_dot_n_over_B
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary)).reshape((1, self.nphi, self.ntheta)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        if hasattr(self, 'bmnc'):
            pointData["B_VMEC"]=Bmag
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        gridToVTK(str(filename), x, y, z, pointData=pointData)

    def to_vmec(self, filename):
        """
        Generates a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. The result will be returned as a string. For saving a
        file, see the ``write_nml()`` function.
        """
        nml = ''
        nml += '&INDATA\n'
        nml += 'LASYM = .FALSE.\n'
        nml += f'NFP = {self.nfp}\n'

        for m in range(self.mpol + 1):
            nmin = -self.ntor
            if m == 0:
                nmin = 0
            for n in range(nmin, self.ntor + 1):
                rc = self.rc[m, n + self.ntor]
                zs = self.zs[m, n + self.ntor]
                if jnp.abs(rc) > 0 or jnp.abs(zs) > 0:
                    nml += f"RBC({n:4d},{m:4d}) ={rc:23.15e},    ZBS({n:4d},{m:4d}) ={zs:23.15e}\n"
        nml += '/\n'
        
        with open(filename, 'w') as f:
            f.write(nml)
            
    def mean_cross_sectional_area(self):
        xyz = self.gamma
        x2y2 = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
        dgamma1 = self.gammadash_phi
        dgamma2 = self.gammadash_theta
        J = jnp.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
        J = J.at[:, :, 0, 0].set((xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0]) / x2y2)
        J = J.at[:, :, 0, 1].set((xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0]) / x2y2)
        J = J.at[:, :, 1, 0].set(0)
        J = J.at[:, :, 1, 1].set(1)
        detJ = jnp.linalg.det(J)
        Jinv = jnp.linalg.inv(J)
        dZ_dtheta = dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
        mean_cross_sectional_area = jnp.abs(jnp.mean(jnp.sqrt(x2y2) * dZ_dtheta * detJ))/(2 * jnp.pi)
        return mean_cross_sectional_area
    
#This class is based on simsopt classifier but translated to fit jax    
class SurfaceClassifier():
    """
    Takes in a toroidal surface and constructs an interpolant of the signed distance function
    :math:`f:R^3\to R` that is positive inside the volume contained by the surface,
    (approximately) zero on the surface, and negative outisde the volume contained by the surface.
    """

    def __init__(self, surface,h=0.05):
        """
        Args:
            surface: the surface to contruct the distance from.
            h: grid resolution of the interpolant
        """
        gammas = surface.gamma
        r = jnp.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(jnp.min(r) - 0.1, 0.)
        rmax = jnp.max(r) + 0.1
        zmin = jnp.min(z) - 0.1
        zmax = jnp.max(z) + 0.1

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*jnp.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        def fbatch(rs, phis, zs):
            xyz = jnp.zeros(( 3))
            xyz=xyz.at[0].set( rs * jnp.cos(phis))
            xyz=xyz.at[1].set(rs * jnp.sin(phis))
            xyz=xyz.at[2].set(zs)
            return signed_distance_from_surface_jax(xyz, surface)   
            #return signed_distance_from_surface_extras(xyz, surface) ####memory bounded

        #rule = sopp.UniformInterpolationRule(p) 
        #self.dist = RegularGridInterpolator((jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),
        #            vmap(vmap(vmap(fbatch,in_axes=(0,None,None)),in_axes=(None,0,None)),in_axes=(None,None,0))(jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)))
        #self.r_list=jnp.linspace(16.9,17.1,nr)
        #self.phi_list=jnp.linspace(0., 0.01, nphi)
        #self.z_list=jnp.linspace(-0.1, 0.1, nz)
        #self.test= vmap(vmap(vmap(fbatch,in_axes=(0,None,None)),in_axes=(None,0,None)),in_axes=(None,None,0))(self.r_list,
        #            self.phi_list, self.z_list)
        #self.r_list=jnp.linspace(rmin,rmax,nr)
        #self.phi_list=jnp.linspace(0., 2*jnp.pi, nphi)
        #self.z_list=jnp.linspace(zmin, zmax, nz)
        #self.test= vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(jnp.linspace(rmin,rmax,nr),
        #            jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz))
        #self.dist = RegularGridInterpolator((self.r_list,self.phi_list, self.z_list),
        #            vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(self.r_list,self.phi_list, self.z_list),fill_value=-1.)        
        self.dist = RegularGridInterpolator((jnp.linspace(rmin,rmax,nr),
                    jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),
                    vmap(vmap(vmap(fbatch,in_axes=(None,None,0)),in_axes=(None,0,None)),in_axes=(0,None,None))(jnp.linspace(rmin,rmax,nr),
                    jnp.linspace(0., 2*jnp.pi, nphi), jnp.linspace(zmin, zmax, nz)),fill_value=-1.)
        #self.dist.interpolate_batch(fbatch)    

    @partial(jit, static_argnames=['self'])
    def evaluate_xyz(self, xyz):
        rphiz = jnp.zeros_like(xyz)
        rphiz=rphiz.at[0].set(jnp.linalg.norm(xyz[:2]))
        rphiz=rphiz.at[1].set(jnp.mod(jnp.arctan2(xyz[1], xyz[0]), 2*jnp.pi))
        rphiz=rphiz.at[2].set(xyz.at[2].get())
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d=self.dist(rphiz)[0][0]
        return d

    @partial(jit, static_argnames=['self'])
    def evaluate_rphiz(self, rphiz):
        # initialize to -1 since the regular grid interpolant will just keep
        # that value when evaluated outside of bounds
        d=self.dist(rphiz)[0][0]
        return d
    

partial(jit, static_argnames=['surface'])
def signed_distance_from_surface_jax(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma.reshape((-1, 3))
    #from scipy.spatial import KDTree ##better for cpu?
    tree = jaxkd.build_tree(gammas)
    mins, _ = jaxkd.query_neighbors(tree, xyz, k=1)    
    n = surface.unitnormal.reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]
    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = jnp.sum((xyz-gammamins) * nmins, axis=1)
    a_point_in_the_surface = jnp.mean(surface.gamma[0, :, :], axis=0)
    sign_of_interiorpoint = jnp.sign(jnp.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))
    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists

#@partial(jit, static_argnames=['surface'])
def signed_distance_from_surface_extras(xyz, surface):
    """
    Compute the signed distances from points ``xyz`` to a surface.  The sign is
    positive for points inside the volume surrounded by the surface.
    """
    gammas = surface.gamma.reshape((-1, 3))
    mins, _ = jaxkd.extras.query_neighbors_pairwise(gammas, xyz, k=1)    
    n = surface.unitnormal.reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]
    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = jnp.sum((xyz-gammamins) * nmins, axis=1)
    a_point_in_the_surface = jnp.mean(surface.gamma[0, :, :], axis=0)
    sign_of_interiorpoint = jnp.sign(jnp.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))
    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists



