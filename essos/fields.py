import jax
jax.config.update("jax_enable_x64", True)
from jax import vmap
from essos.coils import compute_curvature
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap, tree_util, lax
from essos.surfaces import SurfaceRZFourier, BdotN_over_B,SurfaceClassifier
from essos.plot import fix_matplotlib_3d
from essos.util import newton

class BiotSavart():
    def __init__(self, coils):
        self.coils = coils
        self.currents = coils.currents
        self.gamma = coils.gamma
        self.gamma_dash = coils.gamma_dash
        self.gamma_dashdash = coils.gamma_dashdash
        self.coils_length=jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in self.gamma_dash])
        self.coils_curvature= vmap(compute_curvature)(self.gamma_dash, coils.gamma_dashdash)        
        self.r_axis=jnp.mean(jnp.sqrt(vmap(lambda dofs: dofs[0, 0]**2 + dofs[1, 0]**2)(self.coils.dofs_curves)))
        self.z_axis=jnp.mean(vmap(lambda dofs: dofs[2, 0])(self.coils.dofs_curves))


    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return 1.
    
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
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points

@jit
def d_dtheta_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    dF = (1j * k) * Fk
    return jnp.fft.ifft(dF, axis=-1).real * (2*jnp.pi)

@jit
def d2_dtheta2_fft(f_theta):
    ntheta = f_theta.shape[-1]
    k = jnp.fft.fftfreq(ntheta, d=1.0/ntheta)     # integer modes
    Fk = jnp.fft.fft(f_theta, axis=-1)
    d2F = -(k**2) * Fk
    return jnp.fft.ifft(d2F, axis=-1).real * (2*jnp.pi)**2

@jit
def gamma_dash_from_gamma(gamma):
    return jnp.stack([
        d_dtheta_fft(gamma[..., 0]),
        d_dtheta_fft(gamma[..., 1]),
        d_dtheta_fft(gamma[..., 2]),
    ], axis=-1)

@jit
def gamma_dashdash_from_gamma(gamma):
    return jnp.stack([
        d2_dtheta2_fft(gamma[..., 0]),
        d2_dtheta2_fft(gamma[..., 1]),
        d2_dtheta2_fft(gamma[..., 2]),
    ], axis=-1)

class BiotSavart_from_gamma():
    def __init__(self, gamma,gamma_dash=None,gamma_dashdash=None, currents=None):
        if currents is None:
            currents = jnp.ones(len(gamma))
        else:
            currents = currents
        self.currents = currents
        self.gamma = gamma
        self.r_axis=jnp.average(jnp.linalg.norm(jnp.average(gamma,axis=1)[:,0:2],axis=1))
        self.z_axis=jnp.average(jnp.average(gamma,axis=1)[:,2])
        if gamma_dash is not None:
            self.gamma_dash = gamma_dash
        else:
            self.gamma_dash = gamma_dash_from_gamma(gamma)
        self.coils_length=jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in self.gamma_dash])
        if gamma_dashdash is not None:
            self.gamma_dashdash = gamma_dashdash
        else:
            self.gamma_dashdash = gamma_dashdash_from_gamma(gamma)
        self.coils_curvature= vmap(compute_curvature)(self.gamma_dash, self.gamma_dashdash)

    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        return 1.
    
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
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)
    
    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        return points



class Vmec():
    def __init__(self, wout_filename, ntheta=50, nphi=50, close=True, range_torus='full torus'):
        self.wout_filename = wout_filename
        from netCDF4 import Dataset
        self.nc = Dataset(self.wout_filename)
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
        self.z_axis=self.zmns[0,0]
        self.mpol = int(jnp.max(self.xm)+1)
        self.ntor = int(jnp.max(jnp.abs(self.xn)) / self.nfp)
        self.range_torus = range_torus
        self._surface = SurfaceRZFourier(self, ntheta=ntheta, nphi=nphi, close=close, range_torus=range_torus)
        self.Aminor_p = jnp.array(self.nc.variables["Aminor_p"][:])
        #self._classifier=SurfaceClassifier(self._surface,p=1,h=0.05)
        
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
    def sqrtg(self, points):
        s, theta, phi = points
        gmnc_interp = vmap(lambda row: jnp.interp(s, self.s_half_grid, row, left='extrapolate'), in_axes=1)(self.gmnc[1:])
        cosangle_nyq = jnp.cos(self.xm_nyq * theta - self.xn_nyq * phi)
        sqrt_g_vmec = jnp.dot(gmnc_interp, cosangle_nyq)
        return sqrt_g_vmec



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
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)

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

class near_axis():
    def __init__(self, rc=jnp.array([1, 0.1]), zs=jnp.array([0, 0.1]), etabar=1.0,
                    B0=1, sigma0=0, I2=0, nphi=31, spsi=1, sG=1, nfp=2, order='r1', B2c=0, p2=0):
        assert nphi % 2 == 1, 'nphi must be odd'
        self.rc = jnp.array(rc)
        self.zs = jnp.array(zs)
        self.etabar = etabar
        self.nphi = nphi
        self.sigma0 = sigma0
        self.I2 = I2
        self.spsi = spsi
        self.sG = sG
        self.B0 = B0
        self.nfp = nfp
        self.order = order # not used
        self.B2c = B2c # not used
        self.p2 = p2 # not used
        
        self._dofs = jnp.concatenate((jnp.ravel(self.rc), jnp.ravel(self.zs), jnp.array([etabar])))
        
        self.phi = jnp.linspace(0, 2 * jnp.pi / self.nfp, self.nphi, endpoint=False)
        self.nfourier = max(len(self.rc), len(self.zs))
        
        parameters = self.calculate(self.rc, self.zs, self.etabar)
        (self.R0, self.Z0, self.sigma, self.elongation, self.B_axis, self.grad_B_axis, self.axis_length, self.iota, self.iotaN, self.G0,
         self.helicity, self.X1c_untwisted, self.X1s_untwisted, self.Y1s_untwisted, self.Y1c_untwisted,
         self.normal_R, self.normal_phi, self.normal_z, self.binormal_R, self.binormal_phi, self.binormal_z,
         self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature, self.varphi, self.R0p, self.Z0p) = parameters

    @property
    def dofs(self):
        return self._dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        self._dofs = jnp.array(new_dofs)
        self.rc = self._dofs[:self.nfourier]
        self.zs = self._dofs[self.nfourier:2*self.nfourier]
        self.etabar = self._dofs[-1]
        parameters = self.calculate(self.rc, self.zs, self.etabar)
        (self.R0, self.Z0, self.sigma, self.elongation, self.B_axis, self.grad_B_axis, self.axis_length, self.iota, self.iotaN, self.G0,
         self.helicity, self.X1c_untwisted, self.X1s_untwisted, self.Y1s_untwisted, self.Y1c_untwisted,
         self.normal_R, self.normal_z, self.normal_phi, self.binormal_R, self.binormal_z, self.binormal_phi,
         self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature, self.varphi, self.R0p, self.Z0p) = parameters

    @property
    def x(self):
        return self._dofs
    
    @x.setter
    def x(self, new_x):
        self.dofs = new_x
        
    def _tree_flatten(self):
        children = (self.rc, self.zs, self.etabar, self.B0, self.sigma0, self.I2)  # arrays / dynamic values
        aux_data = {"nphi": self.nphi, "spsi": self.spsi, "sG": self.sG,
                    "nfp": self.nfp, "order": self.order, "B2c": self.B2c, "p2": self.p2}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    




    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        r, theta, phi = points
        AbsB = self.AbsB(points)
        return r*self.B0*(self.G0+self.iota*self.I2)/(AbsB*AbsB)
    
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        r, theta, phi = points
        Br = 0
        Btheta = r*r*self.I2
        Bphi = self.G0
        return jnp.array([Br, Btheta, Bphi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        r, theta, phi = points
        jac = self.jacobian(points)
        AbsB = self.AbsB(points)
        Bphi = r*AbsB/jac
        return jnp.array([0, self.iotaN * Bphi, Bphi])
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        r, theta, phi = points
        return self.B0*(1 + r*self.etabar*jnp.cos(theta))
    

    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)
        
    @partial(jit, static_argnames=['self'])
    def jacobian(self, points):
        r, theta, phi = points
        AbsB = self.AbsB(points)
        return r*self.B0*(self.G0+self.iota*self.I2)/(AbsB*AbsB)
        
    @partial(jit, static_argnames=['self'])
    def calculate(self, rc, zs, etabar):
        phi = self.phi
        nphi = self.nphi
        nfp = self.nfp
        nfourier = self.nfourier
        spsi = self.spsi
        sG = self.sG
        B0 = self.B0
        sigma0 = self.sigma0
        I2 = self.I2
        d_phi = phi[1] - phi[0]
        
        n_values = jnp.arange(nfourier) * nfp

        @jit
        def compute_terms(jn):
            n = n_values[jn]
            sinangle = jnp.sin(n * phi)
            cosangle = jnp.cos(n * phi)
            return jnp.array([rc[jn] * cosangle, zs[jn] * sinangle,
                rc[jn] * (-n * sinangle), zs[jn] * (n * cosangle),
                rc[jn] * (-n * n * cosangle), zs[jn] * (-n * n * sinangle),
                rc[jn] * (n * n * n * sinangle), zs[jn] * (-n * n * n * cosangle)])

        @jit
        def spectral_diff_matrix_jax():
            n=nphi
            xmin=0
            xmax=2 * jnp.pi / nfp
            h = 2 * jnp.pi / n
            kk = jnp.arange(1, n)
            n_half = n // 2
            topc = 1 / jnp.sin(jnp.arange(1, n_half + 1) * h / 2)
            temp = jnp.concatenate((topc, jnp.flip(topc[:n_half])))
            col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
            row1 = -col1
            vals = jnp.concatenate((row1[-1:0:-1], col1))
            a, b = jnp.ogrid[0:len(col1), len(row1)-1:-1:-1]
            return 2 * jnp.pi / (xmax - xmin) * vals[a + b]

        @jit
        def determine_helicity(normal_cylindrical):
            x_positive = normal_cylindrical[:, 0] >= 0
            z_positive = normal_cylindrical[:, 2] >= 0
            quadrant = 1 * x_positive * z_positive + 2 * (~x_positive) * z_positive \
                    + 3 * (~x_positive) * (~z_positive) + 4 * x_positive * (~z_positive)
            quadrant = jnp.append(quadrant, quadrant[0])
            delta_quadrant = quadrant[1:] - quadrant[:-1]
            increment = jnp.sum((quadrant[:-1] == 4) & (quadrant[1:] == 1))
            decrement = jnp.sum((quadrant[:-1] == 1) & (quadrant[1:] == 4))
            return (jnp.sum(delta_quadrant) + increment - decrement) * spsi * sG

        summed_values = jnp.sum(jax.vmap(compute_terms)(jnp.arange(nfourier)), axis=0)

        R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp = summed_values
        d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        d_l_d_varphi = abs_G0_over_B0
        G0 = sG * abs_G0_over_B0 * B0

        d_r_d_phi_cylindrical = jnp.stack([R0p, R0, Z0p]).T
        d2_r_d_phi2_cylindrical = jnp.stack([R0pp - R0, 2 * R0p, Z0pp]).T
        d3_r_d_phi3_cylindrical = jnp.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).T


        d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] \
                                    +d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, None] * d_l_d_phi[:, None])
        curvature = jnp.sqrt(jnp.sum(d_tangent_d_l_cylindrical**2, axis=1))
        axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
        varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

        tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, None]
        normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, None]
        binormal_cylindrical = jnp.cross(tangent_cylindrical, normal_cylindrical)

        torsion_numerator = jnp.sum(d_r_d_phi_cylindrical * jnp.cross(d2_r_d_phi2_cylindrical, d3_r_d_phi3_cylindrical), axis=1)
        torsion_denominator = jnp.sum(jnp.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical)**2, axis=1)
        torsion = torsion_numerator / torsion_denominator

        d_d_phi = spectral_diff_matrix_jax()
        d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
        d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
        helicity = determine_helicity(normal_cylindrical)

        @jit
        def replace_first_element(x, new_value):
            return jnp.concatenate([jnp.array([new_value]), x[1:]])

        @jit
        def sigma_equation_residual(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            etaOcurv2 = etabar**2 / curvature**2
            return jnp.matmul(d_d_varphi, sigma) \
                + (iota + helicity * nfp) * (etaOcurv2**2 + 1 + sigma**2) \
                - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0

        @jit
        def sigma_equation_jacobian(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            etaOcurv2 = etabar**2 / curvature**2
            jac = d_d_varphi + (iota + helicity * nfp) * 2 * jnp.diag(sigma)
            return jac.at[:, 0].set(etaOcurv2**2 + 1 + sigma**2)

        @partial(jit, static_argnums=(1,))
        def newton(x0, niter=5):
            def body_fun(i, x):
                residual = sigma_equation_residual(x)
                jacobian = sigma_equation_jacobian(x)
                step = jax.scipy.linalg.solve(jacobian, -residual)
                return x + step
            x = jax.lax.fori_loop(0, niter, body_fun, x0)
            return x

        x0 = jnp.full(nphi, sigma0)
        x0 = replace_first_element(x0, 0.)
        sigma = newton(x0)
        iota = sigma[0]
        iotaN = iota + helicity * nfp
        sigma = replace_first_element(sigma, sigma0)

        X1c = etabar / curvature
        Y1s = sG * spsi * curvature / etabar
        Y1c = sG * spsi * curvature * sigma / etabar
        p = + X1c * X1c + Y1s * Y1s + Y1c * Y1c
        q = - X1c * Y1s
        elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))
        
        B_axis_cylindrical = sG * B0 * tangent_cylindrical.T
        B_x = jnp.cos(phi) * B_axis_cylindrical[0] - jnp.sin(phi) * B_axis_cylindrical[1]
        B_y = jnp.sin(phi) * B_axis_cylindrical[0] + jnp.cos(phi) * B_axis_cylindrical[1]
        B_z = B_axis_cylindrical[2]
        B_axis = jnp.array([B_x, B_y, B_z])

        d_X1c_d_varphi = -etabar / curvature**2
        d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
        d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)
        t = tangent_cylindrical.transpose()
        n = normal_cylindrical.transpose()
        b = binormal_cylindrical.transpose()
        d_X1c_d_varphi = jnp.matmul(d_d_varphi, X1c)
        d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
        d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)
        factor = spsi * B0 / d_l_d_varphi
        tn = sG * B0 * curvature
        nt = tn
        bb = factor * (X1c * d_Y1s_d_varphi - iotaN * X1c * Y1c)
        nn = factor * (d_X1c_d_varphi * Y1s + iotaN * X1c * Y1c)
        bn = factor * (-sG * spsi * d_l_d_varphi * torsion - iotaN * X1c * X1c)
        nb = factor * (d_Y1c_d_varphi * Y1s - d_Y1s_d_varphi * Y1c + sG * spsi * d_l_d_varphi * torsion + iotaN * (Y1s * Y1s + Y1c * Y1c))
        tt = 0
        nablaB = jnp.array([[
                            nn * n[i] * n[j] \
                            + bn * b[i] * n[j] + nb * n[i] * b[j] \
                            + bb * b[i] * b[j] \
                            + tn * t[i] * n[j] + nt * n[i] * t[j] \
                            + tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        grad_B_axis = jnp.array([
            [cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
            sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
            sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
            cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
            sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
            [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
            nablaB[2, 2]]
                ])
        
        grad_B_colon_grad_B = tn * tn + nt * nt \
                            + bb * bb + nn * nn \
                            + nb * nb + bn * bn \
                            + tt * tt
        L_grad_B = self.B0 * jnp.sqrt(2 / grad_B_colon_grad_B)
        inv_L_grad_B = 1.0 / L_grad_B
        
        X1c_untwisted = jnp.where(helicity == 0, X1c, X1c * jnp.cos(-helicity * nfp * varphi))
        X1s_untwisted = jnp.where(helicity == 0, 0 * X1c, X1c * jnp.sin(-helicity * nfp * varphi))
        Y1s_untwisted = jnp.where(helicity == 0, Y1s, Y1s * jnp.cos(-helicity * nfp * varphi) + Y1c * jnp.sin(-helicity * nfp * varphi))
        Y1c_untwisted = jnp.where(helicity == 0, Y1c, Y1s * (-jnp.sin(-helicity * nfp * varphi)) + Y1c * jnp.cos(-helicity * nfp * varphi))
        
        normal_R = normal_cylindrical[:,0]
        normal_phi = normal_cylindrical[:,1]
        normal_z = normal_cylindrical[:,2]
        binormal_R = binormal_cylindrical[:,0]
        binormal_phi = binormal_cylindrical[:,1]
        binormal_z = binormal_cylindrical[:,2]
        
        return (R0, Z0, sigma, elongation, B_axis, grad_B_axis, axis_length, iota, iotaN, G0,
                helicity, X1c_untwisted, X1s_untwisted, Y1s_untwisted, Y1c_untwisted,
                normal_R, normal_phi, normal_z, binormal_R, binormal_phi, binormal_z,
                L_grad_B, inv_L_grad_B, torsion, curvature, varphi, R0p, Z0p)
    
    @jit
    def residual_phi0_of_theta_varphi_func(self, phi_0, r, theta, varphi):
        # Residual = phi + nu - varphi = 0
        # Compute phi off axis
        X_at_this_theta = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi = self.Frenet_to_cylindrical_1_point(phi_0, X_at_this_theta, Y_at_this_theta)
        # phi = phi + 2 * jnp.pi * (phi < 0) - 2 * jnp.pi * (phi > 2 * jnp.pi)
        # Compute nu = nu0 + r (nu1c cos theta + nu1s sin theta)
        nu0 = self.interpolated_array_at_point(self.varphi-self.phi, phi_0)
        X1c = self.interpolated_array_at_point(self.X1c_untwisted, phi_0)
        X1s = self.interpolated_array_at_point(self.X1s_untwisted, phi_0)
        Y1c = self.interpolated_array_at_point(self.Y1c_untwisted, phi_0)
        Y1s = self.interpolated_array_at_point(self.Y1s_untwisted, phi_0)
        bR = self.interpolated_array_at_point(self.binormal_R, phi_0)
        bZ = self.interpolated_array_at_point(self.binormal_z, phi_0)
        nR = self.interpolated_array_at_point(self.normal_R, phi_0)
        nZ = self.interpolated_array_at_point(self.normal_z, phi_0)
        R0 = self.interpolated_array_at_point(self.R0, phi_0)
        R0p = self.interpolated_array_at_point(self.R0p, phi_0)
        Z0p = self.interpolated_array_at_point(self.Z0p, phi_0)
        nu1c = X1c * (bR * Z0p - bZ * R0p)/R0 + Y1c * (nZ * R0p - nR * Z0p)/R0
        nu1s = X1s * (bR * Z0p - bZ * R0p)/R0 + Y1s * (nZ * R0p - nR * Z0p)/R0
        nu = nu0 + r * (nu1c * jnp.cos(theta) + nu1s * jnp.sin(theta))
        # Return residual
        return phi + nu - varphi
    
    @jit
    def phi_of_theta_varphi(self, r, theta, varphi):
        residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta, r=r, varphi=varphi)
        phi_on_axis = lax.custom_root(residual, varphi, newton, lambda g, y: y / g(1.0))
        X_at_this_theta = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi_off_axis = self.Frenet_to_cylindrical_1_point(phi_on_axis, X_at_this_theta, Y_at_this_theta)
        return phi_off_axis# + 2 * jnp.pi * (phi_off_axis < 0) - 2 * jnp.pi * (phi_off_axis > 2 * jnp.pi)
        
    @jit
    def interpolated_array_at_point(self,array,point):
        sp=jnp.interp(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), period=2*jnp.pi/self.nfp)[0]
        ## Using interpax would make the interpolation slightly more accurate, but it is too slow at the moment
        # sp=interpax.interp1d(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), method="cubic", period=2*jnp.pi/self.nfp)[0]
        return sp
        
    @jit
    def Frenet_to_cylindrical_residual_func(self,phi0, phi_target, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
        normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        Frenet_to_cylindrical_residual = jnp.arctan2(total_y, total_x) - phi_target
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual > jnp.pi, Frenet_to_cylindrical_residual - 2 * jnp.pi, Frenet_to_cylindrical_residual)
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual <-jnp.pi, Frenet_to_cylindrical_residual + 2 * jnp.pi, Frenet_to_cylindrical_residual)
        return Frenet_to_cylindrical_residual

    @jit
    def Frenet_to_cylindrical_1_point(self, phi0, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        z0_at_phi0   = self.interpolated_array_at_point(self.Z0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        normal_z     = self.interpolated_array_at_point(self.normal_z,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        binormal_z   = self.interpolated_array_at_point(self.binormal_z,phi0)
        normal_x   = normal_R   * cosphi0 - normal_phi * sinphi0
        normal_y   = normal_R   * sinphi0 + normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z
        total_R = jnp.sqrt(total_x * total_x + total_y * total_y)
        total_phi=jnp.arctan2(total_y, total_x)
        return total_R, total_z, total_phi
    
    @partial(jit, static_argnames=['ntheta'])
    def Frenet_to_cylindrical(self, r, ntheta=20, phi_is_varphi=False):
        nphi_conversion = self.nphi
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / self.nfp, nphi_conversion, endpoint=False)

        def compute_for_theta(theta_j):
            costheta = jnp.cos(theta_j)
            sintheta = jnp.sin(theta_j)
            X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
            Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)

            def compute_for_phi(phi_target):
                
                def residual(z):
                    return jax.lax.cond(
                        phi_is_varphi,
                        # Branch A: solve for phi0 so that phi+nu-varphi = 0
                        lambda _: self.residual_phi0_of_theta_varphi_func(
                            z, r=r, theta=theta_j, varphi=phi_target
                        ),
                        # Branch B: solve for phi so Frenet_to_cylindrical_residual_func = 0
                        lambda _: self.Frenet_to_cylindrical_residual_func(
                            z, phi_target=phi_target,
                            X_at_this_theta=X_at_this_theta,
                            Y_at_this_theta=Y_at_this_theta
                        ),
                        operand=None
                    )
                # residual = partial(self.Frenet_to_cylindrical_residual_func, phi_target=phi_target,
                #                 X_at_this_theta=X_at_this_theta, Y_at_this_theta=Y_at_this_theta)
                # residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta_j, r=r, varphi=phi_target)
                
                phi0_solution = lax.custom_root(residual, phi_target, newton, lambda g, y: y / g(1.0))
                
                final_R, final_Z, _ = self.Frenet_to_cylindrical_1_point(phi0_solution, X_at_this_theta, Y_at_this_theta)
                return final_R, final_Z, phi0_solution

            return vmap(compute_for_phi)(phi_conversion)

        R_2D, Z_2D, phi0_2D = vmap(compute_for_theta)(theta)
        return R_2D, Z_2D, phi0_2D


    @partial(jit, static_argnames=['mpol', 'ntor'])
    def to_Fourier(self, R_2D, Z_2D, nfp, mpol, ntor):
        ntheta, nphi_conversion = R_2D.shape
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / nfp, nphi_conversion, endpoint=False)
        
        phi2d, theta2d = jnp.meshgrid(phi_conversion, theta, indexing='xy')
        factor = 2 / (ntheta * nphi_conversion)

        def compute_RBC_ZBS(m, n):
            angle = m * theta2d - n * nfp * phi2d
            sinangle, cosangle = jnp.sin(angle), jnp.cos(angle)

            # Conditional scaling of factor2
            factor2 = jax.lax.cond(
                (ntheta % 2 == 0) & (m == (ntheta / 2)),
                lambda _: factor / 2, lambda _: factor,
                operand=None)

            factor2 = jax.lax.cond(
                (nphi_conversion % 2 == 0) & (abs(n) == (nphi_conversion / 2)),
                lambda _: factor2 / 2, lambda _: factor2,
                operand=None)

            return jnp.sum(R_2D * cosangle * factor2), jnp.sum(Z_2D * sinangle * factor2)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.concatenate([jnp.array([1]), jnp.arange(-ntor, ntor + 1)]) if mpol == 0 else jnp.arange(-ntor, ntor + 1)
        RBC, ZBS = vmap(lambda n: vmap(lambda m: compute_RBC_ZBS(m, n))(m_vals))(n_vals)

        RBC = RBC.at[ntor, 0].set(jnp.sum(R_2D) / (ntheta * nphi_conversion))
        ZBS = ZBS.at[:ntor, 0].set(0)
        RBC = RBC.at[:ntor, 0].set(0)
        return RBC, ZBS

    @partial(jit, static_argnames=['ntheta_fourier', 'mpol', 'ntor', 'ntheta', 'nphi', 'phi_is_varphi'])
    def get_boundary(self, r=0.1, ntheta=30, nphi=120, ntheta_fourier=20, mpol=5, ntor=5, phi_is_varphi=False, phi_offset=0.0):
        R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier, phi_is_varphi=phi_is_varphi)
        RBC, ZBS = self.to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor)

        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        
        # phi1D = jax.lax.cond(
        #     phi_is_varphi,
        #     lambda _: jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False),
        #     lambda _: jnp.linspace(0, 2 * jnp.pi, nphi),
        #     operand=None
        # )
        # phi1D += phi_offset
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi) + phi_offset
        
        phi2D_original, theta2D = jnp.meshgrid(phi1D, theta1D, indexing='ij')
        
        phi2D = jax.lax.cond(
            phi_is_varphi,
            lambda _: vmap(lambda theta_row, varphi_row: vmap(lambda theta, varphi: self.phi_of_theta_varphi(r, theta, varphi))(theta_row, varphi_row))(theta2D, phi2D_original),
            lambda _: phi2D_original,
            operand=None
        )
        
        def compute_RZ(m, n):
            angle = m * theta2D - n * self.nfp * phi2D_original
            return RBC[n + ntor, m] * jnp.cos(angle), ZBS[n + ntor, m] * jnp.sin(angle)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.arange(-ntor, ntor + 1)

        R_2Dnew, Z_2Dnew = vmap(lambda m: vmap(lambda n: compute_RZ(m, n))(n_vals))(m_vals)
        R_2Dnew, Z_2Dnew = R_2Dnew.sum(axis=(0, 1)), Z_2Dnew.sum(axis=(0, 1))

        x_2D_plot = R_2Dnew.T * jnp.cos(phi2D.T)
        y_2D_plot = R_2Dnew.T * jnp.sin(phi2D.T)
        z_2D_plot = Z_2Dnew.T
        return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew.T
    
    @partial(jit, static_argnames=['self'])
    def B_mag(self, r, theta, phi):
        return self.B0*(1 + r * self.etabar * jnp.cos(theta - (self.iota - self.iotaN) * phi))

    def plot(self, r=0.1, ntheta=40, nphi=120, ntheta_fourier=20, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        kwargs.setdefault('alpha', 1)
        import matplotlib.pyplot as plt 
        from matplotlib import cm
        import matplotlib.colors as clr
        from matplotlib.colors import LightSource
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')   
        x_2D_plot, y_2D_plot, z_2D_plot, _ = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2D, theta2D = jnp.meshgrid(phi1D, theta1D)
        import numpy as np
        Bmag = np.array(self.B_mag(r, theta2D, phi2D))
        norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
        cmap = cm.viridis
        ls = LightSource(azdeg=0, altdeg=10)
        cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=cmap_plot,
                        rstride=1, cstride=1, antialiased=False,
                        linewidth=0, shade=False, **kwargs)
        if ax is None or ax.name != "3d":
            ax.dist = 7
            ax.elev = 5
            ax.azim = 45
            cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            cbar = plt.colorbar(m, cax=cbar_ax)
            cbar.ax.set_title(r'$|B| [T]$')
            ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()
            
    def to_vtk(self, filename, r=0.1, ntheta=40, nphi=120, ntheta_fourier=20, extra_data=None, field=None):
        try: import numpy as np
        except ImportError: raise ImportError("The 'numpy' library is required. Please install it using 'pip install numpy'.")
        try: from pyevtk.hl import gridToVTK
        except ImportError: raise ImportError("The 'pyevtk' library is required. Please install it using 'pip install pyevtk'.")
        x, y, z, _ = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
        x = np.array(x.T.reshape((1, nphi, ntheta)).copy())
        y = np.array(y.T.reshape((1, nphi, ntheta)).copy())
        z = np.array(z.T.reshape((1, nphi, ntheta)).copy())
        pointData = {}
        if field is not None:
            boundary = np.array([x, y, z]).transpose(1, 2, 3, 0)[0]
            B_BiotSavart = np.array(vmap(lambda surf: vmap(lambda x: field.AbsB(x))(surf))(boundary)).reshape((1, nphi, ntheta)).copy()
            pointData["B_BiotSavart"] = B_BiotSavart
        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2D, theta2D = jnp.meshgrid(phi1D, theta1D)
        Bmag = np.array(self.B_mag(r, theta2D, phi2D)).T.reshape((1, nphi, ntheta)).copy()
        pointData["B_NearAxis"]=Bmag
        if extra_data is not None:
            pointData = {**pointData, **extra_data}
        gridToVTK(str(filename), x, y, z, pointData=pointData)
            
tree_util.register_pytree_node(near_axis,
                               near_axis._tree_flatten,
                               near_axis._tree_unflatten)











class near_axis_test():
    def __init__(self, rc=jnp.array([1, 0.1]), zs=jnp.array([0, 0.1]), etabar=1.0,
                    B0=1, sigma0=0, I2=0, nphi=31, spsi=1, sG=1, nfp=2, order='r1', B2c=0, p2=0):
        assert nphi % 2 == 1, 'nphi must be odd'
        self.rc = jnp.array(rc)
        self.zs = jnp.array(zs)
        self.etabar = etabar
        self.nphi = nphi
        self.sigma0 = sigma0
        self.I2 = I2
        self.spsi = spsi
        self.sG = sG
        self.B0 = B0
        self.nfp = nfp
        self.order = order # not used
        self.B2c = B2c # not used
        self.p2 = p2 # not used
        
        self._dofs = jnp.concatenate((jnp.ravel(self.rc), jnp.ravel(self.zs), jnp.array([etabar])))
        
        self.phi = jnp.linspace(0, 2 * jnp.pi / self.nfp, self.nphi, endpoint=False)
        self.nfourier = max(len(self.rc), len(self.zs))
        
        parameters = self.calculate(self.rc, self.zs, self.etabar)
        (self.R0, self.Z0, self.sigma, self.elongation, self.B_axis, self.grad_B_axis, self.axis_length, self.iota, self.iotaN, self.G0,
         self.helicity, self.X1c_untwisted, self.X1s_untwisted, self.Y1s_untwisted, self.Y1c_untwisted,
         self.normal_R, self.normal_phi, self.normal_z, self.binormal_R, self.binormal_phi, self.binormal_z,
         self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature, self.varphi, self.R0p, self.Z0p) = parameters



    @partial(jit, static_argnames=['self'])
    def jacobian(self, points):
        r, theta, phi = points
        #AbsB = self.AbsB(points)
        #return r*self.B0*(self.G0+self.iota*self.I2)/(AbsB*AbsB)
        AbsB = self.AbsB(points)
        return r*self.B0*(self.G0+self.iota*r*r*self.I2)/(AbsB*AbsB)    
    
    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        r, theta, phi = points
        #AbsB = self.AbsB(points)
        #return r*self.B0*(self.G0+self.iota*self.I2)/(AbsB*AbsB)
        AbsB = self.AbsB(points)
        return r*self.B0*(self.G0+self.iota*r*r*self.I2)/(AbsB*AbsB)    
    
    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        r, theta, phi = points
        Br = 0
        Btheta = r*r*self.I2
        Bphi = self.G0
        return jnp.array([Br, Btheta, Bphi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        r, theta, phi = points
        jac = self.sqrtg(points)
        Bphi = r*self.B0/jac
        #return jnp.array([0, self.iotaN * Bphi, Bphi])
        return jnp.array([0, self.iota * Bphi, Bphi])    
    

    @partial(jit, static_argnames=['self'])
    def B_mag(self, r, theta, phi):
        return self.B0*(1 + r * self.etabar * jnp.cos(theta - (self.iota - self.iotaN) * phi))
        
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        r, theta, phi = points
        return self.B0*(1 + r*self.etabar*jnp.cos(theta-self.helicity*self.nfp*phi))
    

    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)
        


    @partial(jit, static_argnames=['self'])
    def calculate(self, rc, zs, etabar):
        phi = self.phi
        nphi = self.nphi
        nfp = self.nfp
        nfourier = self.nfourier
        spsi = self.spsi
        sG = self.sG
        B0 = self.B0
        sigma0 = self.sigma0
        I2 = self.I2
        d_phi = phi[1] - phi[0]
        
        n_values = jnp.arange(nfourier) * nfp

        @jit
        def compute_terms(jn):
            n = n_values[jn]
            sinangle = jnp.sin(n * phi)
            cosangle = jnp.cos(n * phi)
            return jnp.array([rc[jn] * cosangle, zs[jn] * sinangle,
                rc[jn] * (-n * sinangle), zs[jn] * (n * cosangle),
                rc[jn] * (-n * n * cosangle), zs[jn] * (-n * n * sinangle),
                rc[jn] * (n * n * n * sinangle), zs[jn] * (-n * n * n * cosangle)])

        @jit
        def spectral_diff_matrix_jax():
            n=nphi
            xmin=0
            xmax=2 * jnp.pi / nfp
            h = 2 * jnp.pi / n
            kk = jnp.arange(1, n)
            n_half = n // 2
            topc = 1 / jnp.sin(jnp.arange(1, n_half + 1) * h / 2)
            temp = jnp.concatenate((topc, jnp.flip(topc[:n_half])))
            col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
            row1 = -col1
            vals = jnp.concatenate((row1[-1:0:-1], col1))
            a, b = jnp.ogrid[0:len(col1), len(row1)-1:-1:-1]
            return 2 * jnp.pi / (xmax - xmin) * vals[a + b]

        @jit
        def determine_helicity(normal_cylindrical):
            x_positive = normal_cylindrical[:, 0] >= 0
            z_positive = normal_cylindrical[:, 2] >= 0
            quadrant = 1 * x_positive * z_positive + 2 * (~x_positive) * z_positive \
                    + 3 * (~x_positive) * (~z_positive) + 4 * x_positive * (~z_positive)
            quadrant = jnp.append(quadrant, quadrant[0])
            delta_quadrant = quadrant[1:] - quadrant[:-1]
            increment = jnp.sum((quadrant[:-1] == 4) & (quadrant[1:] == 1))
            decrement = jnp.sum((quadrant[:-1] == 1) & (quadrant[1:] == 4))
            return (jnp.sum(delta_quadrant) + increment - decrement) * spsi * sG

        summed_values = jnp.sum(jax.vmap(compute_terms)(jnp.arange(nfourier)), axis=0)

        R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp = summed_values
        d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        d_l_d_varphi = abs_G0_over_B0
        G0 = sG * abs_G0_over_B0 * B0

        d_r_d_phi_cylindrical = jnp.stack([R0p, R0, Z0p]).T
        d2_r_d_phi2_cylindrical = jnp.stack([R0pp - R0, 2 * R0p, Z0pp]).T
        d3_r_d_phi3_cylindrical = jnp.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).T


        d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] \
                                    +d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, None] * d_l_d_phi[:, None])
        curvature = jnp.sqrt(jnp.sum(d_tangent_d_l_cylindrical**2, axis=1))
        axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
        varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

        tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, None]
        normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, None]
        binormal_cylindrical = jnp.cross(tangent_cylindrical, normal_cylindrical)

        torsion_numerator = jnp.sum(d_r_d_phi_cylindrical * jnp.cross(d2_r_d_phi2_cylindrical, d3_r_d_phi3_cylindrical), axis=1)
        torsion_denominator = jnp.sum(jnp.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical)**2, axis=1)
        torsion = torsion_numerator / torsion_denominator

        d_d_phi = spectral_diff_matrix_jax()
        d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
        d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
        helicity = determine_helicity(normal_cylindrical)

        @jit
        def replace_first_element(x, new_value):
            return jnp.concatenate([jnp.array([new_value]), x[1:]])

        @jit
        def sigma_equation_residual(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            etaOcurv2 = etabar**2 / curvature**2
            return jnp.matmul(d_d_varphi, sigma) \
                + (iota + helicity * nfp) * (etaOcurv2**2 + 1 + sigma**2) \
                - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0

        @jit
        def sigma_equation_jacobian(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            etaOcurv2 = etabar**2 / curvature**2
            jac = d_d_varphi + (iota + helicity * nfp) * 2 * jnp.diag(sigma)
            return jac.at[:, 0].set(etaOcurv2**2 + 1 + sigma**2)

        @partial(jit, static_argnums=(1,))
        def newton(x0, niter=5):
            def body_fun(i, x):
                residual = sigma_equation_residual(x)
                jacobian = sigma_equation_jacobian(x)
                step = jax.scipy.linalg.solve(jacobian, -residual)
                return x + step
            x = jax.lax.fori_loop(0, niter, body_fun, x0)
            return x

        x0 = jnp.full(nphi, sigma0)
        x0 = replace_first_element(x0, 0.)
        sigma = newton(x0)
        iota = sigma[0]
        iotaN = iota + helicity * nfp
        sigma = replace_first_element(sigma, sigma0)

        X1c = etabar / curvature
        Y1s = sG * spsi * curvature / etabar
        Y1c = sG * spsi * curvature * sigma / etabar
        p = + X1c * X1c + Y1s * Y1s + Y1c * Y1c
        q = - X1c * Y1s
        elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))
        
        B_axis_cylindrical = sG * B0 * tangent_cylindrical.T
        B_x = jnp.cos(phi) * B_axis_cylindrical[0] - jnp.sin(phi) * B_axis_cylindrical[1]
        B_y = jnp.sin(phi) * B_axis_cylindrical[0] + jnp.cos(phi) * B_axis_cylindrical[1]
        B_z = B_axis_cylindrical[2]
        B_axis = jnp.array([B_x, B_y, B_z])

        d_X1c_d_varphi = -etabar / curvature**2
        d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
        d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)
        t = tangent_cylindrical.transpose()
        n = normal_cylindrical.transpose()
        b = binormal_cylindrical.transpose()
        d_X1c_d_varphi = jnp.matmul(d_d_varphi, X1c)
        d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
        d_Y1c_d_varphi = jnp.matmul(d_d_varphi, Y1c)
        factor = spsi * B0 / d_l_d_varphi
        tn = sG * B0 * curvature
        nt = tn
        bb = factor * (X1c * d_Y1s_d_varphi - iotaN * X1c * Y1c)
        nn = factor * (d_X1c_d_varphi * Y1s + iotaN * X1c * Y1c)
        bn = factor * (-sG * spsi * d_l_d_varphi * torsion - iotaN * X1c * X1c)
        nb = factor * (d_Y1c_d_varphi * Y1s - d_Y1s_d_varphi * Y1c + sG * spsi * d_l_d_varphi * torsion + iotaN * (Y1s * Y1s + Y1c * Y1c))
        tt = 0
        nablaB = jnp.array([[
                            nn * n[i] * n[j] \
                            + bn * b[i] * n[j] + nb * n[i] * b[j] \
                            + bb * b[i] * b[j] \
                            + tn * t[i] * n[j] + nt * n[i] * t[j] \
                            + tt * t[i] * t[j]
                        for i in range(3)] for j in range(3)])
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        grad_B_axis = jnp.array([
            [cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
            sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
            sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
            cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
            sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
            [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
            nablaB[2, 2]]
                ])
        
        grad_B_colon_grad_B = tn * tn + nt * nt \
                            + bb * bb + nn * nn \
                            + nb * nb + bn * bn \
                            + tt * tt
        L_grad_B = self.B0 * jnp.sqrt(2 / grad_B_colon_grad_B)
        inv_L_grad_B = 1.0 / L_grad_B
        
        X1c_untwisted = jnp.where(helicity == 0, X1c, X1c * jnp.cos(-helicity * nfp * varphi))
        X1s_untwisted = jnp.where(helicity == 0, 0 * X1c, X1c * jnp.sin(-helicity * nfp * varphi))
        Y1s_untwisted = jnp.where(helicity == 0, Y1s, Y1s * jnp.cos(-helicity * nfp * varphi) + Y1c * jnp.sin(-helicity * nfp * varphi))
        Y1c_untwisted = jnp.where(helicity == 0, Y1c, Y1s * (-jnp.sin(-helicity * nfp * varphi)) + Y1c * jnp.cos(-helicity * nfp * varphi))
        
        normal_R = normal_cylindrical[:,0]
        normal_phi = normal_cylindrical[:,1]
        normal_z = normal_cylindrical[:,2]
        binormal_R = binormal_cylindrical[:,0]
        binormal_phi = binormal_cylindrical[:,1]
        binormal_z = binormal_cylindrical[:,2]
        
        return (R0, Z0, sigma, elongation, B_axis, grad_B_axis, axis_length, iota, iotaN, G0,
                helicity, X1c_untwisted, X1s_untwisted, Y1s_untwisted, Y1c_untwisted,
                normal_R, normal_phi, normal_z, binormal_R, binormal_phi, binormal_z,
                L_grad_B, inv_L_grad_B, torsion, curvature, varphi, R0p, Z0p)
    
    @partial(jit, static_argnames=['self'])
    def residual_phi0_of_theta_varphi_func(self, phi_0, r, theta, varphi):
        # Residual = phi + nu - varphi = 0
        # Compute phi off axis
        X_at_this_theta = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi = self.Frenet_to_cylindrical_1_point(phi_0, X_at_this_theta, Y_at_this_theta)
        # phi = phi + 2 * jnp.pi * (phi < 0) - 2 * jnp.pi * (phi > 2 * jnp.pi)
        # Compute nu = nu0 + r (nu1c cos theta + nu1s sin theta)
        nu0 = self.interpolated_array_at_point(self.varphi-self.phi, phi_0)
        X1c = self.interpolated_array_at_point(self.X1c_untwisted, phi_0)
        X1s = self.interpolated_array_at_point(self.X1s_untwisted, phi_0)
        Y1c = self.interpolated_array_at_point(self.Y1c_untwisted, phi_0)
        Y1s = self.interpolated_array_at_point(self.Y1s_untwisted, phi_0)
        bR = self.interpolated_array_at_point(self.binormal_R, phi_0)
        bZ = self.interpolated_array_at_point(self.binormal_z, phi_0)
        nR = self.interpolated_array_at_point(self.normal_R, phi_0)
        nZ = self.interpolated_array_at_point(self.normal_z, phi_0)
        R0 = self.interpolated_array_at_point(self.R0, phi_0)
        R0p = self.interpolated_array_at_point(self.R0p, phi_0)
        Z0p = self.interpolated_array_at_point(self.Z0p, phi_0)
        nu1c = X1c * (bR * Z0p - bZ * R0p)/R0 + Y1c * (nZ * R0p - nR * Z0p)/R0
        nu1s = X1s * (bR * Z0p - bZ * R0p)/R0 + Y1s * (nZ * R0p - nR * Z0p)/R0
        nu = nu0 + r * (nu1c * jnp.cos(theta) + nu1s * jnp.sin(theta))
        # Return residual
        return phi + nu - varphi
    
    @partial(jit, static_argnames=['self'])
    def phi_of_theta_varphi(self, r, theta, varphi):
        residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta, r=r, varphi=varphi)
        phi_on_axis = lax.custom_root(residual, varphi, newton, lambda g, y: y / g(1.0))
        X_at_this_theta = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi_off_axis = self.Frenet_to_cylindrical_1_point(phi_on_axis, X_at_this_theta, Y_at_this_theta)
        return phi_off_axis# + 2 * jnp.pi * (phi_off_axis < 0) - 2 * jnp.pi * (phi_off_axis > 2 * jnp.pi)
        
    @partial(jit, static_argnames=['self'])
    def interpolated_array_at_point(self,array,point):
        sp=jnp.interp(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), period=2*jnp.pi/self.nfp)[0]
        ## Using interpax would make the interpolation slightly more accurate, but it is too slow at the moment
        # sp=interpax.interp1d(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), method="cubic", period=2*jnp.pi/self.nfp)[0]
        return sp
        
    @partial(jit, static_argnames=['self'])
    def Frenet_to_cylindrical_residual_func(self,phi0, phi_target, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
        normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        Frenet_to_cylindrical_residual = jnp.arctan2(total_y, total_x) - phi_target
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual > jnp.pi, Frenet_to_cylindrical_residual - 2 * jnp.pi, Frenet_to_cylindrical_residual)
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual <-jnp.pi, Frenet_to_cylindrical_residual + 2 * jnp.pi, Frenet_to_cylindrical_residual)
        return Frenet_to_cylindrical_residual

    @partial(jit, static_argnames=['self'])
    def Frenet_to_cylindrical_1_point(self, phi0, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        z0_at_phi0   = self.interpolated_array_at_point(self.Z0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        normal_z     = self.interpolated_array_at_point(self.normal_z,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        binormal_z   = self.interpolated_array_at_point(self.binormal_z,phi0)
        normal_x   = normal_R   * cosphi0 - normal_phi * sinphi0
        normal_y   = normal_R   * sinphi0 + normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z
        total_R = jnp.sqrt(total_x * total_x + total_y * total_y)
        total_phi=jnp.arctan2(total_y, total_x)
        return total_R, total_z, total_phi
    
    @partial(jit, static_argnames=['self','ntheta'])
    def Frenet_to_cylindrical(self, r, ntheta=20, phi_is_varphi=False):
        nphi_conversion = self.nphi
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / self.nfp, nphi_conversion, endpoint=False)

        def compute_for_theta(theta_j):
            costheta = jnp.cos(theta_j)
            sintheta = jnp.sin(theta_j)
            X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
            Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)

            def compute_for_phi(phi_target):
                
                def residual(z):
                    return jax.lax.cond(
                        phi_is_varphi,
                        # Branch A: solve for phi0 so that phi+nu-varphi = 0
                        lambda _: self.residual_phi0_of_theta_varphi_func(
                            z, r=r, theta=theta_j, varphi=phi_target
                        ),
                        # Branch B: solve for phi so Frenet_to_cylindrical_residual_func = 0
                        lambda _: self.Frenet_to_cylindrical_residual_func(
                            z, phi_target=phi_target,
                            X_at_this_theta=X_at_this_theta,
                            Y_at_this_theta=Y_at_this_theta
                        ),
                        operand=None
                    )
                # residual = partial(self.Frenet_to_cylindrical_residual_func, phi_target=phi_target,
                #                 X_at_this_theta=X_at_this_theta, Y_at_this_theta=Y_at_this_theta)
                # residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta_j, r=r, varphi=phi_target)
                
                phi0_solution = lax.custom_root(residual, phi_target, newton, lambda g, y: y / g(1.0))
                
                final_R, final_Z, _ = self.Frenet_to_cylindrical_1_point(phi0_solution, X_at_this_theta, Y_at_this_theta)
                return final_R, final_Z, phi0_solution

            return vmap(compute_for_phi)(phi_conversion)

        R_2D, Z_2D, phi0_2D = vmap(compute_for_theta)(theta)
        return R_2D, Z_2D, phi0_2D


    @partial(jit, static_argnames=['self','mpol', 'ntor'])
    def to_Fourier(self, R_2D, Z_2D, nfp, mpol, ntor):
        ntheta, nphi_conversion = R_2D.shape
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / nfp, nphi_conversion, endpoint=False)
        
        phi2d, theta2d = jnp.meshgrid(phi_conversion, theta, indexing='xy')
        factor = 2 / (ntheta * nphi_conversion)

        def compute_RBC_ZBS(m, n):
            angle = m * theta2d - n * nfp * phi2d
            sinangle, cosangle = jnp.sin(angle), jnp.cos(angle)

            # Conditional scaling of factor2
            factor2 = jax.lax.cond(
                (ntheta % 2 == 0) & (m == (ntheta / 2)),
                lambda _: factor / 2, lambda _: factor,
                operand=None)

            factor2 = jax.lax.cond(
                (nphi_conversion % 2 == 0) & (abs(n) == (nphi_conversion / 2)),
                lambda _: factor2 / 2, lambda _: factor2,
                operand=None)

            return jnp.sum(R_2D * cosangle * factor2), jnp.sum(Z_2D * sinangle * factor2)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.concatenate([jnp.array([1]), jnp.arange(-ntor, ntor + 1)]) if mpol == 0 else jnp.arange(-ntor, ntor + 1)
        RBC, ZBS = vmap(lambda n: vmap(lambda m: compute_RBC_ZBS(m, n))(m_vals))(n_vals)

        RBC = RBC.at[ntor, 0].set(jnp.sum(R_2D) / (ntheta * nphi_conversion))
        ZBS = ZBS.at[:ntor, 0].set(0)
        RBC = RBC.at[:ntor, 0].set(0)
        return RBC, ZBS

    @partial(jit, static_argnames=['self','ntheta_fourier', 'mpol', 'ntor', 'ntheta', 'nphi', 'phi_is_varphi'])
    def get_boundary(self, r=0.1, ntheta=30, nphi=120, ntheta_fourier=20, mpol=5, ntor=5, phi_is_varphi=False, phi_offset=0.0):
        R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier, phi_is_varphi=phi_is_varphi)
        RBC, ZBS = self.to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor)

        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        
        # phi1D = jax.lax.cond(
        #     phi_is_varphi,
        #     lambda _: jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False),
        #     lambda _: jnp.linspace(0, 2 * jnp.pi, nphi),
        #     operand=None
        # )
        # phi1D += phi_offset
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi) + phi_offset
        
        phi2D_original, theta2D = jnp.meshgrid(phi1D, theta1D, indexing='ij')
        
        phi2D = jax.lax.cond(
            phi_is_varphi,
            lambda _: vmap(lambda theta_row, varphi_row: vmap(lambda theta, varphi: self.phi_of_theta_varphi(r, theta, varphi))(theta_row, varphi_row))(theta2D, phi2D_original),
            lambda _: phi2D_original,
            operand=None
        )
        
        def compute_RZ(m, n):
            angle = m * theta2D - n * self.nfp * phi2D_original
            return RBC[n + ntor, m] * jnp.cos(angle), ZBS[n + ntor, m] * jnp.sin(angle)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.arange(-ntor, ntor + 1)

        R_2Dnew, Z_2Dnew = vmap(lambda m: vmap(lambda n: compute_RZ(m, n))(n_vals))(m_vals)
        R_2Dnew, Z_2Dnew = R_2Dnew.sum(axis=(0, 1)), Z_2Dnew.sum(axis=(0, 1))

        x_2D_plot = R_2Dnew.T * jnp.cos(phi2D.T)
        y_2D_plot = R_2Dnew.T * jnp.sin(phi2D.T)
        z_2D_plot = Z_2Dnew.T
        return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew.T    
    

    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        r, theta_B, phi = points
        theta=theta_B-(self.helicity * self.nfp * phi)
        X = r * (self.X1c_untwisted * jnp.cos(theta)+ self.X1s_untwisted * jnp.sin(theta))
        Y = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        R0_at_phi   = self.interpolated_array_at_point(self.R0,phi)
        z0_at_phi   = self.interpolated_array_at_point(self.Z0,phi)
        X_at_phi    = self.interpolated_array_at_point(X,phi)
        Y_at_phi    = self.interpolated_array_at_point(Y,phi)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi)
        normal_z     = self.interpolated_array_at_point(self.normal_z,phi)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi)
        binormal_z   = self.interpolated_array_at_point(self.binormal_z,phi)
        normal_x   = normal_R   * cosphi - normal_phi * sinphi
        normal_y   = normal_R   * sinphi + normal_phi * cosphi
        binormal_x = binormal_R * cosphi - binormal_phi * sinphi
        binormal_y = binormal_R * sinphi + binormal_phi * cosphi
        total_x = R0_at_phi * cosphi + X_at_phi * normal_x + Y_at_phi * binormal_x
        total_y = R0_at_phi * sinphi + X_at_phi * normal_y + Y_at_phi * binormal_y
        Z = z0_at_phi           + X_at_phi * normal_z + Y_at_phi * binormal_z
        R = jnp.sqrt(total_x * total_x + total_y * total_y)
        phi_cyl=jnp.arctan2(total_y, total_x)
        X_new = R * jnp.cos(phi_cyl)
        Y_new = R * jnp.sin(phi_cyl)
        return jnp.array([X_new,Y_new, Z])
    

    def plot(self, r=0.1, ntheta=40, nphi=120, ntheta_fourier=20, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        kwargs.setdefault('alpha', 1)
        import matplotlib.pyplot as plt 
        from matplotlib import cm
        import matplotlib.colors as clr
        from matplotlib.colors import LightSource
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')   
        x_2D_plot, y_2D_plot, z_2D_plot, _ = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2D, theta2D = jnp.meshgrid(phi1D, theta1D)
        import numpy as np
        Bmag = np.array(self.B_mag(r, theta2D, phi2D))
        norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
        cmap = cm.viridis
        ls = LightSource(azdeg=0, altdeg=10)
        cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=cmap_plot,
                        rstride=1, cstride=1, antialiased=True,
                        linewidth=0, **kwargs)
        if ax is None or ax.name != "3d":
            ax.dist = 7
            ax.elev = 5
            ax.azim = 45
            cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            cbar = plt.colorbar(m, cax=cbar_ax)
            cbar.ax.set_title(r'$|B| [T]$')
            ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()










###########QI_general frame (Alex's version)###############

class near_axis_general_frame():
    def __init__(self, rc=jnp.array([1, 0.1]), zs=jnp.array([0, 0.1]), X1c = None, Y1c = None, B0 = None, I2 = 0, B2s=None, B2c=None, *, p2=0.,
                 nfp = 1, sG = 1, spsi = 1, nphi = 61, sigma0 = 0., order="r1", frame="centroid"):
        assert nphi % 2 == 1, 'nphi must be odd'
        self.rc = jnp.array(rc)
        print(self.rc.shape)
        print(rc)
        self.zs = jnp.array(zs)
        if X1c is None:
            X1c = jnp.ones(nphi)
        if Y1c is None:
            Y1c = jnp.ones(nphi)
        if B0 is None:
            B0 = jnp.ones(nphi)
        if B2s is None:
            B2s = jnp.zeros(nphi)
        if B2c is None:
            B2c = jnp.zeros(nphi)
        self.X1c = jnp.array(X1c)
        self.Y1c = jnp.array(Y1c)
        self.B0 = jnp.array(B0)
        self.B2s = jnp.array(B2s)
        self.B2c = jnp.array(B2c)
        self.nphi = nphi
        self.sigma0 = sigma0
        self.I2 = I2
        self.spsi = spsi
        self.sG = sG
        self.nfp = nfp
        self.order = order # not used
        self.p2 = p2 # not used
        
        self._dofs = jnp.concatenate((jnp.ravel(self.rc), jnp.ravel(self.zs), jnp.ravel(self.X1c), jnp.ravel(self.Y1c), jnp.ravel(self.B0)), axis=0)
        
        self.phi = jnp.linspace(0, 2 * jnp.pi / self.nfp, self.nphi, endpoint=False)
        self.nfourier = max(len(self.rc), len(self.zs))
        
        parameters = self.calculate(self.rc, self.zs)
        (self.R0, self.Z0, self.sigma, self.elongation, self.B_axis, self.grad_B_axis, self.axis_length, self.X1s, self.X1c, self.Y1s, self.Y1c, self.B1s, self.B1c, self.axis_length, self.iota, self.iotaN, self.G0,
                self.frame_p_R, self.frame_p_phi, self.frame_p_z, self.frame_q_R, self.frame_q_phi, self.frame_q_z,
                self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature, self.k1, self.k2, self.k3, self.varphi, self.R0p, self.Z0p) = parameters


    @partial(jit, static_argnames=['self'])
    def B_covariant(self, points):
        #print("Bco")
        r, theta, phi = points
        phiidx = (phi / (2 * jnp.pi / self.nfp)  * self.nphi).astype(jnp.int32) # finds index closest to phi value input. would be better to spline B0, B1c, B1s
        Br = 0
        Btheta = r*r*self.I2
        Bphi = jnp.take(self.G0, phiidx, axis=0)
        return jnp.array([Br, Btheta, Bphi])
    
    @partial(jit, static_argnames=['self'])
    def B_contravariant(self, points):
        r, theta, phi = points
        jac = self.sqrtg(points)
        phi = jnp.squeeze(phi)
        phiidx = (phi / (2 * jnp.pi / self.nfp)  * self.nphi).astype(jnp.int32) # finds index closest to phi value input. would be better to spline B0, B1c, B1s
        B0_at_phi = jnp.take(self.B0, phiidx, axis=0)        
        Bphi = r*B0_at_phi/jac
        return jnp.array([0, self.iota * Bphi, Bphi])
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        print("AbsB")
        r, theta, phi = points

        phi = jnp.squeeze(phi)
        phiidx = (phi / (2 * jnp.pi / self.nfp)  * self.nphi).astype(jnp.int32) # finds index closest to phi value input. would be better to spline B0, B1c, B1s
        B1c_at_phi = jnp.take(self.B1c, phiidx, axis=0)
        B1s_at_phi = jnp.take(self.B1s, phiidx, axis=0)
        B0_at_phi = jnp.take(self.B0, phiidx, axis=0)

        return B0_at_phi + (r*(B1c_at_phi*jnp.cos(theta) + B1s_at_phi*jnp.sin(theta)))
    
    @partial(jit, static_argnames=['self'])
    def jacobian(self, points):
        print("Jac")
        r, theta, phi = points

        phi = jnp.squeeze(phi)
        phiidx = (phi / (2 * jnp.pi / self.nfp)  * self.nphi).astype(jnp.int32) # finds index closest to phi value input. would be better to spline B0, B1c, B1s
        B0_at_phi = jnp.take(self.B0, phiidx, axis=0)
        G0_at_phi = jnp.take(self.G0, phiidx, axis=0)

        AbsB = self.AbsB(points)
        return r*B0_at_phi*(G0_at_phi+self.iota*self.I2*r*r)/(AbsB*AbsB)
    

    @partial(jit, static_argnames=['self'])
    def sqrtg(self, points):
        #print("Jac")
        r, theta, phi = points

        phi = jnp.squeeze(phi)
        phiidx = (phi / (2 * jnp.pi / self.nfp)  * self.nphi).astype(jnp.int32) # finds index closest to phi value input. would be better to spline B0, B1c, B1s
        B0_at_phi = jnp.take(self.B0, phiidx, axis=0)
        G0_at_phi = jnp.take(self.G0, phiidx, axis=0)

        AbsB = self.AbsB(points)
        return r*B0_at_phi*(G0_at_phi+self.iota*self.I2*r*r)/(AbsB*AbsB)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)
    
    
    @partial(jit, static_argnames=['self'])
    def dAbsB_by_dX(self, points):
        return grad(self.AbsB)(points)
    
    @partial(jit, static_argnames=['self'])
    def grad_B_covariant(self, points):
        return jacfwd(self.B_covariant)(points)    
 
    @partial(jit, static_argnames=['self'])
    def curl_B(self, points):
        grad_B_cov=self.grad_B_covariant(points)
        return jnp.array([grad_B_cov[2][1] -grad_B_cov[1][2],
                          grad_B_cov[0][2] -grad_B_cov[2][0],
                          grad_B_cov[1][0] -grad_B_cov[0][1]])/self.sqrtg(points)
    
    @partial(jit, static_argnames=['self'])
    def curl_b(self, points):
        return self.curl_B(points)/self.AbsB(points)+jnp.cross(self.B_covariant(points),jnp.array(self.dAbsB_by_dX(points)))/self.AbsB(points)**2/self.sqrtg(points)

    @partial(jit, static_argnames=['self'])
    def kappa(self, points):
        return -jnp.cross(self.B_contravariant(points),self.curl_b(points))*self.sqrtg(points)/self.AbsB(points)

    @partial(jit, static_argnames=['self'])
    def B_mag(self, r, theta, phi):
        return self.B0*(1 + r * self.etabar * jnp.cos(theta - (self.iota - self.iotaN) * phi))

        
    @partial(jit, static_argnames=['self'])
    def calculate(self, rc, zs):
        phi = self.phi
        nphi = self.nphi
        nfp = self.nfp
        nfourier = self.nfourier
        spsi = self.spsi
        sG = self.sG
        B0 = self.B0
        sigma0 = self.sigma0
        X1c = self.X1c
        Y1c = self.Y1c
        I2 = self.I2
        d_phi = phi[1] - phi[0]
        
        n_values = jnp.arange(nfourier) * nfp

        @jit
        def compute_terms(jn):
            n = n_values[jn]
            sinangle = jnp.sin(n * phi)
            cosangle = jnp.cos(n * phi)
            return jnp.array([rc[jn] * cosangle, zs[jn] * sinangle,
                rc[jn] * (-n * sinangle), zs[jn] * (n * cosangle),
                rc[jn] * (-n * n * cosangle), zs[jn] * (-n * n * sinangle),
                rc[jn] * (n * n * n * sinangle), zs[jn] * (-n * n * n * cosangle)])

        @jit
        def spectral_diff_matrix_jax():
            n=nphi
            xmin=0
            xmax=2 * jnp.pi / nfp
            h = 2 * jnp.pi / n
            kk = jnp.arange(1, n)
            n_half = n // 2
            topc = 1 / jnp.sin(jnp.arange(1, n_half + 1) * h / 2)
            temp = jnp.concatenate((topc, jnp.flip(topc[:n_half])))
            col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
            row1 = -col1
            vals = jnp.concatenate((row1[-1:0:-1], col1))
            a, b = jnp.ogrid[0:len(col1), len(row1)-1:-1:-1]
            return 2 * jnp.pi / (xmax - xmin) * vals[a + b]

        @jit
        def determine_helicity(normal_cylindrical):
            x_positive = normal_cylindrical[:, 0] >= 0
            z_positive = normal_cylindrical[:, 2] >= 0
            quadrant = 1 * x_positive * z_positive + 2 * (~x_positive) * z_positive \
                    + 3 * (~x_positive) * (~z_positive) + 4 * x_positive * (~z_positive)
            quadrant = jnp.append(quadrant, quadrant[0])
            delta_quadrant = quadrant[1:] - quadrant[:-1]
            increment = jnp.sum((quadrant[:-1] == 4) & (quadrant[1:] == 1))
            decrement = jnp.sum((quadrant[:-1] == 1) & (quadrant[1:] == 4))
            return (jnp.sum(delta_quadrant) + increment - decrement) * spsi * sG

        summed_values = jnp.sum(jax.vmap(compute_terms)(jnp.arange(nfourier)), axis=0)

        R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp = summed_values
        d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        d_l_d_varphi = abs_G0_over_B0
        G0 = sG * abs_G0_over_B0 * B0

        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        r = jnp.array([R0*cosphi, R0*sinphi, Z0]).transpose()
        rp = jnp.array([R0p*cosphi - R0*sinphi, R0p*sinphi + R0*cosphi, Z0p]).transpose()
        rpp = jnp.array([(R0pp - R0)*cosphi - 2*R0p*sinphi, (R0pp - R0)*sinphi + 2*R0p*cosphi, Z0pp]).transpose()
        rppp = jnp.array([(R0ppp - 3*R0p)*cosphi - (3*R0pp - R0)*sinphi, (R0ppp - 3*R0p)*sinphi + (3*R0pp - R0)*cosphi, Z0ppp]).transpose()


        dldp = jnp.linalg.norm(rp, axis=1)

        axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp

        t = jnp.empty(rp.shape)
        n = jnp.empty(rp.shape)
        for idx, l in enumerate (dldp):
            t = t.at[idx].set(jnp.divide(rp[idx], l))
            n = n.at[idx].set(jnp.divide(jnp.cross(rp[idx,:], jnp.cross(rpp[idx,:], rp[idx])),(l * jnp.linalg.norm(jnp.cross(rpp[idx], rp[idx])))))
        b = jnp.cross(t, n, axis=1)

        curvature = jnp.divide(jnp.linalg.norm(jnp.cross(rp, rpp, axis=1),axis=1),(dldp*dldp*dldp))
        torsion = jnp.empty(curvature.shape)
        for idx, _ in enumerate(rp):
            torsion = torsion.at[idx].set(jnp.dot(rp[idx], jnp.cross(rpp[idx], rppp[idx])) / jnp.linalg.norm(jnp.cross(rp[idx], rpp[idx]))**2)

        phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)

        dtdp = jnp.empty(t.shape)
        dldp = jnp.linalg.norm(rp, axis=1)
        for idx, m in enumerate(n):
            dtdp = dtdp.at[idx].set(dldp[idx] * curvature[idx] * n[idx])

        p = r
        dpdphi = rp.copy()
        for idx, tan in enumerate(t):
            #P1
            temp = jnp.dot(p[idx], tan)
            dtempdphi = jnp.dot(p[idx], dtdp[idx]) + jnp.dot(dpdphi[idx], tan)

            p = p.at[idx].set(p[idx] - temp * tan)
            dpdphi = dpdphi.at[idx].set(dpdphi[idx] - dtempdphi * tan - temp * dtdp[idx])

            #Pf
            temp2 = jnp.linalg.norm(p[idx])
            p = p.at[idx].set((1 / temp2) * p[idx])
            dpdphi =  dpdphi.at[idx].set((1 / temp2) * dpdphi[idx] - 1/(temp2) * p[idx] * jnp.dot(dpdphi[idx], p[idx]))
        q = jnp.cross(t, p, axis=1)
        dqdphi = (jnp.cross(t, dpdphi, axis=1) + jnp.cross(dtdp, p, axis=1)) 


        k1 = jnp.empty(n[:,0].shape)
        k2 = jnp.empty(n[:,0].shape)
        for idx, _ in enumerate(n):
            cos_a = jnp.dot(p[idx], n[idx])
            minus_sin_a = jnp.dot(q[idx], n[idx])
            k1 = k1.at[idx].set(curvature[idx] * cos_a)
            k2 = k2.at[idx].set(curvature[idx] * minus_sin_a)
        
        k3_0 = jnp.empty(p[:,0].shape)
        k3_1 = jnp.empty(p[:,0].shape)
        k3 = jnp.empty(p[:,0].shape)
        for idx, _ in enumerate(p):
            k3_0 = k3_0.at[idx].set(jnp.dot(dpdphi[idx], q[idx]) / dldp[idx])
            k3_1 = k3_1.at[idx].set(-jnp.dot(dqdphi[idx], p[idx]) / dldp[idx])
            k3 = k3.at[idx].set((k3_0[idx]+k3_1[idx]) / 2)

        d_d_phi = spectral_diff_matrix_jax()
        d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
        d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
        helicity = 0


        tangent_cartesian = t
        frame_p_cartesian = p
        frame_q_cartesian = q

        tan_R = jnp.zeros(nphi)
        tan_phi = jnp.zeros(nphi)
        tan_z = jnp.zeros(nphi)

        p_R = jnp.zeros(nphi)
        p_phi = jnp.zeros(nphi)
        p_z = jnp.zeros(nphi)

        q_R = jnp.zeros(nphi)
        q_phi = jnp.zeros(nphi)
        q_z = jnp.zeros(nphi)

        for phiIndex in range(nphi):
            phi0 = phi[phiIndex]
            sinphi0 = jnp.sin(phi0)
            cosphi0 = jnp.cos(phi0)
            
            p_R = p_R.at[phiIndex].set(p[phiIndex,0] * cosphi0 + p[phiIndex,1] * sinphi0)
            p_phi = p_phi.at[phiIndex].set(-p[phiIndex,0] * sinphi0 + p[phiIndex,1] * cosphi0)
            p_z = p_z.at[phiIndex].set(p[phiIndex,2])

            q_R = q_R.at[phiIndex].set(q[phiIndex,0] * cosphi0 + q[phiIndex,1] * sinphi0)
            q_phi = q_phi.at[phiIndex].set(-q[phiIndex,0] * sinphi0 + q[phiIndex,1] * cosphi0)
            q_z = q_z.at[phiIndex].set(q[phiIndex,2])

            tan_R = tan_R.at[phiIndex].set(t[phiIndex,0] * cosphi0 + t[phiIndex,1] * sinphi0)
            tan_phi = tan_phi.at[phiIndex].set(-t[phiIndex,0] * sinphi0 + t[phiIndex,1] * cosphi0)
            tan_z = tan_z.at[phiIndex].set(t[phiIndex,2])


        tangent_cylindrical = jnp.array([tan_R, tan_phi, tan_z]).T
        frame_p_cylindrical =  jnp.array([p_R, p_phi, p_z]).T
        frame_q_cylindrical = jnp.array([q_R, q_phi, q_z]).T
        Bbar = spsi * jnp.mean(B0)
        abs_G0_over_B0 = abs_G0_over_B0
        varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

        @jit
        def replace_first_element(x, new_value):
            return jnp.concatenate([jnp.array([new_value]), x[1:]])

        @jit
        def sigma_equation_residual(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            
            L2 = jnp.power(X1c,2) + jnp.power(Y1c,2)
            beta = -1 * Bbar / (B0 * L2)

            X1cP = jnp.matmul(d_d_varphi, X1c)
            Y1cP = jnp.matmul(d_d_varphi, Y1c)
            B0P = jnp.matmul(d_d_varphi, B0)

            r = jnp.matmul(d_d_varphi, sigma) \
                - sigma * ((2 * (X1c * X1cP + Y1c * Y1cP)) / L2 + B0P / B0) \
                + beta * iota * (sigma * sigma + 1 + 1 / (beta * beta))  \
                + 2 * self.sG * (Y1c * X1cP - X1c * Y1cP) / L2 \
                + 2 * G0 / B0 * (self.I2 / Bbar - k3)
            return r

        @jit
        def sigma_equation_jacobian(x):
            iota = x[0]
            sigma = replace_first_element(x, sigma0)
            
            X1cP = jnp.matmul(d_d_varphi, X1c)
            Y1cP = jnp.matmul(d_d_varphi, Y1c)
            B0P = jnp.matmul(d_d_varphi, B0)

            L2 = jnp.power(X1c,2) + jnp.power(Y1c,2)
            beta = -1 * Bbar / (B0 * L2)

            jac = jnp.copy(d_d_varphi)
            #dr/d(sigma)
            for j in range(self.nphi):
                jac = jac.at[j, j].set(jac[j,j] + iota * 2 * beta[j] * sigma[j] - ((2 * (X1c[j] * X1cP[j] + Y1c[j] * Y1cP[j])) / L2[j] + B0P[j] / self.B0[j]))
            #dr/d(iota)
            jac = jac.at[:, 0].set(beta * (sigma * sigma + 1 + 1 / (beta * beta)))

            return jac

        @partial(jit, static_argnums=(1,))
        def newton(x0, niter=5):
            def body_fun(i, x):
                residual = sigma_equation_residual(x)
                jacobian = sigma_equation_jacobian(x)
                step = jax.scipy.linalg.solve(jacobian, -residual)
                return x + step
            x = jax.lax.fori_loop(0, niter, body_fun, x0)
            return x

        x0 = jnp.full(nphi, sigma0)
        x0 = replace_first_element(x0, 0.)
        sigma = newton(x0)
        iota = sigma[0]
        iotaN = iota + helicity * nfp
        sigma = replace_first_element(sigma, sigma0)

        L2 = jnp.power(X1c,2) + jnp.power(self.Y1c,2)
        beta = -1 * Bbar / (B0 * L2)

        Y1s = -beta * (X1c + Y1c * sigma)
        X1s = -beta * (jnp.multiply(-1, Y1c) + X1c * sigma)

        B1c = (X1c * k1 + Y1c * k2) * B0
        B1s = (X1s * k1 + Y1s * k2) * B0
        
        p_el = + X1c * X1c + Y1s * Y1s + Y1c * Y1c + X1s * X1s
        q_el = + X1s * Y1c - X1c * Y1s
        elongation = (p_el + jnp.sqrt(p_el * p_el - 4 * q_el * q_el)) / (2 * jnp.abs(q_el))
        
        B_axis_cylindrical = sG * B0 * t.T
        B_x = jnp.cos(phi) * B_axis_cylindrical[0] - jnp.sin(phi) * B_axis_cylindrical[1]
        B_y = jnp.sin(phi) * B_axis_cylindrical[0] + jnp.cos(phi) * B_axis_cylindrical[1]
        B_z = B_axis_cylindrical[2]
        B_axis = jnp.array([B_x, B_y, B_z])

        d_X1c_d_varphi = jnp.matmul(d_d_varphi, self.X1c)
        d_X1s_d_varphi = jnp.matmul(d_d_varphi, X1s)
        d_Y1s_d_varphi = jnp.matmul(d_d_varphi, Y1s)
        d_Y1c_d_varphi = jnp.matmul(d_d_varphi, self.Y1c)
        
        frame_p_cylindrical = p
        frame_q_cylindrical = q

        factor = B0 * B0 / (Bbar * d_l_d_varphi)

        tp = sG * B0 * k1
        pt = tp
        tq = sG * B0 * k2
        qt = tq
        
        
        pp = factor * (d_X1c_d_varphi * Y1s - d_X1s_d_varphi * Y1c + iotaN * (jnp.multiply(X1s,Y1s) + jnp.multiply(X1c,Y1c)))
        qq = factor * (X1c * d_Y1s_d_varphi - X1s * d_Y1c_d_varphi - iotaN * (jnp.multiply(X1s,Y1s) + jnp.multiply(X1c, Y1c)))
        qp = factor * (X1c * d_X1s_d_varphi - X1s * d_X1c_d_varphi - sG * Bbar * k3 * d_l_d_varphi / B0 - iotaN * (jnp.multiply(X1s, X1s) + jnp.multiply(X1c,X1c)))
        pq = factor * (Y1s * d_Y1c_d_varphi - Y1c * d_Y1s_d_varphi + sG * Bbar * k3 * d_l_d_varphi / B0 + iotaN * (jnp.multiply(Y1s, Y1s) + jnp.multiply(Y1c, Y1c)))
        tt = sG * jnp.matmul(d_d_varphi, B0) / d_l_d_varphi

        t = t.transpose()
        p = frame_p_cylindrical.transpose()
        q = frame_q_cylindrical.transpose()
        nablaB = jnp.array([[
                                pp * p[i] * p[j] \
                                + qp * q[i] * p[j] + pq * p[i] * q[j] \
                                + qq * q[i] * q[j] \
                                + tp * t[i] * p[j] + pt * p[i] * t[j] \
                                + tq * t[i] * q[j] + qt * q[i] + t[j] \
                                + tt * t[i] * t[j]
                            for i in range(3)] for j in range(3)])
        
        grad_B_colon_grad_B = tp * tp + pt * pt \
            + tq * tq + qt * qt \
            + qq * qq + pp * pp \
            + pq * pq + qp * qp \
            + tt * tt
        
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        grad_B_axis = jnp.array([
            [cosphi**2*nablaB[0, 0] - cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + 
            sinphi**2*nablaB[1, 1], cosphi**2*nablaB[0, 1] - sinphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), cosphi*nablaB[0, 2] - 
            sinphi*nablaB[1, 2]], [-(sinphi**2*nablaB[0, 1]) + cosphi**2*nablaB[1, 0] + 
            cosphi*sinphi*(nablaB[0, 0] - nablaB[1, 1]), sinphi**2*nablaB[0, 0] + 
            cosphi*sinphi*(nablaB[0, 1] + nablaB[1, 0]) + cosphi**2*nablaB[1, 1], 
            sinphi*nablaB[0, 2] + cosphi*nablaB[1, 2]], 
            [cosphi*nablaB[2, 0] - sinphi*nablaB[2, 1], sinphi*nablaB[2, 0] + cosphi*nablaB[2, 1], 
            nablaB[2, 2]]
                ])
        
        L_grad_B = self.B0 * jnp.sqrt(2 / grad_B_colon_grad_B)
        inv_L_grad_B = 1.0 / L_grad_B
        
        
        frame_p_R = frame_p_cylindrical[:,0]
        frame_p_phi = frame_p_cylindrical[:,1]
        frame_p_z = frame_p_cylindrical[:,2]
        frame_q_R = frame_q_cylindrical[:,0]
        frame_q_phi = frame_q_cylindrical[:,1]
        frame_q_z = frame_q_cylindrical[:,2]
        
        return (R0, Z0, sigma, elongation, B_axis, grad_B_axis, axis_length, X1s, X1c, Y1s, Y1c, B1s, B1c, axis_length, iota, iotaN, G0,
                frame_p_R, frame_p_phi, frame_p_z, frame_q_R, frame_q_phi, frame_q_z,
                L_grad_B, inv_L_grad_B, torsion, curvature, k1, k2, k3, varphi, R0p, Z0p)
    
    @partial(jit, static_argnames=['self'])
    def residual_phi0_of_theta_varphi_func(self, phi_0, r, theta, varphi):
        # Residual = phi + nu - varphi = 0
        # Compute phi off axis
        X_at_this_theta = r * (self.X1c * jnp.cos(theta) + self.X1s * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c * jnp.cos(theta) + self.Y1s * jnp.sin(theta))
        _, _, phi = self.Frenet_to_cylindrical_1_point(phi_0, X_at_this_theta, Y_at_this_theta)
        # phi = phi + 2 * jnp.pi * (phi < 0) - 2 * jnp.pi * (phi > 2 * jnp.pi)
        # Compute nu = nu0 + r (nu1c cos theta + nu1s sin theta)
        nu0 = self.interpolated_array_at_point(self.varphi-self.phi, phi_0)
        X1c = self.interpolated_array_at_point(self.X1c, phi_0)
        X1s = self.interpolated_array_at_point(self.X1s, phi_0)
        Y1c = self.interpolated_array_at_point(self.Y1c, phi_0)
        Y1s = self.interpolated_array_at_point(self.Y1s, phi_0)
        bR = self.interpolated_array_at_point(self.binormal_R, phi_0)
        bZ = self.interpolated_array_at_point(self.binormal_z, phi_0)
        nR = self.interpolated_array_at_point(self.normal_R, phi_0)
        nZ = self.interpolated_array_at_point(self.normal_z, phi_0)
        R0 = self.interpolated_array_at_point(self.R0, phi_0)
        R0p = self.interpolated_array_at_point(self.R0p, phi_0)
        Z0p = self.interpolated_array_at_point(self.Z0p, phi_0)
        nu1c = X1c * (bR * Z0p - bZ * R0p)/R0 + Y1c * (nZ * R0p - nR * Z0p)/R0
        nu1s = X1s * (bR * Z0p - bZ * R0p)/R0 + Y1s * (nZ * R0p - nR * Z0p)/R0
        nu = nu0 + r * (nu1c * jnp.cos(theta) + nu1s * jnp.sin(theta))
        # Return residual
        return phi + nu - varphi
    
    @partial(jit, static_argnames=['self'])
    def phi_of_theta_varphi(self, r, theta, varphi):
        residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta, r=r, varphi=varphi)
        phi_on_axis = lax.custom_root(residual, varphi, newton, lambda g, y: y / g(1.0))
        X_at_this_theta = r * (self.X1c * jnp.cos(theta) + self.X1s * jnp.sin(theta))
        Y_at_this_theta = r * (self.Y1c * jnp.cos(theta) + self.Y1s * jnp.sin(theta))
        _, _, phi_off_axis = self.Frenet_to_cylindrical_1_point(phi_on_axis, X_at_this_theta, Y_at_this_theta)
        return phi_off_axis# + 2 * jnp.pi * (phi_off_axis < 0) - 2 * jnp.pi * (phi_off_axis > 2 * jnp.pi)
        
    @partial(jit, static_argnames=['self'])
    def interpolated_array_at_point(self,array,point):
        sp=jnp.interp(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), period=2*jnp.pi/self.nfp)[0]
        ## Using interpax would make the interpolation slightly more accurate, but it is too slow at the moment
        # sp=interpax.interp1d(jnp.array([point]), jnp.append(self.phi,2*jnp.pi/self.nfp), jnp.append(array,array[0]), method="cubic", period=2*jnp.pi/self.nfp)[0]
        return sp
        
    @partial(jit, static_argnames=['self'])
    def Frenet_to_cylindrical_residual_func(self,phi0, phi_target, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
        normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        Frenet_to_cylindrical_residual = jnp.arctan2(total_y, total_x) - phi_target
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual > jnp.pi, Frenet_to_cylindrical_residual - 2 * jnp.pi, Frenet_to_cylindrical_residual)
        Frenet_to_cylindrical_residual = jnp.where(Frenet_to_cylindrical_residual <-jnp.pi, Frenet_to_cylindrical_residual + 2 * jnp.pi, Frenet_to_cylindrical_residual)
        return Frenet_to_cylindrical_residual

    @partial(jit, static_argnames=['self'])
    def Frenet_to_cylindrical_1_point(self, phi0, X_at_this_theta, Y_at_this_theta):
        sinphi0 = jnp.sin(phi0)
        cosphi0 = jnp.cos(phi0)
        R0_at_phi0   = self.interpolated_array_at_point(self.R0,phi0)
        z0_at_phi0   = self.interpolated_array_at_point(self.Z0,phi0)
        X_at_phi0    = self.interpolated_array_at_point(X_at_this_theta,phi0)
        Y_at_phi0    = self.interpolated_array_at_point(Y_at_this_theta,phi0)
        normal_R     = self.interpolated_array_at_point(self.normal_R,phi0)
        normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi0)
        normal_z     = self.interpolated_array_at_point(self.normal_z,phi0)
        binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi0)
        binormal_z   = self.interpolated_array_at_point(self.binormal_z,phi0)
        normal_x   = normal_R   * cosphi0 - normal_phi * sinphi0
        normal_y   = normal_R   * sinphi0 + normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0
        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z
        total_R = jnp.sqrt(total_x * total_x + total_y * total_y)
        total_phi=jnp.arctan2(total_y, total_x)
        return total_R, total_z, total_phi
    
    @partial(jit, static_argnames=['self','ntheta'])
    def Frenet_to_cylindrical(self, r, ntheta=20, phi_is_varphi=False):
        nphi_conversion = self.nphi
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / self.nfp, nphi_conversion, endpoint=False)

        def compute_for_theta(theta_j):
            costheta = jnp.cos(theta_j)
            sintheta = jnp.sin(theta_j)
            X_at_this_theta = r * (self.X1c * costheta + self.X1s * sintheta)
            Y_at_this_theta = r * (self.Y1c * costheta + self.Y1s * sintheta)

            def compute_for_phi(phi_target):
                
                def residual(z):
                    return jax.lax.cond(
                        phi_is_varphi,
                        # Branch A: solve for phi0 so that phi+nu-varphi = 0
                        lambda _: self.residual_phi0_of_theta_varphi_func(
                            z, r=r, theta=theta_j, varphi=phi_target
                        ),
                        # Branch B: solve for phi so Frenet_to_cylindrical_residual_func = 0
                        lambda _: self.Frenet_to_cylindrical_residual_func(
                            z, phi_target=phi_target,
                            X_at_this_theta=X_at_this_theta,
                            Y_at_this_theta=Y_at_this_theta
                        ),
                        operand=None
                    )
                # residual = partial(self.Frenet_to_cylindrical_residual_func, phi_target=phi_target,
                #                 X_at_this_theta=X_at_this_theta, Y_at_this_theta=Y_at_this_theta)
                # residual = partial(self.residual_phi0_of_theta_varphi_func, theta=theta_j, r=r, varphi=phi_target)
                
                phi0_solution = lax.custom_root(residual, phi_target, newton, lambda g, y: y / g(1.0))
                
                final_R, final_Z, _ = self.Frenet_to_cylindrical_1_point(phi0_solution, X_at_this_theta, Y_at_this_theta)
                return final_R, final_Z, phi0_solution

            return vmap(compute_for_phi)(phi_conversion)

        R_2D, Z_2D, phi0_2D = vmap(compute_for_theta)(theta)
        return R_2D, Z_2D, phi0_2D


    @partial(jit, static_argnames=['self','mpol', 'ntor'])
    def to_Fourier(self, R_2D, Z_2D, nfp, mpol, ntor):
        ntheta, nphi_conversion = R_2D.shape
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / nfp, nphi_conversion, endpoint=False)
        
        phi2d, theta2d = jnp.meshgrid(phi_conversion, theta, indexing='xy')
        factor = 2 / (ntheta * nphi_conversion)

        def compute_RBC_ZBS(m, n):
            angle = m * theta2d - n * nfp * phi2d
            sinangle, cosangle = jnp.sin(angle), jnp.cos(angle)

            # Conditional scaling of factor2
            factor2 = jax.lax.cond(
                (ntheta % 2 == 0) & (m == (ntheta / 2)),
                lambda _: factor / 2, lambda _: factor,
                operand=None)

            factor2 = jax.lax.cond(
                (nphi_conversion % 2 == 0) & (abs(n) == (nphi_conversion / 2)),
                lambda _: factor2 / 2, lambda _: factor2,
                operand=None)

            return jnp.sum(R_2D * cosangle * factor2), jnp.sum(Z_2D * sinangle * factor2)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.concatenate([jnp.array([1]), jnp.arange(-ntor, ntor + 1)]) if mpol == 0 else jnp.arange(-ntor, ntor + 1)
        RBC, ZBS = vmap(lambda n: vmap(lambda m: compute_RBC_ZBS(m, n))(m_vals))(n_vals)

        RBC = RBC.at[ntor, 0].set(jnp.sum(R_2D) / (ntheta * nphi_conversion))
        ZBS = ZBS.at[:ntor, 0].set(0)
        RBC = RBC.at[:ntor, 0].set(0)
        return RBC, ZBS

    @partial(jit, static_argnames=['self','ntheta_fourier', 'mpol', 'ntor', 'ntheta', 'nphi', 'phi_is_varphi'])
    def get_boundary(self, r=0.1, ntheta=30, nphi=120, ntheta_fourier=20, mpol=5, ntor=5, phi_is_varphi=False, phi_offset=0.0):
        R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier, phi_is_varphi=phi_is_varphi)
        RBC, ZBS = self.to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor)

        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        
        # phi1D = jax.lax.cond(
        #     phi_is_varphi,
        #     lambda _: jnp.linspace(2*jnp.pi/nphi/2, 2*jnp.pi + 2*jnp.pi/nphi/2, nphi, endpoint=False),
        #     lambda _: jnp.linspace(0, 2 * jnp.pi, nphi),
        #     operand=None
        # )
        # phi1D += phi_offset
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi) + phi_offset
        
        phi2D_original, theta2D = jnp.meshgrid(phi1D, theta1D, indexing='ij')
        
        phi2D = jax.lax.cond(
            phi_is_varphi,
            lambda _: vmap(lambda theta_row, varphi_row: vmap(lambda theta, varphi: self.phi_of_theta_varphi(r, theta, varphi))(theta_row, varphi_row))(theta2D, phi2D_original),
            lambda _: phi2D_original,
            operand=None
        )
        
        def compute_RZ(m, n):
            angle = m * theta2D - n * self.nfp * phi2D_original
            return RBC[n + ntor, m] * jnp.cos(angle), ZBS[n + ntor, m] * jnp.sin(angle)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.arange(-ntor, ntor + 1)

        R_2Dnew, Z_2Dnew = vmap(lambda m: vmap(lambda n: compute_RZ(m, n))(n_vals))(m_vals)
        R_2Dnew, Z_2Dnew = R_2Dnew.sum(axis=(0, 1)), Z_2Dnew.sum(axis=(0, 1))

        x_2D_plot = R_2Dnew.T * jnp.cos(phi2D.T)
        y_2D_plot = R_2Dnew.T * jnp.sin(phi2D.T)
        z_2D_plot = Z_2Dnew.T
        return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew.T
    

    @partial(jit, static_argnames=['self'])
    def to_xyz(self, points):
        r, theta_B, phi = points
        #theta=theta_B-(self.iota-self.iotaN)*phi
        #X = r * (self.X1c * jnp.cos(theta)+ self.X1s * jnp.sin(theta))
        #Y = r * (self.Y1c * jnp.cos(theta) + self.Y1s * jnp.sin(theta))
        #sinphi = jnp.sin(phi)
        #cosphi = jnp.cos(phi)
        #R0_at_phi   = self.interpolated_array_at_point(self.R0,phi)
        #z0_at_phi   = self.interpolated_array_at_point(self.Z0,phi)
        #X_at_phi    = self.interpolated_array_at_point(X,phi)
        #Y_at_phi    = self.interpolated_array_at_point(Y,phi)
        #normal_R     = self.interpolated_array_at_point(self.normal_R,phi)
        #normal_phi   = self.interpolated_array_at_point(self.normal_phi,phi)
        #normal_z     = self.interpolated_array_at_point(self.normal_z,phi)
        #binormal_R   = self.interpolated_array_at_point(self.binormal_R,phi)
        #binormal_phi = self.interpolated_array_at_point(self.binormal_phi,phi)
        #binormal_z   = self.interpolated_array_at_point(self.binormal_z,phi)
        #normal_x   = normal_R   * cosphi - normal_phi * sinphi
        #normal_y   = normal_R   * sinphi + normal_phi * cosphi
        #binormal_x = binormal_R * cosphi - binormal_phi * sinphi
        #binormal_y = binormal_R * sinphi + binormal_phi * cosphi
        #total_x = R0_at_phi * cosphi + X_at_phi * normal_x + Y_at_phi * binormal_x
        #total_y = R0_at_phi * sinphi + X_at_phi * normal_y + Y_at_phi * binormal_y
        #Z = z0_at_phi           + X_at_phi * normal_z + Y_at_phi * binormal_z
        #R = jnp.sqrt(total_x * total_x + total_y * total_y)
        #phi_cyl=jnp.arctan2(total_y, total_x)
        #X_new = R * jnp.cos(phi_cyl)
        #Y_new = R * jnp.sin(phi_cyl)
        return jnp.array([1.0,1.0, 0.])
        
    def plot(self, r=0.1, ntheta=40, nphi=120, ntheta_fourier=20, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        kwargs.setdefault('alpha', 1)
        import matplotlib.pyplot as plt 
        from matplotlib import cm
        import matplotlib.colors as clr
        from matplotlib.colors import LightSource
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')   
        x_2D_plot, y_2D_plot, z_2D_plot, _ = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2D, theta2D = jnp.meshgrid(phi1D, theta1D)
        import numpy as np
        Bmag = np.array(self.B_mag(r, theta2D, phi2D))
        norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
        cmap = cm.viridis
        ls = LightSource(azdeg=0, altdeg=10)
        cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=cmap_plot,
                        rstride=1, cstride=1, antialiased=False,
                        linewidth=0, shade=False, **kwargs)
        if ax is None or ax.name != "3d":
            ax.dist = 7
            ax.elev = 5
            ax.azim = 45
            cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            cbar = plt.colorbar(m, cax=cbar_ax)
            cbar.ax.set_title(r'$|B| [T]$')
            ax.grid(False)
        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()        