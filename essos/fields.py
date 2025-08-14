import jax
jax.config.update("jax_enable_x64", True)
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
         self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature) = parameters
        
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
         self.L_grad_B, self.inv_L_grad_B, self.torsion, self.curvature) = parameters
    
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
                L_grad_B, inv_L_grad_B, torsion, curvature)
        
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
    def Frenet_to_cylindrical(self, r, ntheta=20):
        nphi_conversion = self.nphi
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi_conversion = jnp.linspace(0, 2 * jnp.pi / self.nfp, nphi_conversion, endpoint=False)

        def compute_for_theta(theta_j):
            costheta = jnp.cos(theta_j)
            sintheta = jnp.sin(theta_j)
            X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
            Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)

            def compute_for_phi(phi_target):
                residual = partial(self.Frenet_to_cylindrical_residual_func, phi_target=phi_target,
                                X_at_this_theta=X_at_this_theta, Y_at_this_theta=Y_at_this_theta)
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


    @partial(jit, static_argnames=['ntheta_fourier', 'mpol', 'ntor', 'ntheta', 'nphi'])
    def get_boundary(self, r=0.1, ntheta=30, nphi=120, ntheta_fourier=20, mpol=5, ntor=5):
        R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier)
        RBC, ZBS = self.to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor)

        theta1D = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi1D = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2D, theta2D = jnp.meshgrid(phi1D, theta1D, indexing='ij')

        def compute_RZ(m, n):
            angle = m * theta2D - n * self.nfp * phi2D
            return RBC[n + ntor, m] * jnp.cos(angle), ZBS[n + ntor, m] * jnp.sin(angle)

        m_vals = jnp.arange(mpol + 1)
        n_vals = jnp.arange(-ntor, ntor + 1)

        R_2Dnew, Z_2Dnew = vmap(lambda m: vmap(lambda n: compute_RZ(m, n))(n_vals))(m_vals)
        R_2Dnew, Z_2Dnew = R_2Dnew.sum(axis=(0, 1)), Z_2Dnew.sum(axis=(0, 1))

        x_2D_plot = R_2Dnew.T * jnp.cos(phi1D)
        y_2D_plot = R_2Dnew.T * jnp.sin(phi1D)
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