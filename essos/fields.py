import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from jax import jit, jacfwd, grad, vmap
from essos.surfaces import SurfaceRZFourier
from essos.plot import fix_matplotlib_3d

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
    def __init__(self, wout, ntheta=50, nphi=50, close=True, range='full torus'):
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

class near_axis():
    # @partial(jit, static_argnums=(0,4,5,6,7,8,9,10))
    def __init__(self, rc=jnp.array([1, 0.1]), zs=jnp.array([0, 0.1]), etabar=1.0,
                    nphi=31, sigma0=0, I2=0, spsi=1, sG=1, B0=1, nfp=2):
        assert nphi % 2 == 1, 'nphi must be odd'
        self.nphi = nphi
        phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
        # d_phi = phi[1] - phi[0]
        nfourier = max(len(rc), len(zs))
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
        self.R0 = R0
        self.Z0 = Z0
        self.phi = phi
        d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
        abs_G0_over_B0 = 1 / B0_over_abs_G0
        d_l_d_varphi = abs_G0_over_B0
        G0 = sG * abs_G0_over_B0 * B0

        d_r_d_phi_cylindrical = jnp.stack([R0p, R0, Z0p]).T
        d2_r_d_phi2_cylindrical = jnp.stack([R0pp - R0, 2 * R0p, Z0pp]).T
        d3_r_d_phi3_cylindrical = jnp.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).T

        tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, None]
        d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] \
                                    +d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, None] * d_l_d_phi[:, None])

        curvature = jnp.sqrt(jnp.sum(d_tangent_d_l_cylindrical**2, axis=1))
        # axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
        # varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

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
        self.elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))
        
        B_axis_cylindrical = sG * B0 * tangent_cylindrical.T
        B_x = jnp.cos(phi) * B_axis_cylindrical[0] - jnp.sin(phi) * B_axis_cylindrical[1]
        B_y = jnp.sin(phi) * B_axis_cylindrical[0] + jnp.cos(phi) * B_axis_cylindrical[1]
        B_z = B_axis_cylindrical[2]
        self.B_axis = jnp.array([B_x, B_y, B_z])

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
        cosphi = jnp.cos(self.phi)
        sinphi = jnp.sin(self.phi)
        self.grad_B_axis = jnp.array([
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
        
    def plot(self, ax=None, show=True, close=False, axis_equal=True, **kwargs):
        if close: raise NotImplementedError("close=True is not implemented, need to have closed surfaces")

        import matplotlib.pyplot as plt 
        from matplotlib import cm
        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
        x_plot = self.R0 * jnp.cos(self.phi)
        y_plot = self.R0 * jnp.sin(self.phi)
        z_plot = self.Z0
        
        plt.plot(x_plot, y_plot, z_plot)
        ax.grid(False)
        # ax.set_xlabel('X (meters)', fontsize=10)
        # ax.set_ylabel('Y (meters)', fontsize=10)
        # ax.set_zlabel('Z (meters)', fontsize=10)

        if axis_equal:
            fix_matplotlib_3d(ax)
        if show:
            plt.show()