"""
MHD interface for ESSOS: VMEC-JAX boundary optimization class
"""
import jax
import jax.numpy as jnp

# Import VMEC-JAX core objects
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import ModeTable
from vmec_jax.driver import solve_fixed_boundary_from_boundary
from vmec_jax.geom import eval_geom


# --- JAX-native, fully differentiable VMEC boundary class ---
from vmec_jax.boundary import boundary_aspect_ratio_from_static
from vmec_jax.field import bsup_from_geom, b_cartesian_from_bsup
from vmec_jax.integrals import volume_from_sqrtg_vmec

class VMECBoundaryJAX:

    """
    Fully JAX-differentiable VMEC boundary class for ESSOS, using only JAX-native routines.
    Provides interface to run VMEC-JAX and extract geometry and physical quantities for optimization.
    """

    def __init__(self, R_cos, R_sin, Z_cos, Z_sin, static, indata, flux, pressure, signgs, modes=None):
        self.R_cos = jnp.asarray(R_cos)
        self.R_sin = jnp.asarray(R_sin)
        self.Z_cos = jnp.asarray(Z_cos)
        self.Z_sin = jnp.asarray(Z_sin)
        self.static = static
        self.indata = indata
        self.flux = flux
        self.pressure = pressure
        self.signgs = signgs
        self.modes = modes or static.modes


    @classmethod
    def from_vmec_input(cls, input_path, signgs=None):
        """
        Initialize VMECBoundaryJAX from a standard VMEC input file.
        Handles optional pressure/current profiles if present.
        Args:
            input_path: Path to VMEC input file
            signgs: Optional signgs override (default: -1)
        Returns:
            VMECBoundaryJAX instance
        """
        from vmec_jax.namelist import read_indata
        from vmec_jax.config import config_from_indata
        from vmec_jax.static import build_static
        from vmec_jax.boundary import boundary_from_indata
        from vmec_jax.energy import flux_profiles_from_indata
        indata = read_indata(input_path)
        cfg = config_from_indata(indata)
        static = build_static(cfg)
        modes = static.modes
        boundary = boundary_from_indata(indata, modes)
        # Handle signgs (VMEC sign convention for Jacobian)
        if signgs is None:
            signgs = int(indata.get("SIGNGS", -1))
        # Handle pressure profile (APRES, PRES_PROFILE, etc.)
        pressure = indata.get("PRES_PROFILE", None)
        if pressure is None:
            pressure = indata.get("APRES", None)
        # Optionally handle current profile if needed (not always present)
        # Build flux profiles
        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        return cls(
            boundary.R_cos, boundary.R_sin, boundary.Z_cos, boundary.Z_sin,
            static, indata, flux, pressure, signgs, modes
        )

    @property
    def state(self):
        """
        JAX-safe: always recompute state from current dofs (pure function).
        """
        return self.run_vmec(self.dofs)

    @property
    def geom(self):
        """
        JAX-safe: always recompute geometry from current dofs (pure function).
        """
        return self.get_geom(self.state)

    @property
    def dofs(self):
        return jnp.concatenate([self.R_cos, self.R_sin, self.Z_cos, self.Z_sin])

    @dofs.setter
    def dofs(self, new_dofs):
        K = self.modes.K
        self.R_cos = new_dofs[:K]
        self.R_sin = new_dofs[K:2*K]
        self.Z_cos = new_dofs[2*K:3*K]
        self.Z_sin = new_dofs[3*K:4*K]

    def get_boundary_coeffs(self):
        return BoundaryCoeffs(
            R_cos=self.R_cos,
            R_sin=self.R_sin,
            Z_cos=self.Z_cos,
            Z_sin=self.Z_sin,
        )

    def run_vmec(self, dofs, **kwargs):
        print("[VMEC] Running VMEC solver for new dofs...")
        """
        Pure function: run VMEC solver for given dofs, return state.
        Args:
            dofs: 1D array of boundary dofs (R_cos, R_sin, Z_cos, Z_sin concatenated)
            **kwargs: extra solver options
        Returns:
            state: VMEC solution object
        """
        K = self.modes.K
        R_cos = dofs[:K]
        R_sin = dofs[K:2*K]
        Z_cos = dofs[2*K:3*K]
        Z_sin = dofs[3*K:4*K]
        boundary = BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)
        grid_keys = ["ntheta", "nzeta", "nphi", "nfp"]
        vmec_kwargs = {k: v for k, v in kwargs.items() if k not in grid_keys}
        result = solve_fixed_boundary_from_boundary(
            boundary=boundary,
            static=self.static,
            indata=self.indata,
            flux=self.flux,
            pressure=self.pressure,
            signgs=self.signgs,
            **vmec_kwargs
        )
        state = result
        profiles = None
        if hasattr(result, 'state') and hasattr(result, 'profiles'):
            state = result.state
            profiles = result.profiles
        if not hasattr(state, 'profiles') or state.profiles is None:
            profiles_dict = profiles if profiles is not None else {}
            if hasattr(state, 'iotas') and state.iotas is not None:
                profiles_dict['iota'] = state.iotas
            if hasattr(state, 'iotaf') and state.iotaf is not None:
                profiles_dict['iotaf'] = state.iotaf
            try:
                setattr(state, 'profiles', profiles_dict)
            except Exception:
                pass
        return state


    def _compute_iota_profile_jax(self, full_mesh=False):
        """
        JAX-differentiable computation of the iota profile from input data and static grid.
        Matches VMEC driver logic for profile evaluation.
        Args:
            full_mesh: If True, return full-mesh iotaf, else half-mesh iota
        Returns:
            iota array (half-mesh or full-mesh)
        """
        import jax.numpy as jnp
        from vmec_jax.energy import _iotaf_from_iotas
        from vmec_jax.profiles import eval_profiles
        # Use the static grid for s
        s = self.static.s
        # Evaluate profiles on s (half-mesh)
        profiles = eval_profiles(self.indata, s)
        iota = profiles.get("iota", None)
        if iota is None:
            raise AttributeError("Input data does not define an iota profile (AI coefficients missing?)")
        if full_mesh:
            lrfp = bool(self.indata.get("LRFP", False))
            iotaf = _iotaf_from_iotas(iota, lrfp=lrfp)
            return iotaf
        return iota
    
    def get_geom(self, state=None, **kwargs):
        import inspect
        # Separate grid args for eval_geom
        grid_keys = ["ntheta", "nzeta", "nphi", "nfp"]
        grid_kwargs = {k: v for k, v in kwargs.items() if k in grid_keys}
        eval_geom_sig = inspect.signature(eval_geom)
        allowed = {k: v for k, v in grid_kwargs.items() if k in eval_geom_sig.parameters}
        if state is None:
            state = self.state
        geom = eval_geom(state, self.static, **allowed)
        return geom

    def B_on_surface(self, s_index=0, **kwargs):
        geom = self.get_geom(**kwargs)
        bsupu, bsupv = bsup_from_geom(
            geom,
            phipf=self.flux.phipf,
            chipf=self.flux.chipf,
            nfp=self.static.cfg.nfp,
            signgs=self.signgs,
            lamscale=self.flux.lamscale,
        )
        zeta = self.static.grid.zeta
        B_cart = b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=zeta, nfp=self.static.cfg.nfp)
        return B_cart[s_index]

    def iota(self, s_index=None, full_mesh=False):
        """
        Extract the iota profile as written to wout (half-mesh by default, full-mesh if requested).
        If not available from the state, compute it JAX-differentiably from input data.
        Args:
            s_index: Optional index to select a single value
            full_mesh: If True, return full-mesh iotaf (VMEC convention)
            **kwargs: Passed to run_vmec
        Returns:
            iota array (half-mesh or full-mesh), or single value if s_index is given
        """
        state = self.state
        profiles = getattr(state, 'profiles', None)
        iota_arr = None
        if profiles is not None:
            if full_mesh and 'iotaf' in profiles:
                iota_arr = profiles['iotaf']
            elif not full_mesh and 'iota' in profiles:
                iota_arr = profiles['iota']
        if iota_arr is None:
            if full_mesh and hasattr(state, 'iotaf'):
                iota_arr = state.iotaf
            elif not full_mesh and hasattr(state, 'iotas'):
                iota_arr = state.iotas
        if iota_arr is not None:
            if s_index is None:
                return iota_arr
            return iota_arr[s_index]
        # Fallback: compute JAX-differentiable iota profile from input
        iota_arr = self._compute_iota_profile_jax(full_mesh=full_mesh)
        if s_index is None:
            return iota_arr
        return iota_arr[s_index]

    def volume(self, s_index=None, **kwargs):
        geom = self.get_geom(**kwargs)
        # Use JAX-native volume computation
        _, vol = volume_from_sqrtg_vmec(
            geom.sqrtg,
            self.static.s,
            self.static.grid.theta,
            self.static.grid.zeta,
            signgs=self.signgs,
        )
        if s_index is None:
            return vol
        return vol[s_index]

    def aspect_ratio(self):
        boundary = self.get_boundary_coeffs()
        return boundary_aspect_ratio_from_static(boundary, self.static)


    def vacuum_well(self, **kwargs):
        """
        Compute a single number W that summarizes the vacuum magnetic well,
        using the formula:
        W = (dV/ds(s=0) - dV/ds(s=1)) / dV/ds(s=0)
        where dV/ds = 4 * pi**2 * abs(sqrt(g)_{0,0})
        sqrt(g) is the Jacobian on the half-mesh, m=n=0 Fourier component.
        """
        import jax.numpy as jnp
        geom = self.get_geom(**kwargs)
        sqrtg = geom.sqrtg  # shape: (ns, ntheta, nzeta)
        # Take m=n=0 Fourier component: average over theta, zeta
        sqrtg00 = jnp.mean(sqrtg, axis=(1,2))  # (ns,)
        dVds = 4 * jnp.pi**2 * jnp.abs(sqrtg00)
        # Extrapolate to s=0 and s=1 using first and last two points
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well


    def dshear(self, **kwargs):
        """
        Mercier shear term (JAX-native, approximate):
        dshear = 0.25 * (diota/ds)^2, where iota is the rotational transform profile.
        Uses robust iota extraction (JAX-differentiable fallback).
        """
        s = self.static.s
        try:
            iota = self.iota(**kwargs)
        except Exception:
            iota = self._compute_iota_profile_jax()
        diota_ds = jnp.gradient(iota, s)
        dshear = 0.25 * diota_ds**2
        return dshear

    def dcurr(self, **kwargs):
        """
        Mercier current term (JAX-native, proxy):
        Approximate as finite-difference of iota (proxy for current gradient).
        True current profile would require more info.
        Uses robust iota extraction (JAX-differentiable fallback).
        """
        s = self.static.s
        try:
            iota = self.iota(**kwargs)
        except Exception:
            iota = self._compute_iota_profile_jax()
        d2iota_ds2 = jnp.gradient(jnp.gradient(iota, s), s)
        dcurr = -s * d2iota_ds2
        return dcurr

    def dwell(self, **kwargs):
        """
        Mercier well term (JAX-native, approximate):
        Use second derivative of volume profile as a proxy for well depth.
        """
        geom = self.get_geom(**kwargs)
        sqrtg = geom.sqrtg  # (ns, ntheta, nzeta)
        sqrtg00 = jnp.mean(sqrtg, axis=(1,2))  # (ns,)
        dVds = 4 * jnp.pi**2 * jnp.abs(sqrtg00)
        s = self.static.s
        d2Vds2 = jnp.gradient(jnp.gradient(dVds, s), s)
        dVds_s0 = dVds[0]
        dwell = -d2Vds2 / dVds_s0
        return dwell

    def dgeod(self, **kwargs):
        """
        Mercier geodesic term (JAX-native, proxy):
        Use a geometric curvature proxy based on the volume profile.
        True geodesic term would require more info.
        """
        geom = self.get_geom(**kwargs)
        sqrtg = geom.sqrtg  # (ns, ntheta, nzeta)
        sqrtg00 = jnp.mean(sqrtg, axis=(1,2))  # (ns,)
        dVds = 4 * jnp.pi**2 * jnp.abs(sqrtg00)
        s = self.static.s
        # Use third derivative of volume as a geometric proxy
        d3Vds3 = jnp.gradient(jnp.gradient(jnp.gradient(dVds, s), s), s)
        dgeod = d3Vds3 / (jnp.abs(dVds[0]) + 1e-12)
        return dgeod

    def DMerc(self, **kwargs):
        """
        Mercier index (sum of all components, JAX-native):
        """
        return self.dshear(**kwargs) + self.dcurr(**kwargs) + self.dwell(**kwargs) + self.dgeod(**kwargs)



    def volume_averaged_B(self, **kwargs):
        """
        Compute the volume-averaged |B| over the plasma volume (JAX-native).
        Returns:
            Scalar: <|B|>_V
        """
        import jax.numpy as jnp
        geom = self.get_geom(**kwargs)
        sqrtg = geom.sqrtg  # (ns, ntheta, nzeta)
        Bmod = geom.Bmod    # (ns, ntheta, nzeta)
        s = self.static.s
        theta = self.static.grid.theta
        zeta = self.static.grid.zeta
        dtheta = theta[1] - theta[0]
        dzeta = zeta[1] - zeta[0]
        # Integrate |B| * sqrt(g) over all grid points and s
        integrand = Bmod * jnp.abs(sqrtg)
        dV = jnp.sum(jnp.sum(integrand, axis=-1), axis=-1) * dtheta * dzeta  # (ns,)
        V = jnp.trapz(dV, s)
        Bint = jnp.trapz(dV * jnp.mean(Bmod, axis=(1,2)), s)
        return Bint / (V + 1e-12)

    def volume_averaged_beta(self, **kwargs):
        """
        Compute the volume-averaged beta = <p>_V / (<B^2>_V / 2mu0) (JAX-native).
        Returns:
            Scalar: <beta>_V
        """
        import jax.numpy as jnp
        mu0 = 4 * jnp.pi * 1e-7
        geom = self.get_geom(**kwargs)
        sqrtg = geom.sqrtg  # (ns, ntheta, nzeta)
        Bmod = geom.Bmod    # (ns, ntheta, nzeta)
        s = self.static.s
        theta = self.static.grid.theta
        zeta = self.static.grid.zeta
        dtheta = theta[1] - theta[0]
        dzeta = zeta[1] - zeta[0]
        # Pressure profile: self.pressure (assume shape (ns,))
        if self.pressure is None:
            raise ValueError("No pressure profile available in VMECBoundaryJAX instance.")
        p = jnp.asarray(self.pressure)
        # Integrate p * sqrt(g) and B^2 * sqrt(g) over all grid points and s
        integrand_p = p[:, None, None] * jnp.abs(sqrtg)
        integrand_B2 = Bmod**2 * jnp.abs(sqrtg)
        dV = jnp.sum(jnp.sum(jnp.abs(sqrtg), axis=-1), axis=-1) * dtheta * dzeta  # (ns,)
        p_int = jnp.trapz(jnp.sum(jnp.sum(integrand_p, axis=-1), axis=-1) * dtheta * dzeta, s)
        B2_int = jnp.trapz(jnp.sum(jnp.sum(integrand_B2, axis=-1), axis=-1) * dtheta * dzeta, s)
        V = jnp.trapz(dV, s)
        p_avg = p_int / (V + 1e-12)
        B2_avg = B2_int / (V + 1e-12)
        beta = p_avg / (B2_avg / (2 * mu0) + 1e-20)
        return beta

    def get_boozer_field(self, s_index=0, mboz=8, nboz=8, asym=False, jit=True, **kwargs):
        """
        Compute the magnetic field in Boozer coordinates for a given surface using booz_xform_jax (JAX-native, differentiable).
        Args:
            s_index: Index of the surface (on VMEC half-grid) to transform.
            mboz: Maximum poloidal mode number in Boozer spectrum.
            nboz: Maximum toroidal mode number in Boozer spectrum.
            asym: Whether to use asymmetric (non-stellsym) Boozer transform.
            jit: Whether to JIT the Boozer transform (default True).
            **kwargs: Passed to get_geom (e.g., for VMEC solver options).
        Returns:
            Dictionary of Boozer field quantities for the selected surface.
        """
        import jax.numpy as jnp
        from booz_xform_jax.jax_api import booz_xform_jax
        geom = self.get_geom(**kwargs)
        static = self.static
        nfp = static.cfg.nfp
        # Prepare VMEC Fourier data for booz_xform_jax
        # These arrays should be (ns, mn) or (ns, ...)
        rmnc = jnp.asarray(geom.rmnc)  # (ns, mn)
        zmns = jnp.asarray(geom.zmns)  # (ns, mn)
        lmns = jnp.asarray(geom.lmns)  # (ns, mn)
        bmnc = jnp.asarray(geom.bmnc)  # (ns, mn)
        bsubumnc = jnp.asarray(geom.bsubumnc)  # (ns, mn)
        bsubvmnc = jnp.asarray(geom.bsubvmnc)  # (ns, mn)
        iota = jnp.asarray(geom.iota)  # (ns,)
        xm = jnp.asarray(static.modes.xm, dtype=jnp.int32)
        xn = jnp.asarray(static.modes.xn, dtype=jnp.int32)
        xm_nyq = jnp.asarray(static.modes.xm_nyq, dtype=jnp.int32)
        xn_nyq = jnp.asarray(static.modes.xn_nyq, dtype=jnp.int32)
        # Call booz_xform_jax for the selected surface
        out = booz_xform_jax(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=xm,
            xn=xn,
            xm_nyq=xm_nyq,
            xn_nyq=xn_nyq,
            nfp=nfp,
            mboz=mboz,
            nboz=nboz,
            asym=asym,
            surface_indices=[s_index],
        )
        return out
    
    def triple_product_metric(self, surfaces, helicity_m, helicity_n, weights=None, ntheta=32, nphi=32, blocksize=None):
        """
        JAX-native, differentiable quasisymmetry metric (triple product metric),
        mathematically equivalent to QuasisymmetryRatioResidual in SIMSOPT.
        Args:
            surfaces: list/array of normalized toroidal flux values (s_j)
            helicity_m: integer M (poloidal mode)
            helicity_n: integer N (toroidal mode)
            weights: list of weights w_j (default 1)
            ntheta: number of poloidal grid points
            nphi: number of toroidal grid points per field period
            blocksize: number of surfaces per block (None means no blocking)
        Returns:
            Scalar metric value (JAX-differentiable)
        """
        import jax
        import jax.numpy as jnp
        static = self.static
        nfp = static.cfg.nfp
        s_arr = jnp.atleast_1d(jnp.asarray(surfaces))
        if weights is None:
            weights = jnp.ones_like(s_arr)
        weights = jnp.asarray(weights)
        s_grid = static.s
        N = helicity_n * nfp
        M = helicity_m
        dtheta = 2 * jnp.pi / ntheta
        dzeta = (2 * jnp.pi / nfp) / nphi
        geom = self.geom

        def metric_single(sj, wj):
            # Find closest s index
            s_idx = jnp.argmin(jnp.abs(s_grid - sj))
            # Extract geometry at this surface
            sqrtg = geom.sqrtg[s_idx]  # (ntheta, nphi)
            # Compute B contravariant components
            bsupu, bsupv = bsup_from_geom(
                geom,
                phipf=self.flux.phipf,
                chipf=self.flux.chipf,
                nfp=nfp,
                signgs=self.signgs,
                lamscale=self.flux.lamscale,
            )
            # Use the actual shape from geom arrays
            shape = geom.Rt[s_idx].shape  # (ntheta, nphi)
            ntheta_g, nphi_g = shape
            # Build cosphi, sinphi with correct shape
            zeta = jnp.linspace(0, 2 * jnp.pi / nfp, nphi_g, endpoint=False)
            phi = zeta / nfp
            cosphi = jnp.cos(phi)
            sinphi = jnp.sin(phi)
            # Broadcast to (ntheta, nphi)
            cosphi2d = jnp.broadcast_to(cosphi, (ntheta_g, nphi_g))
            sinphi2d = jnp.broadcast_to(sinphi, (ntheta_g, nphi_g))
            # e_theta, e_phi: (ntheta, nphi, 3)
            e_theta = jnp.stack([
                geom.Rt[s_idx] * cosphi2d,
                geom.Rt[s_idx] * sinphi2d,
                geom.Zt[s_idx]
            ], axis=-1)
            e_phi = jnp.stack([
                geom.Rp[s_idx] * cosphi2d - geom.R[s_idx] * sinphi2d,
                geom.Rp[s_idx] * sinphi2d + geom.R[s_idx] * cosphi2d,
                geom.Zp[s_idx]
            ], axis=-1)
            # B vector in Cartesian
            Bvec = bsupu[s_idx][..., None] * e_theta + bsupv[s_idx][..., None] * e_phi  # (ntheta, nphi, 3)
            # |B|
            Bmod = jnp.linalg.norm(Bvec, axis=-1)
            # Compute grad B (finite difference)
            dB_dtheta = jnp.gradient(Bmod, axis=0) / dtheta
            dB_dphi = jnp.gradient(Bmod, axis=1) / dzeta
            # grad B in (theta, phi) basis
            gradB = dB_dtheta[..., None] * e_theta + dB_dphi[..., None] * e_phi  # (ntheta, nphi, 3)
            # Compute grad psi (radial direction): e_s
            e_s = jnp.stack([
                geom.Rs[s_idx] * cosphi2d,
                geom.Rs[s_idx] * sinphi2d,
                geom.Zs[s_idx]
            ], axis=-1)
            gradpsi = e_s  # (ntheta, nphi, 3)
            # B x grad B · grad psi
            BxGradB = jnp.cross(Bvec, gradB)
            BxGradB_dot_GradPsi = jnp.sum(BxGradB * gradpsi, axis=-1)
            # B · grad B
            BdotGradB = jnp.sum(Bvec * gradB, axis=-1)
            # iota, G, I
            iota = self.iota()[s_idx]
            G = getattr(static, 'G', 0.0)
            I = getattr(static, 'I', 0.0)
            numer = (N - iota * M) * BxGradB_dot_GradPsi - (M * G + N * I) * BdotGradB
            denom = Bmod ** 3 + 1e-12
            R = numer / denom
            Vp = jnp.sum(jnp.abs(sqrtg)) * dtheta * dzeta
            integrand = (R ** 2) * jnp.abs(sqrtg)
            avg = jnp.sum(integrand) * dtheta * dzeta / (Vp + 1e-12)
            return wj * avg

        if blocksize is None:
            metric = jnp.sum(jax.vmap(metric_single)(s_arr, weights))
        else:
            def block_sum(start):
                end = jnp.minimum(start + blocksize, s_arr.shape[0])
                idxs = jnp.arange(start, end)
                return jnp.sum(jax.vmap(metric_single)(s_arr[idxs], weights[idxs]))
            nblocks = (s_arr.shape[0] + blocksize - 1) // blocksize
            starts = jnp.arange(0, s_arr.shape[0], blocksize)
            metric = jnp.sum(jax.vmap(block_sum)(starts))
        return metric


    def write_wout(self, filename, **kwargs):
        """
        Run VMEC and write a wout file using vmec_jax's write_wout_from_fixed_boundary_run.
        Args:
            filename: Output wout file path (should end with .nc)
            **kwargs: Passed to run_vmec
        """
        from vmec_jax.driver import write_wout_from_fixed_boundary_run
        boundary = self.get_boundary_coeffs()
        write_wout_from_fixed_boundary_run(
            filename,
            boundary=boundary,
            static=self.static,
            indata=self.indata,
            flux=self.flux,
            pressure=self.pressure,
            signgs=self.signgs,
            **kwargs
        )

    def plot_B_contour(self, s=None, s_index=None, nfp=None, ntheta=64, nzeta=64, **kwargs):
        """
        Plot a contour of |B| on a surface in (phi, theta).
        Args:
            s: Normalized flux (0 to 1). If given, will use the closest surface index.
            s_index: Surface index (overrides s if both given).
            nfp: Number of field periods (default from static)
            ntheta: Poloidal grid points
            nzeta: Toroidal grid points
            **kwargs: Passed to get_geom
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from vmec_jax.field import bsup_from_geom
        geom = self.get_geom(ntheta=ntheta, nzeta=nzeta, **kwargs)
        if nfp is None:
            nfp = self.static.cfg.nfp
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zeta = np.linspace(0, 2*np.pi/nfp, nzeta, endpoint=False)
        # Find surface index from s if needed
        if s_index is None:
            if s is not None:
                s_grid = np.asarray(self.static.s)
                s_index = int(np.argmin(np.abs(s_grid - s)))
            else:
                s_index = 0
        # Compute B contravariant components
        bsupu, bsupv = bsup_from_geom(
            geom,
            phipf=self.flux.phipf,
            chipf=self.flux.chipf,
            nfp=nfp,
            signgs=self.signgs,
            lamscale=self.flux.lamscale,
        )
        # Compute |B| using the metric
        g_tt = np.asarray(geom.g_tt[s_index])
        g_tp = np.asarray(geom.g_tp[s_index])
        g_pp = np.asarray(geom.g_pp[s_index])
        Bmod = np.sqrt(g_tt * np.asarray(bsupu[s_index])**2 + 2.0 * g_tp * np.asarray(bsupu[s_index]) * np.asarray(bsupv[s_index]) + g_pp * np.asarray(bsupv[s_index])**2)
        # Robustly handle Bmod shape for plotting
        shape_ok = (Bmod.shape == (ntheta, nzeta))
        if not shape_ok:
            print(f"[WARNING] Bmod shape {Bmod.shape} does not match (ntheta, nzeta)=({ntheta},{nzeta}). Using native shape for plotting.")
            # Try to infer axes from Bmod shape
            if len(Bmod.shape) == 2:
                theta_plot = np.arange(Bmod.shape[0])
                zeta_plot = np.arange(Bmod.shape[1])
            elif len(Bmod.shape) == 1:
                # Try to guess a square-ish shape
                n = int(np.sqrt(Bmod.size))
                if n * n == Bmod.size:
                    Bmod = Bmod.reshape((n, n))
                    theta_plot = np.arange(n)
                    zeta_plot = np.arange(n)
                else:
                    theta_plot = np.arange(Bmod.size)
                    zeta_plot = [0]
            else:
                theta_plot = np.arange(Bmod.shape[0])
                zeta_plot = np.arange(Bmod.shape[1]) if len(Bmod.shape) > 1 else [0]
        else:
            theta_plot = theta
            zeta_plot = zeta
        cp = plt.contourf(zeta_plot, theta_plot, Bmod, 50, cmap='viridis')
        plt.colorbar(cp, label='|B|')
        plt.xlabel('phi (toroidal)')
        plt.ylabel('theta (poloidal)')
        plt.title(f'|B| contour on surface s={getattr(self.static.s, "__getitem__", lambda x: x)(s_index):.3f} (index {s_index})')
        plt.show()

    def plot_3D_surface(self, s=None, s_index=None, ntheta=64, nzeta=64, color_by_B=False, **kwargs):
        """
        Plot the 3D surface for a given s (normalized flux) or s_index using matplotlib.
        Optionally color the surface by |B|, computed from the metric and field.
        Args:
            s: Normalized flux (0 to 1). If given, will use the closest surface index.
            s_index: Surface index (overrides s if both given).
            ntheta: Poloidal grid points
            nzeta: Toroidal grid points
            color_by_B: If True, color the surface by |B| (default False)
            **kwargs: Passed to get_geom
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        geom = self.get_geom(ntheta=ntheta, nzeta=nzeta, **kwargs)
        nfp = self.static.cfg.nfp
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zeta = np.linspace(0, 2*np.pi/nfp, nzeta, endpoint=False)
        # Find surface index from s if needed
        if s_index is None:
            if s is not None:
                s_grid = np.asarray(self.static.s)
                s_index = int(np.argmin(np.abs(s_grid - s)))
            else:
                s_index = 0
        Theta, Zeta = np.meshgrid(theta, zeta, indexing='ij')
        # Fourier synthesis for R, Z
        # Try to auto-detect the correct attribute names for Fourier coefficients
        if hasattr(geom, 'rmnc') and hasattr(geom, 'zmns'):
            Rmn = np.asarray(geom.rmnc[s_index])
            Zmn = np.asarray(geom.zmns[s_index])
            print("[INFO] Using 'rmnc' and 'zmns' for Fourier coefficients.")
        elif hasattr(geom, 'R_cosa') and hasattr(geom, 'Z_sina'):
            Rmn = np.asarray(geom.R_cosa[s_index])
            Zmn = np.asarray(geom.Z_sina[s_index])
            print("[INFO] Using 'R_cosa' and 'Z_sina' for Fourier coefficients.")
        elif hasattr(geom, 'R_cos') and hasattr(geom, 'Z_sin'):
            Rmn = np.asarray(geom.R_cos[s_index])
            Zmn = np.asarray(geom.Z_sin[s_index])
            print("[INFO] Using 'R_cos' and 'Z_sin' for Fourier coefficients.")
        else:
            print("[ERROR] geom object does not have recognized Fourier coefficient attributes. Available attributes:")
            print(sorted(dir(geom)))
            print("Cannot plot 3D surface. Check VMEC-JAX version and eval_geom output.")
            return
        xm = np.array(self.static.modes.xm)
        xn = np.array(self.static.modes.xn)
        R = np.zeros_like(Theta)
        Z = np.zeros_like(Theta)
        for k in range(len(Rmn)):
            angle = xm[k]*Theta + xn[k]*Zeta*nfp
            R += Rmn[k]*np.cos(angle)
            Z += Zmn[k]*np.sin(angle)
        X = R * np.cos(Zeta*nfp)
        Y = R * np.sin(Zeta*nfp)
        # Optionally color by |B|
        if color_by_B:
            from vmec_jax.field import bsup_from_geom
            bsupu, bsupv = bsup_from_geom(
                geom,
                phipf=self.flux.phipf,
                chipf=self.flux.chipf,
                nfp=nfp,
                signgs=self.signgs,
                lamscale=self.flux.lamscale,
            )
            g_tt = np.asarray(geom.g_tt[s_index])
            g_tp = np.asarray(geom.g_tp[s_index])
            g_pp = np.asarray(geom.g_pp[s_index])
            Bmod = np.sqrt(g_tt * np.asarray(bsupu[s_index])**2 + 2.0 * g_tp * np.asarray(bsupu[s_index]) * np.asarray(bsupv[s_index]) + g_pp * np.asarray(bsupv[s_index])**2)
            if Bmod.shape != (ntheta, nzeta):
                Bmod = Bmod.reshape((ntheta, nzeta))
            facecolors = plt.cm.viridis((Bmod - Bmod.min()) / (Bmod.ptp() + 1e-12))
        else:
            facecolors = None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Surface s={getattr(self.static.s, "__getitem__", lambda x: x)(s_index):.3f} (index {s_index})')
        if color_by_B:
            mappable = plt.cm.ScalarMappable(cmap='viridis')
            mappable.set_array(Bmod)
            fig.colorbar(mappable, ax=ax, label='|B|')
        plt.show()


# Register as a JAX pytree for optimization
from jax import tree_util

def _vmecboundaryjax_flatten(obj):
    children = (obj.R_cos, obj.R_sin, obj.Z_cos, obj.Z_sin)
    aux = (obj.static, obj.indata, obj.flux, obj.pressure, obj.signgs, obj.modes)
    return children, aux

def _vmecboundaryjax_unflatten(aux, children):
    R_cos, R_sin, Z_cos, Z_sin = children
    static, indata, flux, pressure, signgs, modes = aux
    return VMECBoundaryJAX(R_cos, R_sin, Z_cos, Z_sin, static, indata, flux, pressure, signgs, modes)

tree_util.register_pytree_node(VMECBoundaryJAX, _vmecboundaryjax_flatten, _vmecboundaryjax_unflatten)
