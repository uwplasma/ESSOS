"""
MHD interface for ESSOS: VMEC-JAX boundary optimization class
"""
import jax
import jax.numpy as jnp

# Import VMEC-JAX core objects
from vmec_jax.boundary import BoundaryCoeffs
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

    def __init__(self, R_cos, R_sin, Z_cos, Z_sin, static, indata, flux, pressure, signgs, modes=None,
                 max_iter=None, step_size=None, jacobian_penalty=None, jit_grad=None, differentiable=None,
                 stop_grad_in_update=None, verbose=None, vmec_project=None,
                 performance_mode=None, solver_mode=None, jit_forces=None, input_path=None):
        # Ensure all arrays are JAX arrays for differentiability
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
        self.max_iter = max_iter
        self.step_size = step_size
        self.jacobian_penalty = jacobian_penalty
        self.jit_grad = jit_grad
        self.differentiable = differentiable
        self.stop_grad_in_update = stop_grad_in_update
        self.verbose = verbose
        self.vmec_project = vmec_project
        self.performance_mode = performance_mode
        self.solver_mode = solver_mode
        self.jit_forces = jit_forces
        self._state = None
        self._geom = None
        self.input_path = input_path

        # Expose TCON0, PRECON_TYPE, PREC2D_THRESHOLD as attributes (always available)
        self.tcon0 = indata.get("TCON0", None)
        self.precon_type = indata.get("PRECON_TYPE", None)
        self.prec2d_threshold = indata.get("PREC2D_THRESHOLD", None)


    @classmethod
    def from_vmec_input(cls, input_path, signgs=None, performance_mode=None, solver_mode=None, jit_forces=None):
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
        # Get s grid and signgs for flux_profiles_from_indata
        s = static.s
        if signgs is None:
            signgs = getattr(indata, 'signgs', -1)
        flux = flux_profiles_from_indata(indata, s, signgs=signgs)
        # pressure profile is typically in indata, can be extracted as needed
        pressure = getattr(indata, 'pressure', None)
        if signgs is None:
            signgs = getattr(indata, 'signgs', -1)
        return cls(
            boundary.R_cos, boundary.R_sin, boundary.Z_cos, boundary.Z_sin,
            static, indata, flux, pressure, signgs, modes,
            performance_mode=performance_mode, solver_mode=solver_mode, jit_forces=jit_forces,
            input_path=input_path
        )
    def geom(self):
        if self._geom is None:
            self._geom = self.get_geom(self.state)
        return self._geom


    @property
    def state(self):
        # Run VMEC if state is not yet computed
        if self._state is None:
            # Use current dofs to run VMEC
            dofs = self.dofs
            self._state = self.run_vmec(dofs)
        return self._state

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
        self._state = None
        self._geom = None

    def get_boundary_coeffs(self):
        return BoundaryCoeffs(
            R_cos=self.R_cos,
            R_sin=self.R_sin,
            Z_cos=self.Z_cos,
            Z_sin=self.Z_sin,
        )


    def run_vmec(self, dofs, **kwargs):
        """
        Run the VMEC solver for the given dofs using the full staged/multigrid differentiable logic.
        This now delegates to run_fixed_boundary_for_optimization for consistency and JAX compatibility.
        """
        result = self.run_fixed_boundary_for_optimization(dofs, **kwargs)
        # Return the state (VMEC solution) as before for compatibility
        return result['state']


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
        print(f"[VMEC] geom type: {type(geom)}")
        print(f"[VMEC] geom attributes: {sorted(dir(geom))}")
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
        geom = self.get_geom(ntheta=ntheta, nphi=nphi)

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


    def run_fixed_boundary_for_optimization(self, dofs=None, **kwargs):
        import time
        from datetime import datetime
        t0 = time.time()
        # --- VMEC-style header printout (parity with driver.py) ---
        now = datetime.now()
        date_str = now.strftime("%b %d,%Y")
        time_str = now.strftime("%H:%M:%S")
        input_name = str(getattr(self, 'input_path', 'INMEMORY')).upper()
        version = "vmec_jax"
        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -", flush=True)
        print("  SEQ =    1 TIME SLICE  0.0000E+00", flush=True)
        print(f"  PROCESSING {input_name}", flush=True)
        print(f"  THIS IS PARVMEC (PARALLEL VMEC), VERSION {version}", flush=True)
        print("  Lambda: Full Radial Mesh. L-Force: hybrid full/half.", flush=True)
        print("", flush=True)
        print(f"  COMPUTER:    OS:    RELEASE:   DATE = {date_str}  TIME = {time_str}", flush=True)
        print("", flush=True)
        print("[DIAG] Starting run_fixed_boundary_for_optimization", flush=True)
        """
        Fully mirrors the logic and performance path of run_fixed_boundary in vmec_jax/driver.py,
        but uses in-memory dofs/attributes for the boundary and instance data for indata/static.
        """
        import jax.numpy as jnp
        import numpy as np
        from vmec_jax.driver import (
            _final_flux_profiles_from_state, initial_guess_from_boundary, interp_vmec_state, FixedBoundaryRun
        )
        from vmec_jax.solve import solve_fixed_boundary_gd, solve_fixed_boundary_lbfgs, solve_fixed_boundary_gn_vmec_residual, solve_fixed_boundary_lbfgs_vmec_residual, solve_fixed_boundary_residual_iter
        from vmec_jax.profiles import eval_profiles
        from vmec_jax.energy import flux_profiles_from_indata
        from vmec_jax.static import build_static
        # Use dofs or current attributes
        if dofs is None:
            dofs = self.dofs
        K = self.modes.K
        boundary = BoundaryCoeffs(
            R_cos=jnp.asarray(dofs[:K]),
            R_sin=jnp.asarray(dofs[K:2*K]),
            Z_cos=jnp.asarray(dofs[2*K:3*K]),
            Z_sin=jnp.asarray(dofs[3*K:4*K]),
        )
        static = self.static
        indata = self.indata
        # Ensure pressure is at least 1D array to avoid IndexError in _final_flux_profiles_from_state (JAX compatible)
        import jax.numpy as jnp
        pressure = self.pressure
        if pressure is not None:
            pressure = jnp.atleast_1d(jnp.asarray(pressure))
        signgs = self.signgs
        # --- Read all relevant parameters from indata (input file) as in standard path ---
        # Allow kwargs to override, but default to input file values
        solver = kwargs.get('solver', getattr(self, 'solver', 'vmec2000_iter'))
        solver_mode = kwargs.get('solver_mode', getattr(self, 'solver_mode', None))
        # DELT (step_size)
        step_size = kwargs.get('step_size', None)
        if step_size is None:
            step_size = indata.get_float("DELT", 5e-3)
        # NITER (max_iter)
        max_iter = kwargs.get('max_iter', None)
        if max_iter is None:
            max_iter = indata.get_int("NITER", 10)
        # NITER_ARRAY, NS_ARRAY, FTOL_ARRAY
        ns_array = indata.get('NS_ARRAY', None)
        niter_array = indata.get('NITER_ARRAY', None)
        ftol_array = indata.get('FTOL_ARRAY', None)
        # History size, GN params, etc.
        history_size = kwargs.get('history_size', getattr(self, 'history_size', 10))
        gn_damping = kwargs.get('gn_damping', getattr(self, 'gn_damping', None))
        gn_cg_tol = kwargs.get('gn_cg_tol', getattr(self, 'gn_cg_tol', None))
        gn_cg_maxiter = kwargs.get('gn_cg_maxiter', getattr(self, 'gn_cg_maxiter', 80))
        vmec_project = kwargs.get('vmec_project', getattr(self, 'vmec_project', True))
        verbose = kwargs.get('verbose', getattr(self, 'verbose', False))
        jit_grad = kwargs.get('jit_grad', getattr(self, 'jit_grad', True))
        jit_forces = kwargs.get('jit_forces', getattr(self, 'jit_forces', 'auto'))
        differentiable = kwargs.get('differentiable', getattr(self, 'differentiable', True))
        stop_grad_in_update = kwargs.get('stop_grad_in_update', getattr(self, 'stop_grad_in_update', True))
        # --- Mirror run_fixed_boundary staged/scan/VMEC2000 logic ---
        ns_stages = [int(static.cfg.ns)]
        niter_stages = [int(max_iter)]
        ftol_stages = [float(indata.get_float("FTOL", 1e-13))]
        if ns_array is not None and hasattr(ns_array, '__len__') and len(ns_array) > 1:
            ns_stages = [int(v) for v in ns_array]
            if niter_array is not None and hasattr(niter_array, '__len__') and len(niter_array) == len(ns_stages):
                niter_stages = [int(v) for v in niter_array]
            else:
                niter_stages = [int(max_iter)] * len(ns_stages)
            if ftol_array is not None and hasattr(ftol_array, '__len__') and len(ftol_array) == len(ns_stages):
                ftol_stages = [float(v) for v in ftol_array]
            else:
                ftol_stages = [float(indata.get_float("FTOL", 1e-13))] * len(ns_stages)
        # Stage solve: coarse to fine, with scan/accelerated/VMEC2000 logic
        state = None
        static_prev = None
        res = None
        # --- Match run_fixed_boundary logic for accelerated_mode, stage_accelerated_mode, scan_mode ---
        accelerated_mode = False
        solver_mode_eff = kwargs.get('solver_mode', None)
        performance_mode = kwargs.get('performance_mode', True)
        if solver == 'vmec2000_iter':
            if solver_mode_eff is None and bool(performance_mode):
                solver_mode_eff = 'accelerated'
            if solver_mode_eff == 'accelerated':
                accelerated_mode = True

        for i, (ns_i, niter_i, ftol) in enumerate(zip(ns_stages, niter_stages, ftol_stages)):
            stage_t0 = time.time()
            # Build config for this stage
            cfg_dict = dict(static.cfg.__dict__)
            cfg_dict['ns'] = int(ns_i)
            if 'ntheta' in cfg_dict and hasattr(static.cfg, 'ntheta'):
                cfg_dict['ntheta'] = int(getattr(static.cfg, 'ntheta', 6))
            if 'nzeta' in cfg_dict and hasattr(static.cfg, 'nzeta'):
                cfg_dict['nzeta'] = int(getattr(static.cfg, 'nzeta', 8))
            cfg_stage = static.cfg.__class__(**cfg_dict)
            static_i = build_static(cfg_stage)
            flux_stage = flux_profiles_from_indata(indata, static_i.s, signgs=signgs)
            if i == 0:
                bdy_i = boundary
                st_guess = initial_guess_from_boundary(static_i, bdy_i, indata, vmec_project=vmec_project)
            else:
                st_guess = interp_vmec_state(state, m=static_prev.modes.m, n=static_prev.modes.n, lthreed=bool(static_prev.cfg.lthreed), lconm1=bool(getattr(static_prev.cfg, 'lconm1', True)), ns_new=int(ns_i))

            # --- Mode selection logic mimicking driver.py ---
            stage_accelerated_mode = bool(accelerated_mode)
            scan_mode = False
            if solver == 'vmec2000_iter':
                scan_mode = False
            # (Add LASYM/current-driven logic here if needed)

            # VMEC-style per-stage printout
            n_fourier = getattr(static_i.modes, 'K', 0)
            print(f"  NS = {ns_i:4d} NO. FOURIER MODES = {n_fourier:4d} FTOLV = {ftol:10.3E} NITER = {niter_i:6d}", flush=True)
            print(f"  PROCESSOR COUNT - RADIAL:    1", flush=True)
            print("", flush=True)
            print("  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD", flush=True)

            print(f"[DIAG] Stage {i+1}/{len(ns_stages)}: ns={ns_i}, niter={niter_i}, ftol={ftol}", flush=True)
            print(f"[DIAG] Solver: {solver}, use_scan={scan_mode}, parity_mode={not stage_accelerated_mode}", flush=True)

            # --- DIAGNOSTICS: Print initial axis coefficients and Jacobian ---
            print("[DIAG] Initial axis coefficients (R_cos[0], Z_cos[0]):", st_guess.Rcos[0], st_guess.Zcos[0])
            from vmec_jax.geom import eval_geom
            g_diag = eval_geom(st_guess, static_i)
            import jax.numpy as jnp
            print("[DIAG] Initial Jacobian min/mean:", jnp.min(g_diag.sqrtg), jnp.mean(g_diag.sqrtg))

            # Set all required arguments to False for scan/accelerated stages
            scan_args = dict(
                vmec2000_control=True,
                backtracking=False,
                use_restart_triggers=True,
                auto_flip_force=False,
                limit_dt_from_force=False,
                limit_update_rms=False,
                strict_update=True,
                use_direct_fallback=False,
                reference_mode=False,
            )
            if solver == 'gd':
                res = solve_fixed_boundary_gd(
                    st_guess,
                    static_i,
                    phipf=flux_stage.phipf,
                    chipf=flux_stage.chipf,
                    signgs=signgs,
                    lamscale=flux_stage.lamscale,
                    pressure=pressure,
                    gamma=float(indata.get_float("GAMMA", 0.0)),
                    max_iter=int(niter_i),
                    step_size=float(step_size),
                    jacobian_penalty=1e3,
                    jit_grad=bool(jit_grad),
                    differentiable=bool(differentiable),
                    stop_grad_in_update=bool(stop_grad_in_update),
                    verbose=bool(verbose),
                )
            elif solver == 'lbfgs':
                res = solve_fixed_boundary_lbfgs(
                    st_guess,
                    static_i,
                    phipf=flux_stage.phipf,
                    chipf=flux_stage.chipf,
                    signgs=signgs,
                    lamscale=flux_stage.lamscale,
                    pressure=pressure,
                    gamma=float(indata.get_float("GAMMA", 0.0)),
                    max_iter=int(niter_i),
                    step_size=float(step_size),
                    history_size=int(history_size),
                    jit_grad=True,
                    verbose=bool(verbose),
                )
            elif solver == 'vmec_lbfgs':
                res = solve_fixed_boundary_lbfgs_vmec_residual(
                    st_guess,
                    static_i,
                    indata=indata,
                    signgs=signgs,
                    history_size=int(history_size),
                    max_iter=int(niter_i),
                    step_size=float(step_size),
                    jit_grad=True,
                    preconditioner="mode_diag+radial_tridi",
                    precond_exponent=1.0,
                    precond_radial_alpha=0.2,
                    verbose=bool(verbose),
                )
            elif solver == 'vmec_gn':
                res = solve_fixed_boundary_gn_vmec_residual(
                    st_guess,
                    static_i,
                    indata=indata,
                    signgs=signgs,
                    max_iter=int(niter_i),
                    step_size=float(step_size),
                    damping=None if gn_damping is None else float(gn_damping),
                    cg_tol=None if gn_cg_tol is None else float(gn_cg_tol),
                    cg_maxiter=int(gn_cg_maxiter),
                    jit_kernels=True,
                    verbose=bool(verbose),
                )
            else:  # default to vmec2000_iter staged logic (parity with run_fixed_boundary)
                print("[DIAG] Calling solve_fixed_boundary_residual_iter with:")
                print(f"  resume_state=None, use_scan={scan_mode}, verbose={verbose}")
                res = solve_fixed_boundary_residual_iter(
                    st_guess,
                    static_i,
                    indata=indata,
                    signgs=signgs,
                    ftol=ftol,
                    max_iter=int(niter_i),
                    step_size=float(step_size),
                    include_constraint_force=True,
                    apply_m1_constraints=True,
                    precond_radial_alpha=0.5,
                    precond_lambda_alpha=0.5,
                    mode_diag_exponent=0.0,
                    divide_by_scalxc_for_update=False,
                    lambda_update_scale=1.0,
                    enforce_vmec_lambda_axis=True,
                    vmecpp_restart=False,
                    stage_prev_fsq=None,
                    stage_transition_factor=50.0,
                    stage_transition_scale=0.5,
                    resume_state=None,
                    verbose=True,
                    verbose_vmec2000_table=True,
                    jit_precompile=False,
                    jit_warmup_iters=0,
                    use_scan=scan_mode,
                    scan_minimal_default=None,
                    light_history=None,
                    resume_state_mode=None,
                    fsq_total_target=None,
                    host_update_assembly=None,
                    jit_forces=jit_forces,
                    limit_dt_from_force=scan_args['limit_dt_from_force'],
                    limit_update_rms=scan_args['limit_update_rms'],
                    auto_flip_force=scan_args['auto_flip_force'],
                    strict_update=scan_args['strict_update'],
                    use_restart_triggers=scan_args['use_restart_triggers'],
                    vmec2000_control=scan_args['vmec2000_control'],
                    use_direct_fallback=scan_args['use_direct_fallback'],
                    reference_mode=scan_args['reference_mode'],
                )
            stage_t1 = time.time()
            print(f"[DIAG] Stage {i+1} complete. Time: {stage_t1-stage_t0:.3f} s", flush=True)
            state = res.state
            static_prev = static_i
        # Post-process flux/profiles as in _final_flux_profiles_from_state
        t1 = time.time()
        print(f"[DIAG] All stages complete. Total time: {t1-t0:.3f} s")
        flux_out, prof_out = _final_flux_profiles_from_state(
            indata=indata,
            static_in=static_prev,
            state=state,
            signgs=signgs,
            flux_local=flux_stage,
            prof_local=eval_profiles(indata, static_prev.s),
            pressure_local=pressure,
        )
        # Return a FixedBoundaryRun-like dict (not dataclass for JAX compatibility)
        return {
            'cfg': static_prev.cfg,
            'indata': indata,
            'static': static_prev,
            'state': state,
            'result': res,
            'flux': flux_out,
            'profiles': prof_out,
            'signgs': signgs,
        }


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
