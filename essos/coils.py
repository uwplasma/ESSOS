



class Curves:
    """
    Class to store the curves

    -----------
    Attributes:
        dofs (jnp.ndarray - shape (n_indcurves, 3, 2*order+1)): Fourier Coefficients of the independent curves
        n_segments (int): Number of segments to discretize the curves
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
        order (int): Order of the Fourier series
        curves jnp.ndarray - shape (n_indcurves*nfp*(1+stellsym), 3, 2*order+1)): Curves obtained by applying rotations and flipping corresponding to nfp fold rotational symmetry and optionally stellarator symmetry
        gamma (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves
        gamma_dash (jnp.array - shape (n_coils, n_segments, 3)): Discretized curves derivatives

    """
    def __init__(self, dofs: jnp.ndarray, n_segments: int = 100, nfp: int = 1, stellsym: bool = False):
        assert isinstance(dofs, jnp.ndarray), "dofs must be a jnp.ndarray"
        assert dofs.ndim == 3, "dofs must be a 3D array with shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[1] == 3, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert dofs.shape[2] % 2 == 1, "dofs must have shape (n_curves, 3, 2*order+1)"
        assert isinstance(n_segments, int), "n_segments must be an integer"
        assert n_segments > 2, "n_segments must be greater than 2"
        assert isinstance(nfp, int), "nfp must be a positive integer"
        assert nfp > 0, "nfp must be a positive integer"
        assert isinstance(stellsym, bool), "stellsym must be a boolean"
    
        self._dofs = dofs
        self._n_segments = n_segments
        self._nfp = nfp
        self._stellsym = stellsym
        self._order = dofs.shape[2]//2
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()

    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n"

    def _tree_flatten(self):
        children = (self._dofs,)  # arrays / dynamic values
        aux_data = {"n_segments": self._n_segments, "nfp": self._nfp, "stellsym": self._stellsym}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    def _set_gamma(self):
        """ Initializes the discritized curves and their derivatives"""

        # Create the quadpoints
        quadpoints = jnp.linspace(0, 1, self.n_segments, endpoint=False)
                        
        # Create the gamma and gamma_dash
        def fori_createdata(order_index: int, data: jnp.ndarray) -> jnp.ndarray:
            return data[0] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], jnp.sin(2 * jnp.pi * order_index * quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], jnp.cos(2 * jnp.pi * order_index * quadpoints)), \
                   data[1] + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index - 1], 2*jnp.pi*order_index*jnp.cos(2 * jnp.pi * order_index * quadpoints)) + jnp.einsum("ij,k->ikj", self._curves[:, :, 2 * order_index], -2*jnp.pi*order_index*jnp.sin(2 * jnp.pi * order_index * quadpoints))
        
        gamma = jnp.einsum("ij,k->ikj", self._curves[:, :, 0], jnp.ones(self.n_segments))
        gamma_dash = jnp.zeros((jnp.size(self._curves, 0), self.n_segments, 3))
        gamma, gamma_dash = fori_loop(1, self._order+1, fori_createdata, (gamma, gamma_dash)) 
        
        length = jnp.array([jnp.mean(jnp.linalg.norm(d1gamma, axis=1)) for d1gamma in gamma_dash])

        # Set the attributes
        self._gamma = gamma
        self._gamma_dash = gamma_dash
        self._length = length
    
    @property
    def dofs(self):
        return self._dofs
    
    @dofs.setter
    def dofs(self, new_dofs):
        assert isinstance(new_dofs, jnp.ndarray)
        assert new_dofs.ndim == 3
        assert jnp.size(new_dofs, 1) == 3
        assert jnp.size(new_dofs, 2) % 2 == 1
        self._dofs = new_dofs
        self._order = jnp.size(new_dofs, 2)//2
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()
    
    @property
    def curves(self):
        return self._curves
    
    @property 
    def order(self):
        return self._order
    
    @order.setter
    def order(self, new_order):
        assert isinstance(new_order, int)
        assert new_order > 0
        self._dofs = jnp.pad(self.dofs, ((0, 0), (0, 0), (0, 2*(new_order-self._order)))) if new_order > self._order else self.dofs[:, :, :2*(new_order)+1]
        self._order = new_order
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()

    @property
    def n_segments(self):
        return self._n_segments
    
    @n_segments.setter
    def n_segments(self, new_n_segments):
        assert isinstance(new_n_segments, int)
        assert new_n_segments > 2
        self._n_segments = new_n_segments
        self._set_gamma()
    
    @property
    def nfp(self):
        return self._nfp
    
    @nfp.setter
    def nfp(self, new_nfp):
        assert isinstance(new_nfp, int)
        assert new_nfp > 0
        self._nfp = new_nfp
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()
    
    @property
    def stellsym(self):
        return self._stellsym
    
    @stellsym.setter
    def stellsym(self, new_stellsym):
        assert isinstance(new_stellsym, bool)
        self._stellsym = new_stellsym
        self._curves = apply_symmetries_to_curves(self.dofs, self.nfp, self.stellsym)
        self._set_gamma()
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def gamma_dash(self):
        return self._gamma_dash
    
    @property
    def length(self):
        return self._length
    
    def save_curves(self, filename: str):
        """
        Save the curves to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")



tree_util.register_pytree_node(Curves,
                               Curves._tree_flatten,
                               Curves._tree_unflatten)

class Coils(Curves):
    def __init__(self, curves: Curves, dofs_currents: jnp.ndarray):
        assert isinstance(curves, Curves)
        assert isinstance(dofs_currents, jnp.ndarray)
        assert jnp.size(dofs_currents) == jnp.size(curves.dofs, 0)
        super().__init__(curves.dofs, curves.n_segments, curves.nfp, curves.stellsym)
        self._dofs_currents = dofs_currents
        self._currents = apply_symmetries_to_currents(self._dofs_currents, self.nfp, self.stellsym)

    def __str__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n"
                
    def __repr__(self):
        return f"nfp stellsym order\n{self.nfp} {self.stellsym} {self.order}\n"\
             + f"Degrees of freedom\n{repr(self.dofs.tolist())}\n" \
             + f"Currents degrees of freedom\n{repr(self.dofs_currents.tolist())}\n"
    
    @property
    def dofs_currents(self):
        return self._dofs_currents
    
    @dofs_currents.setter
    def dofs_currents(self, new_dofs_currents):
        self._dofs_currents = new_dofs_currents
        self._currents = apply_symmetries_to_currents(self._dofs_currents, self.nfp, self.stellsym)
    
    @property
    def currents(self):
        return self._currents
    
    def _tree_flatten(self):
        children = (Curves(self.dofs, self.n_segments, self.nfp, self.stellsym), self._dofs_currents)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    def save_coils(self, filename: str, text=""):
        """
        Save the coils to a file
        """
        with open(filename, "a") as file:
            file.write(f"nfp stellsym order\n")
            file.write(f"{self.nfp} {self.stellsym} {self.order}\n")
            file.write(f"Degrees of freedom\n")
            file.write(f"{repr(self.dofs.tolist())}\n")
            file.write(f"Currents degrees of freedom\n")
            file.write(f"{repr(self._dofs_currents.tolist())}\n")
            # file.write(f"Loss\n")
            file.write(f"{text}\n")


tree_util.register_pytree_node(Coils,
                               Coils._tree_flatten,
                               Coils._tree_unflatten)


def CreateEquallySpacedCurves(n_curves:   int,
                              order:      int,
                              R:          float,
                              r:          float,
                              n_segments: int = 100,
                              nfp:        int  = 1,
                              stellsym:   bool = False,
                              axis_rc_zs = None) -> jnp.ndarray:
    """ Create a toroidal set of cruves equally spaced with an outer radius R and inner radius r.
    Attributes:
        n_curves (int): Number of curves
        order (int): Order of the Fourier series
        R (float): Outer radius of the curves
        r (float): Inner radius of the curves
        nfp (int): Number of field periods
        stellsym (bool): Stellarator symmetry
    Returns:
        curves (Curves): Equally spaced curves
    """
    if axis_rc_zs is not None:
        angle_locations = 1/((1+int(stellsym))*nfp*n_curves)/2+np.linspace(0,1,(n_curves)*(1+int(stellsym))*nfp, endpoint=False)
        ma = CurveRZFourier(angle_locations, len(axis_rc_zs[0])-1, nfp, False)
        ma.rc[:] = axis_rc_zs[0]
        ma.zs[:] = axis_rc_zs[1, 1:]
        ma.x = ma.get_dofs()
        gamma_curves = ma.gamma()
        
        curves = jnp.zeros((n_curves, 3, 1+2*order))
        for i in range(n_curves):
            angle = (i+0.5)*(2*jnp.pi)/((1+int(stellsym))*nfp*n_curves)
            curves = curves.at[i, 0, 0].set(gamma_curves[i,0])
            curves = curves.at[i, 0, 2].set(jnp.cos(angle)*r)
            curves = curves.at[i, 1, 0].set(gamma_curves[i,1])
            curves = curves.at[i, 1, 2].set(jnp.sin(angle)*r)
            curves = curves.at[i, 2, 0].set(gamma_curves[i,2])
            curves = curves.at[i, 2, 1].set(-r)
    else:
        curves = jnp.zeros((n_curves, 3, 1+2*order))
        for i in range(n_curves):
            angle = (i+0.5)*(2*jnp.pi)/((1+int(stellsym))*nfp*n_curves)
            curves = curves.at[i, 0, 0].set(jnp.cos(angle)*R)
            curves = curves.at[i, 0, 2].set(jnp.cos(angle)*r)
            curves = curves.at[i, 1, 0].set(jnp.sin(angle)*R)
            curves = curves.at[i, 1, 2].set(jnp.sin(angle)*r)
            curves = curves.at[i, 2, 1].set(-r)
            # In the previous line, the minus sign is for consistency with
            # Vmec.external_current(), so the coils create a toroidal field of the
            # proper sign and free-boundary equilibrium works following stage-2 optimization.
    return Curves(curves, n_segments=n_segments, nfp=nfp, stellsym=stellsym)

def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    base_curves: shape - (n_indepentdent_curves, 3, 1+2*order)
    """
    """
    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.
    """

    flip_list = jnp.array([False, True]) if stellsym else jnp.array([False])

    fliped_base_curves = jnp.einsum("aic,ib->abc", base_curves, jnp.array([[1, 0, 0],
                                                                           [0, -1, 0],
                                                                           [0, 0, -1]]))

    if stellsym:
        curves = jnp.append(base_curves, fliped_base_curves, axis=0)
    else:
        curves = base_curves

    for fp in jnp.arange(1, nfp):
        for flip in flip_list:
            rotcurves = jnp.einsum("aic,ib->abc", base_curves, jnp.array([[jnp.cos(2*jnp.pi*fp/nfp), -jnp.sin(2*jnp.pi*fp/nfp), jnp.zeros_like(fp)],
                                                                          [jnp.sin(2*jnp.pi*fp/nfp), jnp.cos(2*jnp.pi*fp/nfp), jnp.zeros_like(fp)],
                                                                          [jnp.zeros_like(fp), jnp.zeros_like(fp), jnp.ones_like(fp)]]).T)
            rotcurves = select(flip, jnp.einsum("aic,ib->abc", rotcurves, jnp.array([[1, 0, 0],
                                                                                     [0, -1, 0],
                                                                                     [0, 0, -1]])), rotcurves)
            curves = jnp.append(curves, rotcurves, axis=0)
    return curves

def apply_symmetries_to_currents(base_currents, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`Current`s and return ``n * nfp * (1+int(stellsym))``
    :mod:`Current` objects obtained by copying (for ``nfp`` rotations) and
    sign-flipping (optionally for stellarator symmetry).
    """
    flip_list = jnp.array([False, True]) if stellsym else jnp.array([False])
    currents = jnp.array([])
    for _ in range(0, nfp):
        for flip in flip_list:
            current = select(flip, base_currents * -1, base_currents)
            currents = jnp.append(currents, current)
    return currents
