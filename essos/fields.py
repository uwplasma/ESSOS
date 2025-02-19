# import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# import jax.scipy as jsp
from jax import jit, grad, jacfwd, vmap
from functools import partial

class MagneticField():
    def __add__(self, other):
        """Add two magnetic fields."""
        return MagneticFieldSum([self, other])

    def __mul__(self, other):
        """Multiply a field with a scalar."""
        return MagneticFieldMultiply(other, self)

    def __rmul__(self, other):
        """Multiply a field with a scalar."""
        return MagneticFieldMultiply(other, self)
    
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points), axis=-1)
    
    def GradAbsB(self, points):
        return grad(self.AbsB)(points)

    def to_vtk(self, filename, nr=10, nphi=10, nz=10, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5):
        """Export the field evaluated on a regular grid for visualisation with e.g. Paraview."""
        from pyevtk.hl import gridToVTK
        rs = jnp.linspace(rmin, rmax, nr, endpoint=True)
        phis = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=True)
        zs = jnp.linspace(zmin, zmax, nz, endpoint=True)

        R, Phi, Z = jnp.meshgrid(rs, phis, zs)
        X = R * jnp.cos(Phi)
        Y = R * jnp.sin(Phi)
        Z = Z

        RPhiZ = jnp.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()
        RPhiZ[:, 2] = Z.flatten()

        self.set_points_cyl(RPhiZ)
        vals = self.B().reshape((R.shape[0], R.shape[1], R.shape[2], 3))
        contig = jnp.ascontiguousarray
        gridToVTK(filename, X, Y, Z, pointData={"B": (contig(vals[..., 0]), contig(vals[..., 1]), contig(vals[..., 2]))})


class MagneticFieldMultiply(MagneticField):
    def __init__(self, scalar, Bfield):
        self.scalar = scalar
        self.Bfield = Bfield

    def _B(self, B):
        B[:] = self.scalar*self.Bfield.B()

    def _dB_by_dX(self, dB):
        dB[:] = self.scalar*self.Bfield.dB_by_dX()

    def _d2B_by_dXdX(self, ddB):
        ddB[:] = self.scalar*self.Bfield.d2B_by_dXdX()

class MagneticFieldSum(MagneticField):
    def __init__(self, Bfields):
        self.Bfields = Bfields

    def _B(self, B):
        B[:] = jnp.sum([bf.B() for bf in self.Bfields], axis=0)

    def _dB_by_dX(self, dB):
        dB[:] = jnp.sum([bf.dB_by_dX() for bf in self.Bfields], axis=0)

    def _d2B_by_dXdX(self, ddB):
        ddB[:] = jnp.sum([bf.d2B_by_dXdX() for bf in self.Bfields], axis=0)

class BiotSavart(MagneticField):
    @partial(jit, static_argnames=['self'])
    def B(self, points, coils):
        print(f"gamma shape {coils.gamma.shape}")
        print(f"gamma_dash shape {coils.gamma_dash.shape}")
        print(f"currents shape {coils.currents.shape}")
        dif_R = (points-coils.gamma).T
        dB = jnp.cross(coils.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", coils.currents[0]*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, dB):
        points = self.get_points_cart()
        return jacfwd(self._B)()

