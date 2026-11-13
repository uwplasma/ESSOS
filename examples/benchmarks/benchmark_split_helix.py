"""Prescribed helical field and exact collisionless full-B-star GC flow."""
import jax.numpy as jnp
from essos.constants import PROTON_MASS as M,ELEMENTARY_CHARGE as Q


class HelicalField:
    """B/B0=(sqrt(1-c*c)cos(kz), sqrt(1-c*c)sin(kz), c)."""
    def __init__(self, magnitude, wave, c=.5):
        self.magnitude, self.wave, self.c = magnitude, wave, c

    def gc_quantities(self, point):
        phase = self.wave*point[2]; a = jnp.sqrt(1-self.c**2)
        x, y = a*jnp.cos(phase), a*jnp.sin(phase)
        B = self.magnitude*jnp.array([x, y, self.c])
        return B, B, self.magnitude, jnp.zeros(3), -self.wave*jnp.array([x, y, 0.]), self.wave*self.c*jnp.array([-y, x, 0.]), 1.

    def B_contravariant(self, point): return self.gc_quantities(point)[0]
    B_covariant = B_contravariant
    def AbsB(self, point): return self.magnitude
    def dAbsB_by_dX(self, point): return jnp.zeros(3)
    def curl_b(self, point): return self.gc_quantities(point)[4]
    def kappa(self, point): return self.gc_quantities(point)[5]
    def sqrtg(self, point): return 1.

def exact_orbit(y, h, field):
    """Exact full B-star GC flow, including the v_parallel=0 limit."""
    omega = Q*field.magnitude/M; k, c = field.wave, field.c
    z = y[2]+h*y[3]*c/(1-(1-c*c)*k*y[3]/omega)
    amplitude = jnp.sqrt(1-c*c)/c*(1-k*y[3]/omega)/k
    return y.at[0].add(amplitude*(jnp.sin(k*z)-jnp.sin(k*y[2]))).at[1].add(
        amplitude*(jnp.cos(k*y[2])-jnp.cos(k*z))).at[2].set(z)
