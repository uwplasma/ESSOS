import jax.numpy as jnp
from jax import jit, jacfwd
from functools import partial

class BiotSavart():
    def __init__(self, coils):
        self.coils = coils
        self.currents = coils.currents[0]
        self.gamma = coils.gamma
        self.gamma_dash = coils.gamma_dash
    
    @partial(jit, static_argnames=['self'])
    def B(self, points):
        dif_R = (points-self.gamma).T
        dB = jnp.cross(self.gamma_dash.T, dif_R, axisa=0, axisb=0, axisc=0)/jnp.linalg.norm(dif_R, axis=0)**3
        dB_sum = jnp.einsum("i,bai", self.currents*1e-7, dB, optimize="greedy")
        return jnp.mean(dB_sum, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def AbsB(self, points):
        return jnp.linalg.norm(self.B(points))
    
    @partial(jit, static_argnames=['self'])
    def dB_by_dX(self, points):
        return jacfwd(self.B)(points)

