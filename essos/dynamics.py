import jax.numpy as jnp
from jax import jit, lax, random, vmap
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController

q = 2*1.602176565e-19
m = 4*1.660538921e-27
c = 299792458

@partial(jit, static_argnums=(2))
def GuidingCenter(t,
                  initial_condition,
                  field) -> jnp.ndarray:

    x, y, z, vpar = initial_condition
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])

    B_covariant = field.B_covariant(points)
    B_contravariant = field.B_contravariant(points)
    AbsB = field.AbsB(points)
    gradB = field.dAbsB_by_dX(points)

    omega = q*AbsB/m

    dxdt = vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB
    dvdt = -mu/m*jnp.dot(B_contravariant,gradB)/AbsB

    return jnp.append(dxdt,dvdt)

    # def zero_derivatives(_):
    #     return jnp.zeros(4, dtype=float)

    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@partial(jit, static_argnums=(2))
def Lorentz(t,
            initial_condition,
            field) -> jnp.ndarray:
    
    x, y, z, vx, vy, vz = initial_condition
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    # def dxdt_dvdt(_):
    points = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(points)

    dxdt = jnp.array([vx, vy, vz])
    dvdt = q / m * jnp.cross(dxdt, B_contravariant)

    return jnp.append((dxdt, dvdt))

    # def zero_derivatives(_):
    #     return jnp.zeros(6, dtype=float)

    # return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@partial(jit, static_argnums=(2))
def FieldLine(t,
              initial_condition,
              field) -> jnp.ndarray:

    # assert isinstance(initial_condition, jnp.ndarray), "initial values must be a jnp.ndarray"
    # assert initial_condition.shape == (3,), "initial values must have shape (3,) with x, y, z"
    # assert initial_condition.dtype == float, "initial values must be a float"

    x, y, z = initial_condition
    # velocity_signs = jnp.array([-1.0, 1.0])
    # plus1_minus1 = random.choice(random.PRNGKey(42), velocity_signs)
    # velocity = plus1_minus1*c # speed of light
    # condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    # def compute_derivatives(_):
    position = jnp.array([x, y, z])
    B_contravariant = field.B_contravariant(position)
    dxdt = B_contravariant
    return dxdt

    # def zero_derivatives(_):
    #     return jnp.zeros(3, dtype=float)

    # return lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

class Tracing():
    def __init__(self, field, model, times=None):
        self.field = field
        self.model = model

        if model == 'GuidingCenter':
            self.ODE_term = ODETerm(GuidingCenter)
        elif model == 'Lorentz':
            self.ODE_term = ODETerm(Lorentz)
        elif model == 'FieldLine':
            self.ODE_term = ODETerm(FieldLine)

    @partial(jit, static_argnums=(0, 2, 3))
    def trace(self,
        initial_conditions,
        maxtime: float = 1e-7,
        timesteps: int = 200,
        tol_step_size = 1e-7,
        times=None,
    ):
                        #    particles,
                        #    field,
                        #    initial_values: jnp.ndarray,
                        #    maxtime: float = 1e-7,
                        #    timesteps: int = 200,
                        #    n_cores: int = len(devices()),
                        #    adjoint=RecursiveCheckpointAdjoint(),
                        #    tol_step_size = 5e-5,
                        #    num_adaptative_steps=100000) -> jnp.ndarray:
        if times is None:
            times = jnp.linspace(0, maxtime, timesteps)
        self.times=times
        
        def compute_trajectory(initial_condition) -> jnp.ndarray:
            # modB=field.norm_B(jnp.array((x_idx,y_idx,z_idx)))
            # mu = m * vperp_idx**2 / (2 * modB)
            # y0 = jnp.array((x_idx,y_idx,z_idx,vpar_idx))
            # args=(field, mu)

            trajectory = diffeqsolve(
                self.ODE_term,
                t0=0.0,
                t1=maxtime,
                dt0=maxtime / timesteps,
                y0=initial_condition,
                solver=Tsit5(),
                args=self.field,
                saveat=SaveAt(ts=times),
                throw=False,
                # adjoint=adjoint,
                stepsize_controller = PIDController(pcoeff=0.3, icoeff=0.4, rtol=tol_step_size, atol=tol_step_size, dtmax=None,dtmin=None),
                # max_steps=num_adaptative_steps
            ).ys

            return trajectory

        trajectories = vmap(compute_trajectory,in_axes=(0))(initial_conditions)

        return trajectories
    
    def to_vtk(self, filename, trajectories):
        """
        Export particle tracing or field lines to a vtk file.
        Expects that the xyz positions can be obtained by ``xyz[:, 1:4]``.
        """
        from pyevtk.hl import polyLinesToVTK
        import numpy as np
        x = np.concatenate([xyz[:, 0] for xyz in trajectories])
        y = np.concatenate([xyz[:, 1] for xyz in trajectories])
        z = np.concatenate([xyz[:, 2] for xyz in trajectories])
        ppl = np.array([trajectories.shape[1]]*trajectories.shape[0])
        data = np.array(jnp.concatenate([i*jnp.ones((trajectories[i].shape[0], )) for i in range(len(trajectories))]))
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})