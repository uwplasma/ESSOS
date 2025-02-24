import jax.numpy as jnp
from jax import jit, lax, random
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController

q = 2*1.602176565e-19
m = 4*1.660538921e-27

@jit
def GuidingCenter(t:              float,
                  inital_values:  jnp.ndarray,
                  args:           tuple) -> jnp.ndarray:
    field, mu = args

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (4,), "initial values must have shape (4,) with x, y, z, vpar"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z, vpar = inital_values
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def dxdt_dvdt(_):
        points = jnp.array([x, y, z])

        B_covariant = field.B_covariant(points)
        B_contravariant = field.B_contravariant(points)
        AbsB = field.AbsB(points)
        gradB = field.dAbsB_by_dX(points)

        omega = q*AbsB/m

        dxdt = vpar*B_contravariant/AbsB + (vpar**2/omega+mu/q)*jnp.cross(B_covariant, gradB)/AbsB/AbsB
        dvdt = -mu/m*jnp.dot(B_contravariant,gradB)/AbsB

        return jnp.append(dxdt,dvdt)

    def zero_derivatives(_):
        return jnp.zeros(4, dtype=float)

    return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@jit
def Lorentz(inital_values: jnp.ndarray,
            t:             float,
            field:          tuple) -> jnp.ndarray:
    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (6,), "initial values must have shape (6,) with x, y, z, vx, vy, vz"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z, vx, vy, vz = inital_values
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def dxdt_dvdt(_):
        points = jnp.array([x, y, z])
        B_contravariant = field.B_contravariant(points)

        dxdt = jnp.array([vx, vy, vz])
        dvdt = q / m * jnp.cross(dxdt, B_contravariant)

        return jnp.concatenate((dxdt, dvdt))

    def zero_derivatives(_):
        return jnp.zeros(6, dtype=float)

    return lax.cond(condition, zero_derivatives, dxdt_dvdt, operand=None)

@jit
def FieldLine(t:             float,
              inital_values: jnp.ndarray,
              field) -> jnp.ndarray:

    assert isinstance(inital_values, jnp.ndarray), "initial values must be a jnp.ndarray"
    assert inital_values.shape == (3,), "initial values must have shape (3,) with x, y, z"
    assert inital_values.dtype == float, "initial values must be a float"

    x, y, z = inital_values
    key = random.PRNGKey(42)
    random_value = random.choice(key, jnp.array([-1.0, 1.0]))
    vpar = random_value*299792458 # speed of light
    condition = (jnp.sqrt(x**2 + y**2) > 10) | (jnp.abs(z) > 10)

    def compute_derivatives(_):
        position = jnp.array([x, y, z])
        B_contravariant = field.B_contravariant(position)
        AbsB = field.AbsB(position)
        dxdt = vpar*B_contravariant/AbsB
        return dxdt

    def zero_derivatives(_):
        return jnp.zeros(3, dtype=float)

    return lax.cond(condition, zero_derivatives, compute_derivatives, operand=None)

@partial(jit, static_argnums=(1, 3, 4, 5, 6, 7,8))
def trace_trajectories(
    maxtime: float = 1e-7,
    timesteps: int = 200,
    initial_conditions: jnp.array = None
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

    times = jnp.linspace(0, maxtime, timesteps)
    
    
    def compute_trajectory(particle_idx: jnp.ndarray,x_idx: jnp.ndarray,
                                y_idx: jnp.ndarray,z_idx: jnp.ndarray,vpar_idx: jnp.ndarray,vperp_idx: jnp.ndarray) -> jnp.ndarray:
        modB=field.norm_B(jnp.array((x_idx,y_idx,z_idx)))
        mu = m * vperp_idx**2 / (2 * modB)
        y0 = jnp.array((x_idx,y_idx,z_idx,vpar_idx))
        args=(field, mu)

        trajectory = diffeqsolve(
            ODETerm(GuidingCenter),
            t0=0.0,
            t1=maxtime,
            dt0=maxtime / timesteps,
            y0=y0,
            solver=Tsit5(),
            args=args,
            saveat=SaveAt(ts=times),
            throw=False,
            adjoint=adjoint,
            stepsize_controller = PIDController(pcoeff=0.3, icoeff=0.4, rtol=tol_step_size, atol=tol_step_size, dtmax=None,dtmin=None),
            max_steps=num_adaptative_steps
        ).ys

        return trajectory

    trajectories = vmap(aux_trajectory_device,in_axes=(0,0,0,0,0,0))(particles_indeces_part,x_part,y_part,z_part,vpar_part,vperp_part)

    return trajectories