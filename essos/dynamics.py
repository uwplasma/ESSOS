import jax.numpy as jnp
from jax.scipy.optimize import bisect
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
    def __init__(self, field, model):
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
        if times is None:
            times = jnp.linspace(0, maxtime, timesteps)
        self.times=times
        
        def compute_trajectory(initial_condition) -> jnp.ndarray:

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
        from pyevtk.hl import polyLinesToVTK
        import numpy as np
        x = np.concatenate([xyz[:, 0] for xyz in trajectories])
        y = np.concatenate([xyz[:, 1] for xyz in trajectories])
        z = np.concatenate([xyz[:, 2] for xyz in trajectories])
        ppl = np.array([trajectories.shape[1]]*trajectories.shape[0])
        data = np.array(jnp.concatenate([i*jnp.ones((trajectories[i].shape[0], )) for i in range(len(trajectories))]))
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})
    
    def get_phi(self, x, y, phi_last):
        """Compute the toroidal angle phi with continuity correction."""
        phi = jnp.arctan2(y, x)
        phi += (phi_last - phi + jnp.pi) // (2 * jnp.pi) * (2 * jnp.pi)  # Ensure continuity
        return phi

    @partial(jit, static_argnums=(0))
    def poincare(self, trajectories, phis_poincare):
        """
        Compute the points where trajectories cross the given Poincaré sections.
        
        Parameters:
        - trajectories: jnp.ndarray of shape (num_particles, num_time_steps, 3) containing (x, y, z).
        - phis_poincare: List of phi values where intersections are checked.

        Returns:
        - res_phi_hits: jnp.ndarray containing (t, phi_index, x, y, z) at intersection points.
        """
        num_particles, num_steps, _ = trajectories.shape
        res_phi_hits = []

        for n in range(num_particles):
            traj = trajectories[n]
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            phi_last = self.get_phi(x[0], y[0], 0.0)

            for t_idx in range(1, num_steps):
                phi_current = self.get_phi(x[t_idx], y[t_idx], phi_last)
                t_last, t_current = t_idx - 1, t_idx  # Discrete time indices

                for i, phi_target in enumerate(phis_poincare):
                    if jnp.floor((phi_last - phi_target) / (2 * jnp.pi)) != jnp.floor((phi_current - phi_target) / (2 * jnp.pi)):
                        # Root finding: interpolate t_root where phi crosses phi_target
                        rootfun = lambda t: self.get_phi(jnp.interp(t, [t_last, t_current], [x[t_last], x[t_current]]),
                                                    jnp.interp(t, [t_last, t_current], [y[t_last], y[t_current]]),
                                                    phi_last) - phi_target
                        
                        t_root = bisect(rootfun, t_last, t_current)
                        x_root = jnp.interp(t_root, [t_last, t_current], [x[t_last], x[t_current]])
                        y_root = jnp.interp(t_root, [t_last, t_current], [y[t_last], y[t_current]])
                        z_root = jnp.interp(t_root, [t_last, t_current], [z[t_last], z[t_current]])

                        res_phi_hits.append(jnp.array([t_root, i, x_root, y_root, z_root]))

                phi_last = phi_current

        return jnp.array(res_phi_hits)

    def poincare_plot(self, filename, res_phi_hits=None, phis=None, mark_lost=False, aspect='equal', dpi=300, xlims=None, 
                        ylims=None, s=2, marker='o'):
        if res_phi_hits is None:
            res_phi_hits = self.poincare(self.trajectories, phis)
        import matplotlib.pyplot as plt
        from math import ceil, sqrt
        nrowcol = ceil(sqrt(len(phis)))
        plt.figure()
        fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
        for ax in axs.ravel():
            ax.set_aspect(aspect)
        color = None
        for i in range(len(phis)):
            row = i//nrowcol
            col = i % nrowcol
            if i != len(phis) - 1:
                if phis is not None: axs[row, col].set_title(f"$\\phi = {phis[i]/jnp.pi:.2f}\\pi$ ", loc='left', y=0.0)
            else:
                if phis is not None: axs[row, col].set_title(f"$\\phi = {phis[i]/jnp.pi:.2f}\\pi$ ", loc='right', y=0.0)
            if row == nrowcol - 1:
                axs[row, col].set_xlabel("$r$")
            if col == 0:
                axs[row, col].set_ylabel("$z$")
            if col == 1:
                axs[row, col].set_yticklabels([])
            if xlims is not None:
                axs[row, col].set_xlim(xlims)
            if ylims is not None:
                axs[row, col].set_ylim(ylims)
            for j in range(len(fieldlines_phi_hits)):
                lost = fieldlines_phi_hits[j][-1, 1] < 0
                if mark_lost:
                    color = 'r' if lost else 'g'
                data_this_phi = fieldlines_phi_hits[j][jnp.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
                if data_this_phi.size == 0:
                    continue
                r = jnp.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
                axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)

            plt.rc('axes', axisbelow=True)
            axs[row, col].grid(True, linewidth=0.5)

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()