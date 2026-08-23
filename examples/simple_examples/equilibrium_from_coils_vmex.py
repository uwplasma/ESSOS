# A VMEC equilibrium held by ESSOS coils, solved by vmex (pip install vmex).
# Coils go out as a tabulated Biot-Savart grid, the equilibrium comes back as
# a wout that essos.fields.Vmec reads: neither package imports the other.
import os
number_of_processors_to_use = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
import vmex as vj
from dataclasses import replace
from essos.coils import Coils
from essos.fields import BiotSavart, Vmec
from essos.dynamics import Tracing

# Input parameters
ns = 16                      # radial surfaces; raise for a production run
phiedge = -0.025             # toroidal flux these coil currents hold [Wb]
mgrid_bounds = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6)
mgrid_shape = dict(ir=96, jz=96, kp=32)   # NZETA below must divide kp
s_fieldline = 0.5
nfieldlines = 4
tmax_fieldline = 800.        # about 120 toroidal turns
num_steps = 3000
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"   # the switch vmex's examples read

input_dir = os.path.join(os.path.dirname(__file__), '..', 'input_files')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)

# Load coils and build the field
coils = Coils.from_json(os.path.join(input_dir, 'ESSOS_biot_savart_LandremanPaulQA.json'))
biot_savart = jit(vmap(BiotSavart(coils).B))


# vmex tabulates a Cartesian field onto a cylindrical grid; its adapter calls
# B once per point, so vmap it and chunk to bound the coil-segment intermediate
def coil_field(points):
    return np.concatenate([np.asarray(biot_savart(chunk))
                           for chunk in np.array_split(np.asarray(points), 64)])


time0 = time()
external_field = vj.MgridField.from_cartesian_field(
    coil_field, nfp=int(coils.nfp), **mgrid_bounds, **mgrid_shape)
print(f"Tabulating {np.asarray(coils.currents).size} filaments took {time()-time0:.2f} seconds")

# The QA deck ESSOS ships is reactor scale; scale_input maps it onto the unit
# size these coils were optimized for.  It only seeds the guess: with LFREEB
# the last closed surface is solved for, not prescribed.
seed = vj.VmecInput.from_file(os.path.join(input_dir, 'input.LandremanPaul2021_QA_reactorScale_lowres'))
seed = vj.scale_input(seed, r_scale=1./float(seed.rbc[seed.ntor, 0]))
seed = replace(seed, lfreeb=True, mgrid_file='essos_coils', nzeta=16, phiedge=phiedge,
               ns_array=[ns], niter_array=[4000], ftol_array=[1e-10])

# Solve the free boundary in vmex
time0 = time()
result = vj.solve_free_boundary(seed, external_field=external_field, error_on_no_convergence=False)
wout = vj.wout_from_state(inp=seed, state=result.state, fsqr=float(result.fsqr),
                          fsqz=float(result.fsqz), fsql=float(result.fsql),
                          niter=int(result.iterations), converged=bool(result.converged),
                          vacuum_output=result.vacuum)
print(f"vmex free-boundary solve took {time()-time0:.2f} seconds, converged: {bool(result.converged)}")
print(f"Aspect ratio {float(wout.aspect):.3f}, |B| on axis {abs(float(wout.b0)):.4f} T")

# Read it back as an ESSOS field.  Released vmex hands the equilibrium over as
# a wout file, which is also what essos.fields.Vmec reads.
wout_file = os.path.join(output_dir, 'wout_essos_coils_QA.nc')
vj.write_wout(wout_file, wout)
vmec = Vmec(wout_file)

# Trace field lines: in flux coordinates ds/dt vanishes, so iota is the slope
# of theta against phi.  vmex computed iotaf independently from force balance.
initial_conditions = jnp.array([[s_fieldline]*nfieldlines,
                                jnp.linspace(0, 2*jnp.pi, nfieldlines, endpoint=False),
                                jnp.zeros(nfieldlines)]).T
time0 = time()
tracing = Tracing(field=vmec, model='FieldLine', initial_conditions=initial_conditions,
                  maxtime=tmax_fieldline, timestep=1e-2, times_to_trace=num_steps)
print(f"ESSOS field-line tracing took {time()-time0:.2f} seconds")
trajectories = np.asarray(tracing.trajectories)
iota_essos = np.mean([np.polyfit(trajectories[i, :, 2], trajectories[i, :, 1], 1)[0]
                      for i in range(nfieldlines)])
iota_vmex = float(np.interp(s_fieldline, np.linspace(0, 1, int(wout.ns)), np.asarray(wout.iotaf)))
print(f"iota at s={s_fieldline}: {iota_essos:.6f} traced, {iota_vmex:.6f} from vmex")

# Plot the coils, the solved boundary and the traced field lines
if not CI:
    fig = plt.figure(figsize=(9, 4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    coils.plot(ax=ax1, show=False)
    vmec.surface.plot(ax=ax1, show=False, alpha=0.4)
    tracing.plot(ax=ax1, show=False)
    # poincare_plot reads trajectories as XYZ; the traced ones are (s, theta, phi)
    tracing.trajectories = vmap(vmap(vmec.to_xyz, in_axes=0), in_axes=0)(trajectories)
    tracing.poincare_plot(ax=ax2, show=False, shifts=[0])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equilibrium_from_coils_vmex.png'))
    plt.show()
