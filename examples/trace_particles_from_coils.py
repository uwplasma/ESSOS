import os
os.mkdir("output") if not os.path.exists("output") else None
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32'
current_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(1, os.path.join(current_path,'..'))
import jax
import jax.numpy as jnp
#Which devices are being used by Jax
print("JAX running on", len(jax.devices()), jax.devices()[0].platform.upper())
from ESSOS import Curves, Coils, Particles, projection2D, projection2D_top,Plot_3D_trajectories
from MagneticField import norm_B
import matplotlib.pyplot as plt
from time import time
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import particles_to_vtk
from simsopt import load as simsopt_load
import numpy as np

coils_file = os.path.join(current_path,'inputs','biot_savart_opt.json')
stellarator_symmetry = True
text='LandremanPaulQA' #Name in the output files
r_init = 0.0775  #distance of the toroidal shell in which to initialize the particles
loss_radius_LCFS=6.  # Radius defined where particles are lost

bs=simsopt_load(coils_file)

# Currently using the number of cores set in line 3 (device_count)
n_particles=len(jax.devices()*2)

#These parameters are for the coils
n_curves=sum(['Current' in name for name in bs.full_dof_names]) #Number of unique coils in the input coilset
order=bs.coils[0].curve.order    # order used in the Fourier decomposition of each coil parametrization
nfp = int(len(bs.coils)/n_curves/(1+stellarator_symmetry))    #number of field periods assumed for the coils

B_target_on_axis=5.7 #Intended B on axis for scalled coils
major_radius_from_coils = np.mean([np.sqrt(coil.curve.x[coil.curve.local_dof_names.index('xc(0)')]**2+coil.curve.x[coil.curve.local_dof_names.index('yc(0)')]**2) for coil in bs.coils[:n_curves]])
R = major_radius_from_coils # Major Radius of the device, for now particles are being uniformly initialized in a toroidal shell of radius r_init, centered at major radius R

minor_radius_from_coils = np.mean([np.sqrt(
    coil.curve.x[coil.curve.local_dof_names.index('xc(1)')]**2+
    coil.curve.x[coil.curve.local_dof_names.index('yc(1)')]**2+
    coil.curve.x[coil.curve.local_dof_names.index('zc(1)')]**2+
    coil.curve.x[coil.curve.local_dof_names.index('xs(1)')]**2+
    coil.curve.x[coil.curve.local_dof_names.index('ys(1)')]**2+
    coil.curve.x[coil.curve.local_dof_names.index('zs(1)')]**2
) for coil in bs.coils[:4]])/2
r=minor_radius_from_coils  # original itargeted minor radius of the coils
axis_rc_zs = np.array([[1, 0.0], [0, 0.0]])*R #This changes the axis which  is used to center the initial toroidal shell for particle initialization
energy = 3.52e6  # Energy of the particles in eV
#Pitch angle initialization
more_trapped_particles=2 #If ==0 uniform distribution in pitch angle, if ==1 constraints the initial pitch angle to [-trapped_fraction_more ,trapped_fraction_more]
#If ==2 constraints the initial pitch angle to being =trapped_fraction_more for every particle
trapped_fraction_more=0.2

maxtime = 1.0e-4 #Final time for tracing in seconds
timesteps = int(maxtime*1.0e7)  #How many timesteps at which to save the particles' position´
num_adaptative_steps=100000 #maximum number of adaptative steps used at each timestep, if siimulation is taking too much time, it is better to keep this fixed and decrease tolerance below
#Tolerance used for the time solver, for the configuration QA used here, this tolerance may take a while to run for larger pitch angle values, decrease to 1.e-3 for quicker runs
tol_step_size=1.e-5     

model = "Guiding Center"  # Using guiding center equations. Use model = 'Lorentz' for full Lorentz orbits

n_segments = order*11  # Number of segments used to describe each coil

particles = Particles(n_particles) # Initialize particle object

dofs_list=[]
currents_list=[]
bs=simsopt_load(coils_file)
for coil in bs.coils:
    dofs_index=coil.curve.get_dofs()
    dofs_list.append(dofs_index)

for i in range(n_curves): 
    currents_list.append(bs.coils[i].current.get_dofs())

#Function for creating coils TODO update to simsopt discussion
dofs = jnp.reshape(jnp.array(dofs_list[0:n_curves]), (n_curves, 3, 2*order+1))
currents = jnp.reshape(jnp.array(currents_list), (n_curves))
curves = Curves(dofs, nfp=nfp, stellsym=True)
current_scaling_factor= 1.e-7*B_target_on_axis*2.*jnp.pi*r
stel = Coils(curves, current_scaling_factor*currents)

## Writing the shape of coils (n_curves, dimensions, number of Fourier coeficients used per dimension)
print(f"Dofs shape: {stel.dofs.shape}")

#Initializing particles uniformly at r_init
#more_trapped_particles=False => uniform distribution in pitch angle
#more traapped_particles=True =>
initial_values = stel.initial_conditions(particles, R, r_init, model=model, more_trapped_particles=more_trapped_particles, trapped_fraction_more=trapped_fraction_more, axis_rc_zs=axis_rc_zs , nfp=nfp)
initial_vperp = initial_values[4, :]

#Calculate field on axis once to correct for the missing lenght factor on the current target field
norm_B_correction = jnp.mean(jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents))
stel = Coils(curves, current_scaling_factor*currents*B_target_on_axis/norm_B_correction)

#Trace particles trajectories once
time0 = time()
trajectories = stel.trace_trajectories(particles, initial_values, maxtime=maxtime, timesteps=timesteps,tol_step_size=tol_step_size,num_adaptative_steps=num_adaptative_steps)
print("Trajectories shape:", trajectories.shape)
print(f"Time to trace trajectories: {time()-time0:.2f} seconds")
normB = jnp.apply_along_axis(norm_B, 0, initial_values[:3, :], stel.gamma, stel.gamma_dash, stel.currents)
print(f"Mean normB for all particles at t=0: {jnp.mean(normB):2f} T")

print(f"Creating plots...")

#Post processing function to save the axis
def save_axis(axis_rc_zs, nfp, file="output/axis"):
    phi_axis = jnp.linspace(0, 2 * jnp.pi, 100)
    i = jnp.arange(len(axis_rc_zs[0]))  # Index array
    cos_terms = jnp.cos(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[0]), 30)
    sin_terms = jnp.sin(i[:, None] * phi_axis * nfp)  # Shape: (len(axis_rc_zs[1]), 30)
    R_axis = jnp.sum(axis_rc_zs[0][:, None] * cos_terms, axis=0)  # Sum over `i` (first axis)
    Z_axis = jnp.sum(axis_rc_zs[1][:, None] * sin_terms, axis=0)  # Sum over `i` (first axis)
    pos_axis = jnp.array([R_axis*jnp.cos(phi_axis), R_axis*jnp.sin(phi_axis), Z_axis]).transpose(1, 0)
    particles_to_vtk(res_tys=jnp.array([jnp.concatenate([jnp.zeros((pos_axis.shape[0],1)), pos_axis], axis=1)]), filename=file)

#Create different plots regarding the trajectory of the particles of the trajectories, assumes circular coil shape of minor radius r and major radius R
def create_trajectory_plots(trajectories, text):
    projection2D(R, r+2., trajectories, show=False, save_as=f"output/pol_{text}.pdf", close=True)   #Poloidal projection
    projection2D_top(R, r, trajectories, show=False, save_as=f"output/tor_{text}.pdf", close=True) #Toroidal projection
    #Plot_3D_trajectories(R, r, trajectories, show=True, save_as=None, close=False)
    plt.figure()
    # for i in range(len(trajectories)):
    #     plt.plot(jnp.arange(jnp.size(trajectories, 1))*maxtime/timesteps, trajectories[i, :, 3])
    time_vals = jnp.arange(jnp.size(trajectories, 1)) * maxtime / timesteps
    plt.plot(time_vals, trajectories[:, :, 3].T)
    
    plt.title("Parallel Velocity")
    plt.xlabel("time [s]")
    plt.ylabel(r"parallel velocity [ms$^{-1}$]")
    y_limit = max(jnp.abs(jnp.max(trajectories[:, :, 3])), jnp.abs(jnp.min(trajectories[:, :, 3])))
    plt.ylim(-1.2*y_limit, 1.2*y_limit)
    plt.savefig(f"output/vpar_{text}.pdf", transparent=True)
    plt.close()

    mu = particles.mass*initial_vperp**2/(2*normB)

    y_limit = 0
    plt.figure()
    normB_fn = jax.vmap(norm_B, in_axes=(0, None, None, None))
    normB_batch = jax.vmap(normB_fn, in_axes=(0, None, None, None))(trajectories[:, :, :3], stel.gamma, stel.gamma_dash, stel.currents)
    normalized_energy = jnp.abs((mu[:, None] * normB_batch + 0.5 * particles.mass * trajectories[:, :, 3]**2) / particles.energy - 1)
    time_vals = jnp.arange(jnp.size(trajectories, 1)) * maxtime / timesteps
    plt.plot(time_vals, normalized_energy.T)
    plt.yscale('log');plt.title("Energy Conservation")
    plt.xlabel("time [s]");plt.ylabel(r"$\frac{E-E_\alpha}{E_\alpha}$")
    plt.savefig(f"output/energy_{text}.pdf", transparent=True)
    plt.close()

    stel.plot(trajectories=trajectories, title="Initial Stellator", save_as=f"output/3D_{text}.pdf", show=False)
    plt.close('all')

def create_simsopt_curves(curves):
    curves_simsopt = []
    for i, curve in enumerate(curves):
        curves_simsopt.append( CurveXYZFourier(100, order) )
        curves_simsopt[i].x = jnp.ravel(curve)
    return curves_simsopt

###########################################
#Post processing
create_trajectory_plots(trajectories, text=text)  
curves_to_vtk(create_simsopt_curves(stel._curves), f"output/curves_{text}", close=True)   #create .vtk file of coils
particles_to_vtk(res_tys=jnp.concatenate([trajectories[:, :, 3:4], trajectories[:, :, :3]], axis=2), filename=f"output/particles_{text}") #create .vtk file of particles

#Example of lost fraction calculation
x_final=trajectories[:,-1,0]  #trajectories dimensions are (n_particles,n_timesteps_saved,n_coordinates) 
y_final=trajectories[:,-1,1] 
z_final=trajectories[:,-1,2]
R_axis=R
Z_axis=0.

r_final = jnp.sqrt((jnp.sqrt(x_final**2 + y_final**2) - R_axis)**2 + (z_final - Z_axis)**2)
lost_fraction = jnp.mean((r_final > loss_radius_LCFS).astype(jnp.float32)) * 100.

print(f"Lost particle fraction at time {maxtime:2f}s, considering r lcfs {loss_radius_LCFS:2f} m: {lost_fraction:2f} %")
############################################################################################################
