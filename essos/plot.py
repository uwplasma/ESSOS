import numpy as np
from scipy.interpolate import  splrep, splev, PPoly
import jax.numpy as jnp
import matplotlib.pyplot as plt


def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def roots(x,y, shift = 0):
              
    '''
    Finds roots using scipy 
    
    x: 1D array of independent values, must be strictly increasing -- such as Time
    y: 1D array of dependent values -- such as X, Y, Z, B, or V 
    shift: option to shift y to find roots at a non-zero value 
    '''         
    interp = splrep(x, (y - shift), k=3)
              
    roots = PPoly.from_spline(interp)
              
    x_values = roots.roots(extrapolate=False)
             
    return x_values

def poincare_plot(essos_trace,shift = 0, orientation = 'torodial',figsize=(12,6),bounds = None, length = 1,):
    '''
    plot Poincare plots by using scipy to find the roots of an interpolation
    Can take particle trace or field lines

    shift: apply a linear shift to dependent data, default is zero 
    orientation: 'toroidal' find time values when torodial angle = shift [0,2pi]
                   'z' find time values where z coordinate  = shift 

    Plotting args:
    bounds: [xmin,xmax,ymin,ymax], default None to ignore
    length: a way to shorten data, 1 - plot full length, .1 - plot 1/10 of data length

    Notes:
    If the data seem ill-behaved, there may not be enough steps in the trace for a good interpolation 
    This will break if there are any Nan's

    To-Do:
    Format colorbar's
    issues with toroidal interpolation: jnp.arctan2(Y,X) % (2 * jnp.pi) causes distortion in interpolation near phi = 0 
    Maybe determine a lower limit on resolution needed per toroidal turn for "good" results"
    '''

    fig = plt.figure(figsize = (12,6),layout = 'tight') 
    
    trace =  essos_trace.trajectories
    T = essos_trace.times
    
    z_orbits = []
    poloidal_drift = []
    start_list = []
    
    for i in range(len(trace)): 
        try: # particle
            X,Y,Z,E = trace[i].T
            cbar ='time'
        
        except: # field line
            X,Y,Z = trace[i].T
            cbar ='surface'
 
        if orientation == 'torodial':
            # prep splines 
            R = jnp.sqrt(X**2 + Y**2)
            spR = splrep(T, R, k=3)
            spZ = splrep(T, Z, k=3)
    
            phi = jnp.arctan2(Y,X) % (2 * jnp.pi)
            T_slice = roots(T,phi, shift = shift)
            
            # there is a bug that always counts phi = 0 as a root?
            # temp fix
            T_slice = T_slice[1::2] 
            
            R_slice = splev(T_slice,spR)
            Z_slice = splev(T_slice,spZ)
            
            lenth_ = int(len(R_slice)*length) 
            if cbar =='time':
                hits = plt.scatter(R_slice[0:lenth_], Z_slice[0:lenth_],c = T_slice[0:lenth_],s = 5 )
            if cbar =='surface':
                hits = plt.scatter(R_slice[0:lenth_], Z_slice[0:lenth_],s = 5 )
            plt.xlabel('R',fontsize = 20)
            plt.ylabel('Z',fontsize = 20)
            plt.title(r'$\phi$ = {:.2f} $\pi$'.format(shift/jnp.pi),fontsize = 20)

        if orientation == 'z':
            # prep splines 
            spX = splrep(T, X, k=3)
            spY = splrep(T, Y, k=3)
            spZ = splrep(T, Z, k=3)

            T_slice = roots(T,Z, shift = shift)
            X_slice = splev(T_slice,spX)
            Y_slice = splev(T_slice,spY)
    
            lenth_ = int(len(X_slice)*length) 
            if cbar =='time':
                hits = plt.scatter(X_slice[0:lenth_], Y_slice[0:lenth_],c = T_slice[0:lenth_],s = 5 )
            if cbar =='surface':
                hits = plt.scatter(X_slice[0:lenth_], Y_slice[0:lenth_],s = 5 )

            plt.xlabel('X',fontsize = 20)
            plt.xlabel('Y',fontsize = 20)
            plt.title('Z = {:.2f}'.format(shift),fontsize = 20)
            

    if bounds is not None:
        plt.xlim(bounds[0],bounds[1])
        plt.ylim(bounds[2],bounds[3])
    else:
        plt.axis('equal')
            
    plt.grid()
    plt.show()