#!/usr/bin/env python3.11
import os
import numpy as np
from time import time
import booz_xform as bx
import plotly.graph_objects as go
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from simsopt.mhd import Vmec, Boozer
from essos.coils import fit_dofs_from_coils, Curves, Coils

input_dir = os.path.join(os.path.dirname(__file__), 'input_files')
output_dir = os.path.join(os.path.dirname(__file__), 'output_files')
os.makedirs(output_dir, exist_ok=True)

wout_filename = os.path.join(input_dir, 'wout_LandremanPaul2021_QA_reactorScale_lowres.nc')
boozmn_filename = os.path.join(output_dir, 'boozmn_LandremanPaul2021_QA_reactorScale_lowres.nc')

ncoils = 6

# if boozmn_filename.split('/')[-1] in os.listdir(output_dir):
#     print(f"File {boozmn_filename} already exists, skipping computation")
#     b = bx.Booz_xform()
#     b.read_boozmn(boozmn_filename)
# else:
print(f"Computing {boozmn_filename}")
vmec = Vmec(wout_filename, verbose=False)
b = Boozer(vmec, mpol=64, ntor=64, verbose=True)
b.register([1])
b.run()
b.bx.write_boozmn(boozmn_filename)
b = b.bx

# bx.wireplot(b1, refine=6, orig=True).show()

js = None
ntheta = 30
# nphi = 80
nphi   = ncoils * 2 * b.nfp
surf = True
orig = True

theta1D = np.linspace(0, 2 * np.pi, ntheta)
phi1D = np.linspace(0, 2 * np.pi, nphi)
varphi, theta = np.meshgrid(phi1D, theta1D)

R = np.zeros_like(theta)
Z = np.zeros_like(theta)
nu = np.zeros_like(theta)

for jmn in range(b.mnboz):
    m = b.xm_b[jmn]
    n = b.xn_b[jmn]
    angle = m * theta - n * varphi
    sinangle = np.sin(angle)
    cosangle = np.cos(angle)
    R += b.rmnc_b[jmn, js] * cosangle
    Z += b.zmns_b[jmn, js] * sinangle
    nu += b.numns_b[jmn, js] * sinangle

# Following the sign convention in the code, to convert from the
# Boozer toroidal angle to the standard toroidal angle, we
# *subtract* nu:
phi = varphi - nu
X = R * np.cos(phi)
Y = R * np.sin(phi)

# for i in phi_idx:
#     loop = jnp.stack([x_2D_coils[:, i], y_2D_coils[:, i], z_2D_coils[:, i]], axis=-1)  # (ntheta,3)
#     coils_gamma = coils_gamma.at[coil_i].set(loop)
#     coil_i += 1

time0 = time()
dofs, gamma_uni = fit_dofs_from_coils(coils_gamma[:ncoils], order=order, n_segments=n_segments, assume_uniform=True)
curves = Curves(dofs=dofs, n_segments=n_segments, nfp=nfp, stellsym=True)
coils = Coils(curves=curves, currents=[-current_on_each_coil]*(ncoils))
field_coils_DOFS = BiotSavart(coils)
print(f"Fitting coils took {time()-time0:.2f} seconds")



exit()

color = '#FF9999'
# Hack to get a uniform surface color:
colorscale = [[0, color], [1, color]]

Rsurf = R
Zsurf = Z
Xsurf = Rsurf * np.cos(phi)
Ysurf = Rsurf * np.sin(phi)
data = [go.Surface(x=Xsurf, y=Ysurf, z=Zsurf,
                    colorscale=colorscale,
                    opacity=0.5,
                    showscale=False, # Turns off colorbar
                    lighting={"specular": 0.3, "diffuse":0.9})]

line_width = 4

line_marker = dict(color='red', width=line_width)
index = 0
for i, j, k in zip(X, Y, Z):
    index += 1
    showlegend = True
    if index > 1:
        showlegend = False
    data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=showlegend, name="Boozer coordinates"))
index = 0
for i, j, k in zip(X.T, Y.T, Z.T):
    index += 1
    data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=False))

# js = b.compute_surfs[js]
# R = np.zeros_like(theta)
# Z = np.zeros_like(theta)
# phi = varphi
    
# for jmn in range(b.mnmax):
#     angle = b.xm[jmn] * theta - b.xn[jmn] * phi
#     sinangle = np.sin(angle)
#     cosangle = np.cos(angle)
#     R += b.rmnc[jmn, js] * cosangle
#     Z += b.zmns[jmn, js] * sinangle
            
# X = R * np.cos(phi)
# Y = R * np.sin(phi)
# line_marker = dict(color='black', width=line_width)
    
# index = 0
# for i, j, k in zip(X, Y, Z):
#     index += 1
#     showlegend = True
#     if index > 1:
#         showlegend = False
#     data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=showlegend, name="Original coordinates"))

# index = 0
# for i, j, k in zip(X.T, Y.T, Z.T):
#     index += 1
#     data.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker, showlegend=False))
        
fig = go.Figure(data=data)

# Turn off hover contours on the surface:
fig.update_traces(contours_x_highlight=False,
                contours_y_highlight=False,
                contours_z_highlight=False,
                selector={"type":"surface"})

# Make x, y, z coordinate scales equal, and turn off more hover stuff
fig.update_layout(scene={"aspectmode": "data",
                            "xaxis_showspikes": False,
                            "yaxis_showspikes": False,
                            "zaxis_showspikes": False,
                            "xaxis_visible": False,
                            "yaxis_visible": False,
                            "zaxis_visible": False},
                    hovermode=False,
                    margin={"l":0, "r":0, "t":25, "b":0},
                    title="Curves of constant poloidal or toroidal angle")

fig.show()