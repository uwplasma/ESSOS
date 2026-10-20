from essos.surfaces import SurfaceRZFourier
from essos.coils import CreateCoilsAroundAxis
import os

input_filepath = '../input_files'
vmec_input = os.path.join(input_filepath, 'wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc')
surface = SurfaceRZFourier.from_wout_file(vmec_input, s=1, ntheta=60, nphi=60, range_torus='full torus')

print(f'Surface NFP: {surface.nfp}')

# Check what the script says
N_COILS = 4
NFP = 3  # From the script
STELLSYM = True

expected_from_script = N_COILS * NFP * (2 if STELLSYM else 1)
expected_from_surface = N_COILS * surface.nfp * (2 if STELLSYM else 1)

print(f'Expected coils with NFP={NFP}, stellsym={STELLSYM}: {expected_from_script}')
print(f'Expected coils with surface.nfp={surface.nfp}, stellsym={STELLSYM}: {expected_from_surface}')

# Actually create the curves and see what we get
init_curves = CreateCoilsAroundAxis(n_coils=N_COILS, order=6, coil_radius=5.7, n_segments=60, nfp=NFP, stellsym=STELLSYM, surface=surface)

print(f'\nActual number of curves created: {len(init_curves.gamma)}')
print(f'Curves.nfp: {init_curves.nfp}')
print(f'Curves.stellsym: {init_curves.stellsym}')
print(f'Number of base curves: {init_curves.n_base_curves}')
