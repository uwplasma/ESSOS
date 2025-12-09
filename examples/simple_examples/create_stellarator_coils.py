from essos.coils import Coils, CreateEquallySpacedCurves

# Initialize coils
current_on_each_coil = 1e5
number_of_field_periods = 2
number_coils_per_half_field_period = 4
major_radius_coils = 1.0
minor_radius_coils = 0.3
order_Fourier_series_coils = 1
number_coil_points = 50
curves = CreateEquallySpacedCurves(n_curves=number_coils_per_half_field_period,
                                   order=order_Fourier_series_coils,
                                   R=major_radius_coils, r=minor_radius_coils,
                                   n_segments=number_coil_points,
                                   nfp=number_of_field_periods, stellsym=True)
coils = Coils(curves=curves, currents=[current_on_each_coil]*number_coils_per_half_field_period)

# Change the parameters of a given coil
print("Coils before changing parameters:")
print(coils.x)

coils.x = coils.x.at[0].set(1.0)
coils.x = coils.x.at[1].set(0.4)
coils.x = coils.x.at[2].set(0.3)

print("Coils after changing parameters:")
print(coils.x)

# Plot the result
coils.plot()

# # Save the coils to a json file
# coils.to_json("stellarator_coils.json")
# # Load the coils from a json file
# from essos.coils import Coils_from_json
# coils = Coils_from_json("stellarator_coils.json")

# # View coils in Paraview
# coils.to_vtk('stellarator_coils')