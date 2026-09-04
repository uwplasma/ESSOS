import jax.numpy as jnp
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import CombinedField, BiotSavart


def main():

    number_of_field_periods = 2
    number_coils_per_half_field_period = 4
    major_radius_coils = 1.0
    minor_radius_coils = 0.3
    order_Fourier_series_coils = 1
    number_coil_points = 50

    curves = CreateEquallySpacedCurves(
        n_curves=number_coils_per_half_field_period,
        order=order_Fourier_series_coils,
        R=major_radius_coils, r=minor_radius_coils,
        n_segments=number_coil_points,
        nfp=number_of_field_periods, stellsym=True,
    )
    coils = Coils(curves=curves, currents=[1e5]*number_coils_per_half_field_period)

    field = BiotSavart(coils)
    test_point = jnp.array([1.0, 0.0, 0.0])

    print("=== Single-field baseline ===")
    B_single = field.B(test_point)
    print(f"field.B(test_point) = {B_single}")

    print("\n=== CombinedField with ONE field (should equal single) ===")
    combined_one = CombinedField(field)
    B_combined_one = combined_one.B(test_point)
    print(f"CombinedField(field).B(test_point) = {B_combined_one}")
    print(f"Matches single field: {jnp.allclose(B_combined_one, B_single)}")

    print("\n=== CombinedField with the SAME field twice (should equal 2x) ===")
    combined_two = CombinedField(field, field)
    B_combined_two = combined_two.B(test_point)
    print(f"CombinedField(field, field).B(test_point) = {B_combined_two}")
    print(f"Matches 2x single field: {jnp.allclose(B_combined_two, 2*B_single)}")

    print("\n=== B_contravariant matches B for BiotSavart (Cartesian field) ===")
    Bc_combined = combined_two.B_contravariant(test_point)
    print(f"B_contravariant = {Bc_combined}")
    print(f"Matches B: {jnp.allclose(Bc_combined, B_combined_two)}")

    print("\n=== AbsB (inherited from MagneticField base class) ===")
    absB = combined_two.AbsB(test_point)
    absB_expected = jnp.linalg.norm(B_combined_two)
    print(f"AbsB = {absB}  expected = {absB_expected}  match = {jnp.allclose(absB, absB_expected)}")

    print("\n=== to_xyz (delegates to first field) ===")
    xyz = combined_two.to_xyz(test_point)
    print(f"to_xyz(test_point) = {xyz}  matches input (BiotSavart is Cartesian passthrough): {jnp.allclose(xyz, test_point)}")

    print("\n=== Empty CombinedField() should raise ValueError ===")
    try:
        CombinedField()
        print("ERROR: should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly raised: {e}")

    print("\n=== Three-field construction (genuinely variadic) ===")
    combined_three = CombinedField(field, field, field)
    B_three = combined_three.B(test_point)
    print(f"CombinedField(field, field, field).B(test_point) = {B_three}")
    print(f"Matches 3x single field: {jnp.allclose(B_three, 3*B_single)}")

    print("\nAll checks complete.")



if __name__ == "__main__":
    main()
