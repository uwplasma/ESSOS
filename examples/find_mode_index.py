"""Find the correct coefficient index for a proper minor radius."""

import sys
sys.path.insert(0, '/home/exouser/ESSOS')
import numpy as np

# Mode numbers for mpol=2, ntor=2
xm = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
xn = np.array([0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2])

print("Mode index mapping:")
for i in range(len(xm)):
    print(f"  Index {i}: (m={xm[i]}, n={xn[i]})")
    if xm[i] == 1 and xn[i] == 0:
        print(f"    >>> This is the pure poloidal mode (m=1, n=0) - use index {i} for minor radius!")
    if xm[i] == 0 and xn[i] == 0:
        print(f"    >>> This is the major radius mode (m=0, n=0) - use index {i} for major radius!")

print("\nFor a surface with minor radius:")
print("  x1_coeffs[0] = R_major  # Major radius (m=0,n=0)")  
print("  x1_coeffs[5] = r_minor  # Minor radius (m=1,n=0)")
print("\nCurrent examples incorrectly use:")
print("  x1_coeffs[2] = r_minor  # This is (m=0,n=2) - toroidal only, no minor radius effect!")
