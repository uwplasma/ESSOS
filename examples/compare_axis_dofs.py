import jax.numpy as jnp
import matplotlib.pyplot as plt

from essos.surfaces import SurfaceGFrameFourier


def build_circular_cross_section(minor_radius):
    # mpol=1, ntor=0 -> modes: (m=0,n=0) and (m=1,n=0)
    # X1 = a*cos(theta), X2 = a*sin(theta)
    x1_coeffs = jnp.array([0.0, minor_radius])
    x2_coeffs = jnp.array([0.0, minor_radius])
    return x1_coeffs, x2_coeffs


def build_axis_coeffs(axis_ntor, major_radius, r_perturb=None, z_perturb=None):
    n_modes = axis_ntor + 1
    rc = jnp.zeros(n_modes)
    zs = jnp.zeros(n_modes)
    rc = rc.at[0].set(major_radius)
    if r_perturb is not None:
        r_list = r_perturb if isinstance(r_perturb, (list, tuple)) and isinstance(r_perturb[0], (list, tuple)) else [r_perturb]
        for idx, amp in r_list:
            rc = rc.at[idx].set(amp)
    if z_perturb is not None:
        z_list = z_perturb if isinstance(z_perturb, (list, tuple)) and isinstance(z_perturb[0], (list, tuple)) else [z_perturb]
        for idx, amp in z_list:
            zs = zs.at[idx].set(amp)
    return rc, zs


def main():
    nfp = 1
    mpol = 1
    ntor = 0
    axis_mpol = 0
    axis_ntor = 5

    major_radius = 1.0
    minor_radius = 0.2
    intensity = 0.45

    x1_coeffs, x2_coeffs = build_circular_cross_section(minor_radius)

    cases = [
        ("Base", None, None),
        ("R n=1", (1, intensity), None),
        ("R n=3", (3, intensity), None),
        ("Z n=1", None, (1, intensity)),
        ("Z n=3", None, (3, intensity)),
        ("R n=1 + Z n=1", (1, intensity), (1, intensity)),
        ("R n=2 + Z n=3", (2, intensity * 0.8), (3, intensity)),
        ("R n=4 + Z n=2", (4, intensity), (2, intensity * 0.8)),
        ("R n=1,3 + Z n=1", [(1, intensity * 0.7), (3, intensity * 0.6)], (1, intensity)),
    ]

    fig, axes = plt.subplots(3, 3, subplot_kw={"projection": "3d"}, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (label, r_perturb, z_perturb) in zip(axes, cases):
        rc_coeffs, zs_coeffs = build_axis_coeffs(axis_ntor, major_radius, r_perturb, z_perturb)
        surface = SurfaceGFrameFourier(
            x1_coeffs=x1_coeffs,
            x2_coeffs=x2_coeffs,
            nfp=nfp,
            mpol=mpol,
            ntor=ntor,
            ntheta=40,
            nphi=60,
            close=True,
            range_torus="full torus",
            axis_major_radius=major_radius,
            axis_shape="circle",
            axis_pitch=0.0,
            axis_twist_rate=0.0,
            axis_function=None,
            enable_axis_dofs=True,
            axis_rc_coeffs=rc_coeffs,
            axis_zs_coeffs=zs_coeffs,
            axis_mpol=axis_mpol,
            axis_ntor=axis_ntor,
        )
        surface.plot(ax=ax, show=False, axis_equal=True, color="#1f77b4")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig("compare_axis_dofs.png", dpi=300)


if __name__ == "__main__":
    main()
