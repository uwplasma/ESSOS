import json
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essos.coils import Coils, Coils_from_json, Curves


def test_curves_initialization():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 100
    assert curves.nfp == 1
    assert curves.stellsym
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)


def test_curves_initialization_with_params():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, n_segments=50, nfp=2, stellsym=False)
    assert curves.dofs.shape == (2, 3, 5)
    assert curves.n_segments == 50
    assert curves.nfp == 2
    assert not curves.stellsym
    assert curves.order == 2
    assert curves.curves.shape == (4, 3, 5)
    assert curves.gamma.shape == (4, 50, 3)
    assert curves.gamma_dash.shape == (4, 50, 3)


def test_curves_computed_attributes():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert curves.gamma.shape == (4, 100, 3)
    assert curves.gamma_dash.shape == (4, 100, 3)
    assert curves.length.shape == (4,)


def test_curves_property_setters():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    new_dofs = jnp.ones((2, 3, 5))
    curves.dofs = new_dofs
    assert jnp.allclose(curves.dofs, new_dofs)
    curves.n_segments = 50
    assert curves.n_segments == 50
    curves.nfp = 2
    assert curves.nfp == 2
    curves.stellsym = False
    assert not curves.stellsym


def test_curves_str_repr():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    assert isinstance(str(curves), str)
    assert isinstance(repr(curves), str)


def test_curves_save_curves(tmp_path):
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    filename = tmp_path / "curves.txt"
    curves.save_curves(filename)
    with open(filename, "r") as file:
        content = file.read()
    assert "nfp stellsym order" in content


def test_curves_plot():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs)
    curves.plot(show=False)


def test_curves_len():
    dofs = jnp.zeros((2, 3, 5))
    nfp = random.randint(1, 10)
    curves = Curves(dofs, nfp=nfp)
    assert len(curves) == 2 * 2 * nfp


def test_curves_getitem():
    dofs = jnp.ones((2, 3, 5))
    nfp = random.randint(4, 10)
    curves = Curves(dofs, nfp=nfp)
    assert curves[0].curves.shape == (1, 3, 5)
    assert curves[1].curves.shape == (1, 3, 5)
    assert curves[2].curves.shape == (1, 3, 5)
    assert curves[3].curves.shape == (1, 3, 5)
    assert curves[1:3].curves.shape == (2, 3, 5)
    assert curves[jnp.array([0, 3])].curves.shape == (2, 3, 5)


def test_curves_add():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    curves3 = curves1 + curves2
    assert curves3.dofs.shape == (4, 3, 5)


def test_curves_contains():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    assert curves[0] in curves
    assert curves[1] in curves


def test_curves_eq():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=False)
    assert curves1 == curves2


def test_curves_ne():
    dofs = jnp.zeros((2, 3, 5))
    curves1 = Curves(dofs, stellsym=False)
    curves2 = Curves(dofs, stellsym=True)
    assert curves1 != curves2


def test_curves_iter():
    dofs = jnp.zeros((2, 3, 5))
    curves = Curves(dofs, stellsym=False)
    for curve in curves:
        assert curve.curves.shape == (1, 3, 5)


def test_coils_pytree_roundtrip_preserves_physical_current_scale():
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    rebuilt = jax.jit(lambda item: item)(coils)

    np.testing.assert_allclose(
        rebuilt.base_currents, coils.base_currents, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(rebuilt.currents, coils.currents, rtol=0.0, atol=0.0)


def test_coils_pytree_unflatten_accepts_an_all_zero_current_tangent():
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    leaves, definition = jax.tree_util.tree_flatten(coils)
    rebuilt = jax.tree_util.tree_unflatten(
        definition,
        [leaves[0], jnp.zeros_like(leaves[1])],
    )

    np.testing.assert_array_equal(rebuilt.base_currents, np.zeros(2))
    np.testing.assert_array_equal(rebuilt.currents, np.zeros(2))
    assert np.isfinite(float(rebuilt.currents_scale))


def test_coils_json_roundtrip_preserves_physical_current_scale(tmp_path):
    curves = Curves(jnp.ones((2, 3, 5)), n_segments=12, stellsym=False)
    coils = Coils(curves, jnp.asarray([1.0e5, 2.0e5]))
    path = tmp_path / "coils.json"
    coils.to_json(path)
    rebuilt = Coils_from_json(path)

    np.testing.assert_allclose(
        rebuilt.base_currents, coils.base_currents, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(rebuilt.currents, coils.currents, rtol=0.0, atol=0.0)


def test_coils_json_supports_scale_aware_and_legacy_current_formats(tmp_path):
    common = {
        "nfp": 1,
        "stellsym": False,
        "order": 2,
        "n_segments": 12,
        "dofs_curves": np.ones((2, 3, 5)).tolist(),
    }
    scale_aware = tmp_path / "scale-aware.json"
    scale_aware.write_text(
        json.dumps(
            {
                **common,
                "dofs_currents": [2.0 / 3.0, 4.0 / 3.0],
                "currents_scale": 1.5e5,
            }
        )
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({**common, "dofs_currents": [1.0e5, 2.0e5]}))

    for path in (scale_aware, legacy):
        coils = Coils_from_json(path)
        np.testing.assert_allclose(coils.base_currents, [1.0e5, 2.0e5])


def test_coils_json_rejects_inconsistent_redundant_currents(tmp_path):
    path = tmp_path / "inconsistent.json"
    path.write_text(
        json.dumps(
            {
                "nfp": 1,
                "stellsym": False,
                "order": 2,
                "n_segments": 12,
                "dofs_curves": np.ones((2, 3, 5)).tolist(),
                "dofs_currents": [2.0 / 3.0, 4.0 / 3.0],
                "currents_scale": 1.5e5,
                "base_currents": [1.0e5, 3.0e5],
            }
        )
    )

    with pytest.raises(ValueError, match="inconsistent current representations"):
        Coils_from_json(path)


if __name__ == "__main__":
    pytest.main()
