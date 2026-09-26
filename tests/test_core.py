import numpy as np
import pytest

from vbc import cosine_angle, decompose, transform_endpoints
from vbc.core import preprocess


def test_perpendicular_vectors_have_ninety_degree_angle():
    cosine, angle = cosine_angle([1, 0], [0, 2])
    assert cosine == pytest.approx(0.0)
    assert np.degrees(angle) == pytest.approx(90.0)


def test_small_angle_retains_precision():
    _, angle = cosine_angle([1., 0.], [1., 1e-12])
    assert angle == pytest.approx(1e-12, rel=1e-12, abs=0.)


def test_parallel_and_antiparallel_angles():
    assert cosine_angle([1., 2.], [1., 2.])[1] == 0.
    assert cosine_angle([1., 2.], [-1., -2.])[1] == pytest.approx(np.pi)


def test_geometric_orthogonal_decomposition():
    result = decompose([1, 0], [1, 1], preprocessing="none")
    assert result.angle_degrees == pytest.approx(45.0)
    np.testing.assert_allclose(result.parallel, [1.0, 0.0])
    np.testing.assert_allclose(result.perpendicular, [0.0, 1.0])
    assert np.dot([1, 0], result.perpendicular) == pytest.approx(0.0)
    assert np.linalg.norm(result.perpendicular) == pytest.approx(
        np.linalg.norm([1, 1]) * np.sin(result.angle_radians)
    )


def test_multiple_endpoints_excludes_primary():
    results = transform_endpoints(
        {"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 1, 2], "D": [1, 3, 2]}, "A"
    )
    assert set(results) == {"B", "C", "D"}
    primary = np.asarray([1, 2, 3])
    for result in results.values():
        assert np.dot(primary, result.perpendicular) == pytest.approx(0.0, abs=1e-12)


def test_minmax_is_unitless_and_non_negative():
    result = preprocess([-10, 0, 30], "minmax")
    np.testing.assert_allclose(result, [0.0, 0.25, 1.0])


def test_minmax_handles_extreme_mixed_sign_values():
    result = preprocess([-1e308, 0, 1e308], "minmax")
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_l2_normalization_preserves_angle():
    a = [10, 12, 9, 14]
    b = [5, 7, 6, 8]
    raw = decompose(a, b, preprocessing="none")
    normalized = decompose(a, b, preprocessing="l2")
    assert normalized.angle_radians == pytest.approx(raw.angle_radians)
    assert np.linalg.norm(preprocess(a, "l2")) == pytest.approx(1.0)
    assert np.linalg.norm(preprocess(b, "l2")) == pytest.approx(1.0)


def test_constant_endpoint_cannot_be_minmax_normalized():
    with pytest.raises(ValueError, match="non-constant"):
        decompose([1, 2, 3], [5, 5, 5], preprocessing="minmax")


def test_unknown_preprocessing_is_rejected():
    with pytest.raises(ValueError, match="unknown preprocessing"):
        preprocess([1, 2, 3], "typo")  # type: ignore[arg-type]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number"):
        decompose([1, 2], [1, 2, 3])


def test_vbc_requires_at_least_two_endpoints():
    with pytest.raises(ValueError, match="at least one other endpoint"):
        transform_endpoints({"A": [1, 2, 3]}, "A")


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_projection_is_stable_across_scales(scale):
    result = decompose(np.array([1., 0.]) * scale, np.array([1., 1.]) * scale)
    assert result.angle_degrees == pytest.approx(45.)
    np.testing.assert_allclose(result.parallel / scale, [1., 0.], atol=1e-14)
    np.testing.assert_allclose(result.perpendicular / scale, [0., 1.], atol=1e-14)


@pytest.mark.parametrize("mode", ["none", "l2", "minmax", "zscore"])
def test_projection_reconstruction_and_pythagorean_identity(mode):
    rng = np.random.default_rng(2026)
    a, b = rng.normal(size=(2, 100))
    aa, bb = preprocess(a, mode), preprocess(b, mode)
    result = decompose(a, b, preprocessing=mode)
    np.testing.assert_allclose(result.parallel + result.perpendicular, bb, atol=1e-14)
    assert np.dot(aa, result.perpendicular) == pytest.approx(0., abs=1e-12)
    assert np.dot(result.parallel, result.perpendicular) == pytest.approx(0., abs=1e-12)
    assert np.dot(bb, bb) == pytest.approx(
        np.dot(result.parallel, result.parallel) + np.dot(result.perpendicular, result.perpendicular)
    )


def test_common_participant_permutation_preserves_results():
    a, b = np.array([2., 4., 1., 3.]), np.array([3., 1., 5., 2.])
    order = [2, 0, 3, 1]
    original = decompose(a, b)
    permuted = decompose(a[order], b[order])
    assert original.angle_radians == pytest.approx(permuted.angle_radians)
    np.testing.assert_allclose(permuted.perpendicular, original.perpendicular[order])


@pytest.mark.parametrize("bad", [[0, 0], [1, np.nan], [1, np.inf], [1]])
def test_undefined_inputs_raise(bad):
    with pytest.raises(ValueError):
        decompose([1, 2], bad)


def test_zscore_matches_pearson_and_sample_standardization():
    a = np.array([2., 5., 4., 9., 1.])
    b = np.array([8., 3., 6., 2., 7.])
    expected = (a - a.mean()) / a.std(ddof=1)
    np.testing.assert_allclose(preprocess(a, "zscore"), expected)
    result = decompose(a, b, preprocessing="zscore")
    assert result.cosine_similarity == pytest.approx(np.corrcoef(a, b)[0, 1])
    assert result.perpendicular.mean() == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize("values", [[0., 0., 0.], [5., 5., 5.]])
def test_zscore_rejects_constant_endpoints(values):
    with pytest.raises(ValueError, match="non-constant"):
        preprocess(values, "zscore")


@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_zscore_is_stable_across_scales(scale):
    values = np.array([-2., 0., 1., 4.])
    np.testing.assert_allclose(preprocess(values * scale, "zscore"),
                               (values - values.mean()) / values.std(ddof=1))
