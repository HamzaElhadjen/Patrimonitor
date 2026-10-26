import numpy as np
from synth.simulator import generate_crack_image
from crackgraph import extract_crack_graph
from uncertainty import local_tolerance_map, compare_masks_with_tolerance
from diagnostic import suggest_mechanisms


def test_synthetic_generator_reproducible():
    a, ma, _ = generate_crack_image(seed=123)
    b, mb, _ = generate_crack_image(seed=123)
    assert np.array_equal(a, b)
    assert np.array_equal(ma, mb)


def test_graph_empty_mask():
    result = extract_crack_graph(np.zeros((40, 40), np.uint8))
    assert result["summary"]["length_px"] == 0


def test_uncertainty_proxy():
    a = np.zeros((40, 40), np.uint8)
    b = np.zeros_like(a)
    b[10:13, 10:13] = 255
    tol = local_tolerance_map(a.shape, 1.0)
    new, info = compare_masks_with_tolerance(a, b, tol)
    assert new.shape == a.shape
    assert info["statistical_significance_established"] is False


def test_diagnostic_warning():
    result = suggest_mechanisms({"orientation_pattern": "diagonal"})
    assert "aucun diagnostic" in result["warning"].lower()
