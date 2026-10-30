"""Check that cached PT reblocking reproduces the raw-history analysis."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "scripts"))
from phi4_analysis.statistics import NAMES, block_jackknife
from phi4_analysis.tempering import synchronized_blocks
from review_pt_production import block_estimates, reblock


class CachedPTReview(unittest.TestCase):
    def test_cached_and_raw_errors_agree(self):
        rng = np.random.default_rng(504)
        m = rng.normal(size=(8000, 3)) + np.linspace(0, 0.8, 3)
        e = 0.3 * m**2 + rng.normal(size=m.shape)
        temperatures = [14.9, 15.32, 16.5]
        cached = synchronized_blocks(e, m, 100)
        longer = reblock(cached, 4)
        central, errors = block_estimates(longer, 4, temperatures)
        for slot, T in enumerate(temperatures):
            raw = block_jackknife(e[:, slot], m[:, slot], 4**3, T, 400)
            np.testing.assert_allclose(
                central[slot], [raw["estimate"][name] for name in NAMES], rtol=1e-12
            )
            np.testing.assert_allclose(
                errors[slot],
                [raw["standard_error"][name] for name in NAMES],
                rtol=1e-11,
            )
        # Covariance-preserving blocks must refer to exactly the same time bins.
        np.testing.assert_allclose(longer, synchronized_blocks(e, m, 400), rtol=1e-13)

    def test_incomplete_reblocking_is_refused(self):
        with self.assertRaises(ValueError):
            reblock(np.ones((21, 3, 6)), 2)
        with self.assertRaises(ValueError):
            block_estimates(np.ones((19, 3, 6)), 4, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
