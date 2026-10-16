import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from phi4_analysis.statistics import block_bootstrap, block_jackknife, observables


class Statistics(unittest.TestCase):
    def test_exact_observables(self):
        o = observables([-2, 0, -2, 0], [-2, -1, 1, 2], 27, 2)
        self.assertAlmostEqual(o["m"], 0)
        self.assertAlmostEqual(o["m2"], 2.5)
        self.assertAlmostEqual(o["B4"], 8.5 / 6.25)
        self.assertAlmostEqual(o["chi_abs"], 27 / 2 * (2.5 - 2.25))
        self.assertAlmostEqual(o["chi_signed"], 27 / 2 * 2.5)
        self.assertAlmostEqual(o["heat_capacity_per_site"], 27 / 4)
        self.assertAlmostEqual(observables([1, 1], [2, 2], 27, 1)["B4"], 1)
        with self.assertRaises(ValueError):
            observables([0, 0], [0, 0], 27, 1)

    def test_invalid_and_extreme_inputs(self):
        with self.assertRaises(ValueError):
            observables([1, 2], [1, 2], 27, True)
        with self.assertRaises((ValueError, FloatingPointError)):
            observables([1, 2], [1, 2], 27, 1e-300)
        with self.assertRaises((ValueError, FloatingPointError)):
            observables([1, 2], [1e200, 1e200], 27, 1)

    def test_jackknife_against_explicit_deletions(self):
        m = np.linspace(-2, 3, 203)
        e = np.sin(m)
        result = block_jackknife(e, m, 64, 3, 10)
        self.assertEqual(result["dropped_tail_samples"], 3)
        e, m = e[:200], m[:200]
        deleted = []
        for i in range(20):
            mask = np.ones(200, dtype=bool)
            mask[i * 10 : (i + 1) * 10] = False
            deleted.append(observables(e[mask], m[mask], 64, 3)["B4"])
        self.assertAlmostEqual(
            result["standard_error"]["B4"], np.sqrt(19 * np.var(deleted))
        )

    def test_correlated_variance_inflation(self):
        rng = np.random.default_rng(912)
        n, rho = 262144, 0.8
        noise = rng.normal(size=n + 2000)
        x = np.zeros_like(noise)
        for i in range(1, len(x)):
            x[i] = rho * x[i - 1] + noise[i]
        x = x[2000:]
        se = block_jackknife(x, x + 5, 27, 1, 256)["standard_error"]["e"]
        predicted = np.sqrt(np.var(x) / n * (1 + rho) / (1 - rho))
        self.assertTrue(0.8 < se / predicted < 1.2, (se, predicted))
        naive = np.std(x) / np.sqrt(n)
        self.assertGreater(se, 2.5 * naive)
        iid = rng.normal(size=n)
        iid_se = block_jackknife(iid, iid + 5, 27, 1, 256)["standard_error"]["e"]
        self.assertTrue(0.8 < iid_se / (np.std(iid) / np.sqrt(n)) < 1.2)

    def test_bootstrap_preserves_joint_moment_identity(self):
        # Since e=m² in this fixture, joint resampling must retain their equality
        # in every replicate and therefore in both percentile endpoints.
        m = np.linspace(0.5, 3, 400)
        result = block_bootstrap(m * m, m, 27, 2, 4, seed=17, replicates=100)
        self.assertEqual(
            result["percentile_interval_95"]["e"],
            result["percentile_interval_95"]["m2"],
        )

    def test_joint_bootstrap_and_replay(self):
        m = np.tile([-2.0, -1.0, 1.0, 2.0], 100)
        a = block_bootstrap(m * m, m, 27, 2, 4, seed=7, replicates=100)
        self.assertEqual(a, block_bootstrap(m * m, m, 27, 2, 4, seed=7, replicates=100))
        self.assertEqual(a["percentile_interval_95"]["B4"], [8.5 / 6.25, 8.5 / 6.25])
        with self.assertRaises(ValueError):
            block_jackknife(m, m, 27, 1, 100)


if __name__ == "__main__":
    unittest.main()
