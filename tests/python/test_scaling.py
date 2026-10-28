"""Independent statistical and finite-size oracles; no scientific sampling."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "scripts"))
import mill
from phi4_analysis.scaling import (
    autocorrelation,
    draw_ensembles,
    power_fit,
    scaling_fit,
)
from phi4_analysis.statistics import NAMES


class ScalingTests(unittest.TestCase):
    def synthetic(self):
        Tc, nu, beta, gamma = 16.25, 0.63, 0.326, 1.237
        groups, values = [], []
        for L in (8, 12, 16, 24, 32):
            for T in (16.15, 16.20, 16.25, 16.30, 16.35):
                row = np.ones(len(NAMES))
                # Independent exact scaling curves with a common known crossing.
                row[NAMES.index("B4")] = 1.6 + 0.02 * (T - Tc) * L ** (1 / nu)
                row[NAMES.index("abs_m")] = 2 * L ** (-beta / nu) * (1 - 0.2 * (T - Tc))
                row[NAMES.index("chi_abs")] = (
                    0.3 * L ** (gamma / nu) * (1 + 0.1 * (T - Tc))
                )
                groups.append({"L": L, "T": T, "errors": np.full(len(NAMES), 0.001)})
                values.append(row)
        return groups, np.array(values)

    def test_known_exponents_and_common_crossing(self):
        groups, values = self.synthetic()
        fit = scaling_fit(groups, values)
        for key, value in [
            ("Tc", 16.25),
            ("nu", 0.63),
            ("beta", 0.326),
            ("gamma", 1.237),
        ]:
            self.assertAlmostEqual(fit[key], value, places=9)
        self.assertEqual(fit["crossing_sizes"], (24, 32))
        smaller = scaling_fit(groups, values, lmin=12, crossing_sizes=(24, 32))
        self.assertAlmostEqual(smaller["nu"], 0.63, places=9)

    def test_no_extrapolated_crossing(self):
        groups, values = self.synthetic()
        for i, g in enumerate(groups):
            values[i, NAMES.index("B4")] = 1.6 + 0.02 * (g["T"] - 18) * g["L"] ** (
                1 / 0.63
            )
        with self.assertRaisesRegex(ValueError, "crossing"):
            scaling_fit(groups, values)

    def test_too_few_sizes_or_temperatures(self):
        groups, values = self.synthetic()
        with self.assertRaises(ValueError):
            scaling_fit(groups, values, lmin=24)
        keep = [i for i, g in enumerate(groups) if g["T"] in (16.2, 16.3)]
        with self.assertRaises(ValueError):
            scaling_fit([groups[i] for i in keep], values[keep])

    def test_power_law_native_oracle(self):
        sizes = np.array([4, 8, 16, 32])
        fit = power_fit(sizes, 3 * sizes**1.7, np.ones(4) * 0.01)
        self.assertAlmostEqual(fit["amplitude"], 3, places=10)
        self.assertAlmostEqual(fit["slope"], 1.7, places=10)
        self.assertEqual(fit["dof"], 2)

    def test_iid_and_ar1_autocorrelation(self):
        rng = np.random.default_rng(1041)
        iid = rng.normal(size=150000)
        self.assertAlmostEqual(autocorrelation(iid)["tau_samples"], 0.5, delta=0.08)
        ar = np.zeros(len(iid))
        for i in range(1, len(ar)):
            ar[i] = 0.8 * ar[i - 1] + iid[i]
        result = autocorrelation(ar)
        # Analytic tau = (1+rho)/(2*(1-rho)) = 4.5.
        self.assertAlmostEqual(result["tau_samples"], 4.5, delta=0.6)
        self.assertTrue(result["tail_resolved"])
        with self.assertRaises(ValueError):
            autocorrelation(np.ones(100))

    def test_bootstrap_preserves_joint_moments_and_chain_weights(self):
        def blocks(m):
            m = np.array(m, dtype=float)
            return np.column_stack((2 * m, m, abs(m), m * m, m**4, 4 * m * m))

        group = {
            "L": 8,
            "T": 16.0,
            "blocks": [blocks([1, 2, 3, 4]), blocks([2, 3, 4, 5, 6, 7, 8, 9])],
        }
        rng = np.random.default_rng(94)
        samples = np.array([draw_ensembles([group], rng)[0] for _ in range(400)])
        np.testing.assert_allclose(
            samples[:, NAMES.index("e")], 2 * samples[:, NAMES.index("m")]
        )
        expected = (
            np.mean([1, 2, 3, 4]) * 4 + np.mean([2, 3, 4, 5, 6, 7, 8, 9]) * 8
        ) / 12
        self.assertAlmostEqual(samples[:, NAMES.index("m")].mean(), expected, delta=0.1)

    def test_account_cap_counts_expanded_array_elements(self):
        self.assertEqual(mill.slots("1\n2\n", 27), 2)
        with self.assertRaises(ValueError):
            mill.slots("1\n2\n3\n", 27)
        self.assertEqual(mill.queue_count("12_0\n12_1\n12_2\n"), 3)
        for name in ("../bad", "name; sbatch", ""):
            with self.assertRaises(ValueError):
                mill.checked_name(name)


if __name__ == "__main__":
    unittest.main()
