"""Whole-ladder covariance and block alignment oracles."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.statistics import NAMES, _block_means
from phi4_analysis.tempering import (
    bootstrap_finite_size,
    draw_ladder,
    ladder_bootstrap,
    synchronized_blocks,
)


class PTStatistics(unittest.TestCase):
    def test_block_alignment(self):
        rng = np.random.default_rng(99)
        e = rng.normal(size=(4000, 3))
        m = rng.normal(size=(4000, 3))
        blocks = synchronized_blocks(e, m, 100)
        self.assertEqual(blocks.shape, (40, 3, 6))
        for slot in range(3):
            np.testing.assert_array_equal(
                blocks[:, slot, :], _block_means(e[:, slot], m[:, slot], 100)[0]
            )
        with self.assertRaises(ValueError):
            synchronized_blocks(e, m[:10], 100)

    def test_temperature_covariance_is_preserved(self):
        rng = np.random.default_rng(513)
        a = rng.normal(size=4000)
        b = rng.normal(size=4000)
        m = np.column_stack((a, 2 * a, b))
        e = 3 * m
        blocks = synchronized_blocks(e, m, 100)
        draw = draw_ladder(blocks, rng)
        np.testing.assert_allclose(
            draw[1],
            [
                2 * draw[0, 0],
                2 * draw[0, 1],
                2 * draw[0, 2],
                4 * draw[0, 3],
                16 * draw[0, 4],
                4 * draw[0, 5],
            ],
        )
        samples = ladder_bootstrap(
            blocks, 27, np.array([1.0, 1.5, 2.0]), replicates=200, seed=7
        )
        np.testing.assert_allclose(
            samples[:, 0, NAMES.index("B4")], samples[:, 1, NAMES.index("B4")]
        )
        np.testing.assert_allclose(
            2 * samples[:, 0, NAMES.index("abs_m")], samples[:, 1, NAMES.index("abs_m")]
        )
        self.assertGreater(
            np.corrcoef(
                samples[:, 0, NAMES.index("B4")], samples[:, 1, NAMES.index("B4")]
            )[0, 1],
            0.999999,
        )

    def test_joint_finite_size_oracle_and_stream_independence(self):
        rng = np.random.default_rng(81)
        temperatures = np.array([16.23, 16.24, 16.25, 16.26, 16.27])
        ladders = []
        for L in (8, 12, 16, 24, 32):
            means = []
            for T in temperatures:
                abs_m = 2 * L ** (-0.326 / 0.63) * (1 - 0.2 * (T - 16.25))
                chi = 0.03 * L ** (1.237 / 0.63) * (1 + 0.1 * (T - 16.25))
                m2 = abs_m**2 + T / L**3 * chi
                B4 = 1.6 + 0.02 * (T - 16.25) * L ** (1 / 0.63)
                means.append([1.0, 0.0, abs_m, m2, B4 * m2 * m2, 2.0])
            noise = rng.normal(size=40)
            noise -= noise.mean()
            blocks = np.array(means)[None, :, :] * (1 + 0.001 * noise[:, None, None])
            ladders.append(
                {
                    "L": L,
                    "temperatures": temperatures,
                    "blocks": blocks,
                    "replica_seeds": list(range(L * 1000, L * 1000 + 5)),
                }
            )
        result = bootstrap_finite_size(ladders, degree=1, replicates=100, seed=81)
        for key, target in (
            ("Tc", 16.25),
            ("nu", 0.63),
            ("beta", 0.326),
            ("gamma", 1.237),
        ):
            self.assertAlmostEqual(result["fit"][key], target, places=7)
        self.assertEqual(result["bootstrap_successes"], 100)
        ladders[1]["replica_seeds"] = ladders[0]["replica_seeds"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            bootstrap_finite_size(ladders, degree=1, replicates=100)


if __name__ == "__main__":
    unittest.main()
