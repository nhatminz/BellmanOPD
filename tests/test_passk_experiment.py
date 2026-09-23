from __future__ import annotations

import math
import unittest

from b200_experiment.passk_experiment import (
    compute_passk_metrics,
    summarize_checkpoint_correctness,
)


class PassKExperimentTests(unittest.TestCase):
    def test_multiple_passk_values_reuse_one_correctness_vector(self):
        # With one success among 16 samples, sampling 8 without replacement
        # finds that success with probability exactly 8/16.
        correctness = [[True] + [False] * 15]
        scores = compute_passk_metrics(correctness, [1, 8, 16])
        self.assertAlmostEqual(scores[1], 1.0 / 16.0)
        self.assertAlmostEqual(scores[8], 0.5)
        self.assertAlmostEqual(scores[16], 1.0)

    def test_k_cannot_exceed_generation_sample_count(self):
        with self.assertRaisesRegex(ValueError, "requires 1 <= k"):
            compute_passk_metrics([[True, False, False, False]], [8])

    def test_aggregation_keeps_multiple_checkpoints_distinct(self):
        k_values = {
            "AIME24": (8, 16),
            "AIME25": (8, 16),
            "AMC23": (4, 8),
        }
        first = summarize_checkpoint_correctness(
            model_group="qwen3_1.7b",
            method="opd",
            label="OPD",
            checkpoint="/tmp/opd/final",
            correctness_by_benchmark={
                "AIME24": [[True] + [False] * 15],
            },
            k_by_benchmark=k_values,
        )
        second = summarize_checkpoint_correctness(
            model_group="qwen3_1.7b",
            method="cmt",
            label="CMT",
            checkpoint="/tmp/cmt/final",
            correctness_by_benchmark={
                "AIME24": [[False] * 16],
            },
            k_by_benchmark=k_values,
        )
        rows = first + second
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["method"] for row in rows}, {"opd", "cmt"})
        self.assertEqual(len({row["checkpoint"] for row in rows}), 2)
        self.assertTrue(all(math.isfinite(row["pass_at_k"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
