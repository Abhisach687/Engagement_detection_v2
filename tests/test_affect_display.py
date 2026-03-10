import unittest

from utils.affect import display_label_to_raw_label, infer_display_level


class AffectDisplayTests(unittest.TestCase):
    def test_centered_distribution_infers_medium(self) -> None:
        inferred = infer_display_level([0.10, 0.40, 0.40, 0.10])
        self.assertEqual(inferred["label"], "Medium")
        self.assertAlmostEqual(float(inferred["score"]), 2.0, places=2)

    def test_extreme_mass_blocks_medium(self) -> None:
        inferred = infer_display_level([0.25, 0.30, 0.30, 0.15])
        self.assertEqual(inferred["label"], "Low")

    def test_strong_extreme_distribution_stays_non_medium(self) -> None:
        inferred = infer_display_level([0.85, 0.10, 0.04, 0.01])
        self.assertEqual(inferred["label"], "Very Low")

    def test_medium_maps_to_no_raw_training_label(self) -> None:
        self.assertIsNone(display_label_to_raw_label("Medium"))
        self.assertEqual(display_label_to_raw_label("High"), 2)


if __name__ == "__main__":
    unittest.main()
