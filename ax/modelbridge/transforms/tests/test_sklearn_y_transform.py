import numpy as np
from ax.modelbridge.transforms.sklearn_y import (
    InfeasibleTransformer,
    LogWarpingTransformer,
)
from ax.utils.common.testutils import TestCase


class LogWarpingTransformerTest(TestCase):
    def test_init(self) -> None:
        # Test valid initialization
        transformer = LogWarpingTransformer(offset=1.5)
        self.assertEqual(transformer.offset, 1.5)
        self.assertTrue(transformer.copy)

        # Test invalid offset
        with self.assertRaisesRegex(ValueError, "offset must be greater than 1"):
            LogWarpingTransformer(offset=0.5)

    def test_transform_simple(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer = LogWarpingTransformer()
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        # Test shape preservation
        self.assertEqual(X_transformed.shape, X.shape)

        # Test values are changed
        self.assertFalse(np.allclose(X_transformed, X))

        # Test inverse transform recovers original
        X_recovered = transformer.inverse_transform(X_transformed)
        self.assertTrue(np.allclose(X_recovered, X))

    def test_nan_handling(self) -> None:
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 2.0]])
        transformer = LogWarpingTransformer()
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        # Test NaN values remain NaN
        self.assertTrue(np.isnan(X_transformed[0, 1]))
        self.assertTrue(np.isnan(X_transformed[2, 0]))

        # Test non-NaN values are transformed
        self.assertFalse(np.isnan(X_transformed[0, 0]))
        self.assertFalse(np.isnan(X_transformed[1, 0]))
        self.assertFalse(np.isnan(X_transformed[1, 1]))
        self.assertFalse(np.isnan(X_transformed[2, 1]))

        # Test inverse transform preserves NaN
        X_recovered = transformer.inverse_transform(X_transformed)
        self.assertTrue(np.isnan(X_recovered[0, 1]))
        self.assertTrue(np.isnan(X_recovered[2, 0]))

        # Test non-NaN values are recovered correctly
        X_no_nan = X[~np.isnan(X)]
        X_recovered_no_nan = X_recovered[~np.isnan(X_recovered)]
        self.assertTrue(np.allclose(X_no_nan, X_recovered_no_nan))

    def test_transform_bounds(self) -> None:
        # Test with values near bounds
        X = np.array([[1.0, 10.0], [2.0, 20.0]])
        transformer = LogWarpingTransformer()
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        # Test transformed values are bounded
        self.assertTrue(np.all(X_transformed[np.isfinite(X_transformed)] <= 0.5))

        # Test inverse transform recovers original values
        X_recovered = transformer.inverse_transform(X_transformed)
        self.assertTrue(np.allclose(X_recovered, X))

    def test_input_validation(self) -> None:
        transformer = LogWarpingTransformer()

        # Test 1D array raises error
        with self.assertRaises(ValueError):
            transformer.fit(np.array([1.0, 2.0]))

        # Test wrong shape in transform after fit
        transformer.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with self.assertRaises(ValueError):
            transformer.transform(np.array([[1.0], [2.0]]))

    def test_copy_behavior(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_orig = X.copy()

        # Test with copy=True (default)
        transformer = LogWarpingTransformer(copy=True)
        transformer.fit(X)
        self.assertTrue(np.array_equal(X, X_orig))  # Original should be unchanged

        # Test with copy=False
        transformer = LogWarpingTransformer(copy=False)
        X_transform = transformer.fit_transform(X)
        self.assertFalse(np.array_equal(X, X_orig))  # Original should be modified
        self.assertTrue(np.array_equal(X, X_transform))  # Should be the same object

    def test_partial_fit(self) -> None:
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])

        transformer = LogWarpingTransformer()
        transformer.partial_fit(X1)

        # Test that the transformer uses the full range of values
        self.assertTrue(np.allclose(transformer.labels_min_, np.array([[1.0, 2.0]])))

        transformer.partial_fit(X2)
        self.assertTrue(np.allclose(transformer.labels_max_, np.array([[7.0, 8.0]])))


class TestInfeasibleTransformer(TestCase):
    def test_transform_basic(self) -> None:
        """Test basic transformation with simple data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]])
        transformer = InfeasibleTransformer()
        transformer.fit(X)

        # Transform the data
        X_transformed = transformer.transform(X)

        # Check that non-nan values are shifted
        self.assertFalse(np.allclose(X_transformed[0:2], X[0:2]))
        self.assertTrue(np.allclose(X_transformed[0:2] - transformer.shift_, X[0:2]))

        # Check that all values are finite
        self.assertFalse(np.isnan(X_transformed).any())

        # Check that previously nan values are replaced with warped_bad_value
        self.assertEqual(X_transformed[2, 0], transformer.warped_bad_value_[0, 0])

        # Check inverse transform
        X_inverse = transformer.inverse_transform(X_transformed)
        # Non-nan values should be recovered exactly
        self.assertTrue(np.allclose(X_inverse[0:2], X[0:2]))

    def test_transform_all_nan_column(self) -> None:
        """Test handling of columns that are all NaN."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        transformer = InfeasibleTransformer()
        with self.assertRaisesRegex(
            ValueError, "Cannot fit InfeasibleTransformer on all-NaN feature columns."
        ):
            transformer.fit(X)

    def test_transform_single_value(self) -> None:
        """Test transformation of single non-nan value."""
        X = np.array([[1.0], [np.nan], [np.nan]])
        transformer = InfeasibleTransformer()
        transformer.fit(X)

        X_transformed = transformer.transform(X)

        # Check that transformation preserves the relative ordering
        self.assertGreater(X_transformed[0, 0], X_transformed[1, 0])
        self.assertEqual(X_transformed[1, 0], X_transformed[2, 0])

    def test_shape_validation(self) -> None:
        """Test that the transformer validates input shapes."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer = InfeasibleTransformer()
        transformer.fit(X)

        # Try to transform data with wrong number of features
        X_wrong_shape = np.array([[1.0], [2.0]])
        with self.assertRaisesRegex(
            ValueError, "features, but InfeasibleTransformer is expecting"
        ):
            transformer.transform(X_wrong_shape)

    def test_copy_behavior(self) -> None:
        """Test that copy parameter works as expected."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_orig = X.copy()

        # Test with copy=True (default)
        transformer = InfeasibleTransformer(copy=True)
        transformer.fit(X)
        self.assertTrue(np.array_equal(X, X_orig))  # Original should be unchanged

        # Test with copy=False
        transformer = InfeasibleTransformer(copy=False)
        X_transform = transformer.fit_transform(X)
        self.assertFalse(np.array_equal(X, X_orig))  # Original should be modified
        self.assertTrue(np.array_equal(X, X_transform))  # Should be the same object

    def test_p_feasible_calculation(self) -> None:
        """Test that p_feasible is calculated correctly."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]])
        transformer = InfeasibleTransformer()
        transformer.fit(X)

        # For first column: 2 feasible out of 3 total
        expected_p_feasible_1 = (0.5 + 2) / (1 + 3)
        # For second column: 2 feasible out of 3 total
        expected_p_feasible_2 = (0.5 + 2) / (1 + 3)

        # Calculate actual p_feasible from the shift formula
        # shift = -mean(X) * p_feasible - warped_bad_value * (1 - p_feasible)
        p_feasible_1 = -(
            transformer.shift_[0, 0] + transformer.warped_bad_value_[0, 0]
        ) / (np.nanmean(X[:, 0]) - transformer.warped_bad_value_[0, 0])
        p_feasible_2 = -(
            transformer.shift_[0, 1] + transformer.warped_bad_value_[0, 1]
        ) / (np.nanmean(X[:, 1]) - transformer.warped_bad_value_[0, 1])

        self.assertTrue(np.allclose(p_feasible_1, expected_p_feasible_1))
        self.assertTrue(np.allclose(p_feasible_2, expected_p_feasible_2))


if __name__ == "__main__":
    import unittest

    unittest.main()
