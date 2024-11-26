import os
import tempfile
import unittest
import pandas as pd

from crunch.api import Target, PhaseType, Metric

from scoring import score, _mean_squared_error


class ScoringTest(unittest.TestCase):
    def test_score_function(self):
        # Prepare the inputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create y_test file
            target_name = 'DC1'
            target = Target(None, {'name': target_name})
            phase_type = PhaseType.SUBMISSION
            group_type = 'validation' if phase_type == PhaseType.SUBMISSION else 'test'
            filename = f'{group_type}-{target.name}.csv'
            filepath = os.path.join(temp_dir, filename)

            # Sample y_test DataFrame
            y_test_data = pd.DataFrame({
                'A2M': [2.0, 1.0],
                'ACP5': [4.0, 3.0]
            }, index=[94, 92])
            y_test_data.to_csv(filepath, index=True)

            # Sample prediction DataFrame:
            # - The order of data was changed to test the reindex function, ensuring predictions are matched to the correct indices.
            # - A new sample, 'DC5', was added to have two samples, testing the filtering logic for the target-specific sample.
            # - cell_id 163, which is not part of the validation group (y_test), was included to test the cell_id filter.

            prediction_data = pd.DataFrame({
                'sample': [target_name] * 6 + ['DC5', 'DC5'],
                'cell_id': [92, 92, 94, 94, 163, 163, 92, 92],
                'gene': ['A2M', 'ACP5', 'A2M', 'ACP5', 'A2M', 'ACP5', 'A2M', 'ACP5'],
                'prediction': [1.1, 3.9, 2.0, 3.4, 2.3, 1.5, 2.8, 9.9]
            })

            metric = Metric(None, target, {'id': 10, 'name': 'mse'})
            target_and_metrics = [(target, [metric])]

            # Call the score function
            scores = score(
                prediction=prediction_data,
                data_directory_path=temp_dir,
                phase_type=phase_type,
                target_and_metrics=target_and_metrics
            )

            # Expected score calculation:
            # Reconstructed predictions after filtering and pivoting:
            # Pivoted prediction for target 'DC1':
            #           A2M  ACP5
            # cell_id
            # 92       1.1   3.9
            # 94       2.0   3.4
            # Note: cell 163 is ignored as it's not in y_test.

            # y_test DataFrame:
            #           A2M  ACP5
            # cell_id
            # 92       1.0   3.0
            # 94       2.0   4.0

            # MSE calculations:
            # For cell 92:
            # MSE_cell92 = mean((1.0 - 1.1)^2, (3.0 - 3.9)^2) = mean(0.01, 0.81) = 0.41
            # For cell 94:
            # MSE_cell94 = mean((2.0 - 2.0)^2, (4.0 - 3.4)^2) = mean(0.0, 0.36) = 0.18
            #
            # Overall MSE = weighted average across cells:
            # Overall MSE = (0.41 + 0.18) / 2 = 0.295

            expected_score = 0.295

            # Assertions
            self.assertIn(metric.id, scores)
            scored_metric = scores[metric.id]
            self.assertAlmostEqual(scored_metric.value, expected_score)
            print(f"Test passed, score: {scored_metric.value}")

    @unittest.skip("This is a manual test, not executed automatically.")
    def test_with_truth_files(self):
        """
        Quickly test the `score` function using predefined truth files for y_test and predictions.
        """
        # Provide the paths to the truth files
        y_test_dir = "test/ground_truth/"
        prediction_path = "test/prediction.parquet"

        # Ensure the files exist
        if not os.path.exists(y_test_dir):
            raise FileNotFoundError(f"y_test directory not found: {y_test_dir}")
        if not os.path.exists(prediction_path):
            raise FileNotFoundError(f"Predictions file not found: {prediction_path}")

        # Load the data from the truth files
        prediction_data = pd.read_parquet(prediction_path)

        # Define target and metrics (adjust as needed)
        target_names = ['DC1', 'DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I', 'ALL']
        targets = [Target(None, {'name': target_name}) for target_name in target_names]
        metrics = [Metric(None, target, {'id': i + 1, 'name': 'mse'}) for i, target in enumerate(targets)]
        target_and_metrics = [(target, [metric]) for target, metric in zip(targets, metrics)]

        group_type = 'validation'  # Adjust based on your test phase
        # Call the score function
        scores = score(
            prediction=prediction_data,
            data_directory_path=y_test_dir,
            phase_type=PhaseType.SUBMISSION,
            target_and_metrics=target_and_metrics
        )

        # Print results for manual verification
        print("Scores from truth file test:")
        for metric_id, scored_metric in scores.items():
            print(f"Metric ID: {metric_id}, Score: {scored_metric.value}")


if __name__ == '__main__':
    unittest.main()
