import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class RandomForestModel:
    def __init__(self, region: str):
        self.region = region
        self.model = RandomForestRegressor(n_estimators=200, criterion='friedman_mse', random_state=42)
        self.feature_names_ = None
        print(f"Initialised Random Forest model for {self.region}.")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"Training Random Forest model for {self.region}...")
        self.model.fit(X_train, y_train)
        self.feature_names_ = X_train.columns.tolist()
        print("Training complete.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        print(f"Making predictions with Random Forest model for {self.region}...")
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        print(f"Evaluating {self.model.__class__.__name__} for {self.region} on test data...")
        y_pred = self.predict(X_test)
        # Ensure arrays are flat for metric calculation and plotting
        y_test_flat = y_test.values
        y_pred_flat = y_pred.flatten()
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        results = {"MAE": mae, "RMSE": rmse}
        print(f"Evaluation complete. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        # Return the metrics and the data needed for plotting
        return results, y_test_flat, y_pred_flat

    def save_results(self, results: dict, y_true: np.ndarray, y_pred: np.ndarray, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save Text Results
        results_path = os.path.join(directory, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.model.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Metrics saved to {results_path}")

        # Generate and Save Plots

        # Predicted vs. Actual Scatter Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.3, label='Model Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title('Predicted vs. Actual RRP')
        plt.xlabel('Actual RRP')
        plt.ylabel('Predicted RRP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'predicted_vs_actual.png'))
        plt.close()

        # Residuals Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs. Predicted Values')
        plt.xlabel('Predicted RRP')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'residuals_plot.png'))
        plt.close()

        # Time Series Comparison (first 1000 points)
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:1000], label='Actual Values', color='blue')
        plt.plot(y_pred[:1000], label='Predicted Values', color='orange', alpha=0.8)
        plt.title('Time Series Comparison (First 1000 Test Points)')
        plt.xlabel('Time Step')
        plt.ylabel('RRP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'timeseries_comparison.png'))
        plt.close()

        # Feature Importance Plot (Specific to tree-based models)
        if hasattr(self, 'feature_names_') and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': self.feature_names_,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            plt.figure(figsize=(10, 8))
            plt.barh(feature_df['Feature'], feature_df['Importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(directory, 'feature_importance.png'))
            plt.close()

        print(f"Diagnostic plots saved to {directory}")