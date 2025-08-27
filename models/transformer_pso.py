import numpy as np
import torch
from models.pso import PSO_Optimiser
from models.transformer import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Transformer_PSO_Model:
    def __init__(self, region: str, input_shape: int):
        self.region = region
        self.input_shape = input_shape
        self.optimiser = PSO_Optimiser(
            model_class=TransformerModel,
            region=self.region,
            input_shape=self.input_shape
        )
        self.final_model = None
        self.best_hyperparameters = None
        print(f"Initialized Transformer_PSO_Model wrapper for {self.region}.")

    def train_modelself(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, search_space: Dict[str, Tuple[Union[int, float], Union[int, float]]], pso_n_particles: int, pso_iters: int):
        print("\n--- Step 1: Finding Optimal Hyperparameters for Transformer using PSO ---")
        # Store the best params in the instance variable
        self.best_hyperparameters = self.optimiser.find_optimal_hyperparameters(
            search_space, X_train, y_train, X_val, y_val, pso_n_particles, pso_iters
        )
        print(f"\nOptimal hyperparameters found: {self.best_hyperparameters}")
        print("\nDecoding Hyperparameters")
        n_heads = int(2 ** round(self.best_hyperparameters['n_heads']))
        model_dim = int(round(self.best_hyperparameters['model_dim'] / n_heads)) * n_heads
        model_constructor_params = {
            'model_dim': model_dim,
            'n_heads': n_heads,
            'n_encoder_layers': int(round(self.best_hyperparameters['n_encoder_layers'])),
            'ff_dim': int(round(self.best_hyperparameters['ff_dim']))
        }
        training_params = {
            'epochs': int(round(self.best_hyperparameters['epochs'])),
            'batch_size': int(2 ** round(self.best_hyperparameters['batch_size_power'])),
            'learning_rate': self.best_hyperparameters['learning_rate']
        }
        print(f"Decoded architecture: {model_constructor_params}")
        print(f"Training parameters: {training_params}")
        print("\nTraining Final Transformer Model with Optimal Hyperparameters")
        self.final_model = TransformerModel(
            region=self.region,
            input_shape=self.input_shape,
            **model_constructor_params
        )
        # Combine train and validation sets for final training, as best settings have already been found and there is no need for a validation set.
        X_full_train = np.concatenate((X_train, X_val))
        y_full_train = np.concatenate((y_train, y_val))
        self.final_model.train_model(
            X_full_train,
            y_full_train,
            X_val=None,
            y_val=None,
            **training_params
        )
        print("\nFinal model training complete.")

    def predict(self, X_test):
        if self.final_model:
            return self.final_model.predict(X_test)
        else:
            raise RuntimeError("Model has not been trained yet. Please call the train_model() method first.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        print(f"Evaluating final {self.__class__.__name__} for {self.region} on test data...")
        y_pred = self.predict(X_test)
        # Ensure arrays are flat for metric calculation and plotting
        y_test_flat = y_test.flatten()
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
            print(f"Created directory: {directory}")
        # Save Text Results
        results_path = os.path.join(directory, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Results for {self.__class__.__name__} on {self.region} data:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
            # Also save the best hyperparameters found by PSO
            if self.best_hyperparameters:
                f.write("\nBest Hyperparameters Found by PSO:\n")
                for key, value in self.best_hyperparameters.items():
                    f.write(f"{key}: {value}\n")

        print(f"Metrics and hyperparameters saved to {results_path}")

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

        # Time Series Comparison (zoomed in on the first 1000 points)
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

        print(f"Diagnostic plots saved to {directory}")

    def save_model(self, directory: str):
        if self.final_model:
            if not os.path.exists(directory):
                os.makedirs(directory)

            model_path = os.path.join(directory, 'best_model.pth')
            torch.save(self.final_model.state_dict(), model_path)
            print(f"✅ Best model saved to {model_path}")
        else:
            print("⚠️ Cannot save model, as it has not been trained yet.")