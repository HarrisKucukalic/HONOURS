import numpy as np
import pandas as pd
import pyswarms as ps
from ann import ANNModel
from transformer import TransformerModel
from sklearn.metrics import mean_squared_error
class PSO_Optimizer:
    def __init__(self, model_class, region: str, input_shape: int):
        """
        Initialises the PSO Optimiser.
        model_class: The actual model class to be optimized (e.g., ANNModel).
        region: The NEM region for which the model is being trained.
        input_shape: The number of input features for the model.
        """
        self.model_class = model_class
        self.region = region
        self.input_shape = input_shape
        # Initialize data placeholders
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.param_keys = None
        print(f"Initialised PSO Optimizer for '{self.model_class.__name__}' model.")

    def get_params_from_pos(self, position: np.ndarray) -> dict:
        """Converts a particle's position vector back to a dictionary of parameters."""
        if self.param_keys is None:
            raise RuntimeError("Parameter keys are not set. Call find_optimal_hyperparameters first.")
        return {key: val for key, val in zip(self.param_keys, position)}

    def _objective_function(self, particles) -> np.ndarray:
        """
        Evaluates each particle. This function now handles different model types.
        """
        n_particles = particles.shape[0]
        results = np.zeros(n_particles)

        for i in range(n_particles):
            params = self.get_params_from_pos(particles[i])
            model_params = {}
            training_params = {}
            if self.model_class == ANNModel:
                # Decode parameters for the ANN
                num_layers = int(np.round(params['num_layers']))
                architecture_list = []

                for j in range(num_layers):
                    # 1. Get the exponent value using the CORRECT key name
                    power = int(np.round(params[f'neurons_power_layer_{j}']))  # Use 'neurons_power_layer'

                    # 2. Calculate 2 to the power of that exponent
                    neuron_count = 2 ** power
                    architecture_list.append(neuron_count)

                model_params = {
                    'region': self.region,
                    'input_shape': self.input_shape,
                    'layer_neurons': architecture_list
                }

            elif self.model_class == TransformerModel:
                # Decode parameters for the Transformer
                # Ensure d_model is divisible by n_heads
                n_heads = 2 ** int(np.round(params['n_heads']))  # e.g., 2, 4, 8
                d_model = int(np.round(params['model_dim'] / n_heads)) * n_heads  # Make it divisible

                model_params = {
                    'region': self.region,
                    'input_shape': self.input_shape,  # Transformer expects the feature dimension
                    'model_dim': d_model,
                    'n_heads': n_heads,
                    'n_encoder_layers': int(np.round(params['n_encoder_layers'])),
                    'ff_dim': int(np.round(params['ff_dim']))
                }

            # Common training parameters
            training_params = {
                'epochs': int(np.round(params['epochs'])),
                'batch_size': int(2 ** np.round(params['batch_size_power'])),
                'learning_rate': params['learning_rate']
            }

            # --- Create, train, and evaluate the temporary model ---
            temp_model = self.model_class(**model_params)
            temp_model.train_model(self.X_train, self.y_train, self.X_val, self.y_val, **training_params)

            y_pred_val = temp_model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val.flatten(), y_pred_val.flatten()))
            results[i] = rmse

        return results

    def find_optimal_hyperparameters(self, search_space: dict, X_train, y_train, X_val, y_val,
                                     n_particles: int, iters: int) -> dict:
        """
        Uses PSO to find the best hyperparameters.
        """
        print(f"\nStarting PSO for {self.model_class.__name__}...")

        # --- Store data and parameter keys on the instance ---
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.param_keys = list(search_space.keys())

        # --- PySwarms Implementation ---
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        bounds = (
            np.array([search_space[key][0] for key in self.param_keys]),
            np.array([search_space[key][1] for key in self.param_keys])
        )
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(self.param_keys),
            options=options,
            bounds=bounds
        )

        # --- Run the optimization ---
        # The objective function can now access data via `self`
        best_cost, best_pos = optimizer.optimize(self._objective_function, iters=iters, verbose=True)

        print(f"\nPSO process finished. Best cost (validation RMSE): {best_cost:.4f}")

        # Use the new, consistent function to get the final parameters
        best_params_dict = self.get_params_from_pos(best_pos)
        print("Best hyperparameters found:")
        print(best_params_dict)

        return best_params_dict
