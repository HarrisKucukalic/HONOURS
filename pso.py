import numpy as np
import pandas as pd
class PSO_Optimizer:
    """A generic PSO optimizer for hyperparameter tuning of PyTorch models."""

    def __init__(self, model_class, region: str, input_shape: int):
        """
        Initializes the PSO Optimizer.
        model_class: The model class to be optimized (e.g., ANNModel).
        region: The NEM region.
        input_shape: The number of input features for the model.
        """
        self.model_class = model_class
        self.region = region
        self.input_shape = input_shape
        self.best_params = None
        print(f"Initialized PSO Optimizer for {model_class.__name__} in {region}.")

    def _objective_function(self, swarm_params):
        """
        Objective function for PSO to minimize. Calculates RMSE for a batch of particles.
        """
        all_losses = []
        for params in swarm_params:
            # --- Unpack hyperparameters ---
            # This part needs to be adapted based on the model and search space
            # Example for ANNModel:
            hyperparams = {
                'learning_rate': params[0],
                'batch_size': int(params[1]),
                'n_layer1': int(params[2]),
                'n_layer2': int(params[3]),
                'n_layer3': int(params[4]),
            }

            # --- Build and Train a temporary model ---
            temp_model = self.model_class(
                region=self.region,
                input_shape=self.input_shape,
                n_layer1=hyperparams['n_layer1'],
                n_layer2=hyperparams['n_layer2'],
                n_layer3=hyperparams['n_layer3']
            )

            temp_model.train_model(
                self.X_train,
                self.y_train,
                epochs=10,  # Fewer epochs for faster optimization
                batch_size=hyperparams['batch_size'],
                learning_rate=hyperparams['learning_rate']
            )

            # --- Evaluate the model ---
            predictions = temp_model.predict(self.X_val)
            mse = np.mean((self.y_val - predictions.flatten()) ** 2)
            rmse = np.sqrt(mse)
            all_losses.append(rmse)

        return np.array(all_losses)

    def find_optimal_hyperparameters(self, X_train, y_train, X_val, y_val, search_space, n_particles, iters):
        """
        Uses PSO to find the best hyperparameters.

        search_space: A dictionary with 'min' and 'max' keys for the bounds of each hyperparameter.
        """
        # Store data as class attributes for the objective function to access
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

        print(f"Starting PSO to find optimal hyperparameters for {self.model_class.__name__}...")

        # --- PSO Implementation Placeholder ---
        # You would use a library like pyswarms here.
        # import pyswarms as ps
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(search_space['min']), options=options, bounds=(search_space['min'], search_space['max']))
        # best_cost, best_params = optimizer.optimize(self._objective_function, iters=iters)
        # self.best_params = best_params
        # print(f"PSO process finished. Best cost (RMSE): {best_cost}")
        # --- End Placeholder ---

        # Clean up stored data
        del self.X_train, self.y_train, self.X_val, self.y_val

        # For demonstration, returning a dummy dictionary
        print("Demonstration complete. Returning dummy parameters.")
        self.best_params = {'learning_rate': 0.001, 'batch_size': 32, 'n_layer1': 128, 'n_layer2': 64, 'n_layer3': 32}
        return self.best_params