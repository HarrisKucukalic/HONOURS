import numpy as np
import pandas as pd
import pyswarms as ps


class PSO_Optimizer:
    def __init__(self, model_name: str):
        """
        Initialises the PSO Optimiser.
        model_name: The name of the model to be optimized (e.g., "ann", "transformer").
        """
        self.model_name = model_name
        self.best_params_dict = None
        print(f"Initialised PSO Optimizer for '{model_name}' model.")

    def _objective_function(self, swarm_params):
        """
        Objective function for PSO to minimize.
        In a real scenario, this function would train and evaluate a model.
        For this demonstration, it calculates a simple mathematical cost (Sphere function).
        """
        all_losses = []
        for particle_params in swarm_params:
            # In a real implementation, you would unpack params, train a model,
            # and calculate the validation loss (e.g., RMSE).

            # For this example, we just calculate a simple cost.
            # The Sphere function (sum of squares) is a common benchmark.
            # The optimizer will try to find the parameters that result in the lowest cost (zero).
            cost = np.sum(particle_params ** 2)
            all_losses.append(cost)

        return np.array(all_losses)

    def _unpack_params(self, params_array):
        """Helper function to convert a numpy array of params back to a named dictionary."""
        # The order of keys here MUST match the order in your search_space dictionary
        param_keys = list(self.search_space_keys)

        unpacked_params = {}
        for i, key in enumerate(param_keys):
            # Handle integer hyperparameters by checking for common naming conventions
            if any(s in key for s in ['n_', '_dim', '_layers', 'batch_size', 'max_depth', 'min_samples']):
                unpacked_params[key] = int(params_array[i])
            else:
                unpacked_params[key] = params_array[i]
        return unpacked_params

    def find_optimal_hyperparameters(self, search_space, n_particles, iters):
        """
        Uses PSO to find the best hyperparameters.

        search_space: A dictionary where keys are hyperparameter names and values are tuples of (min, max).
        n_particles: The number of particles in the swarm.
        iters: The number of iterations to run the optimization.
        """
        if ps is None:
            raise ImportError("PySwarms is not installed. Please run 'pip install pyswarms'.")

        # Store search space keys for the unpacking function to access
        self.search_space_keys = search_space.keys()

        print(f"Starting PSO to find optimal hyperparameters for {self.model_name} model...")

        # --- PySwarms Implementation ---
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Extract bounds from the search space dictionary
        min_bounds = [v[0] for v in search_space.values()]
        max_bounds = [v[1] for v in search_space.values()]
        bounds = (np.array(min_bounds), np.array(max_bounds))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(search_space),
            options=options,
            bounds=bounds
        )

        # Run the optimization
        best_cost, best_pos = optimizer.optimize(self._objective_function, iters=iters, verbose=True)

        print(f"\nPSO process finished. Best cost: {best_cost:.4f}")

        # Convert the best position array back to a named dictionary
        self.best_params_dict = self._unpack_params(best_pos)
        print("Best hyperparameters found:")
        print(self.best_params_dict)

        # Clean up stored data
        del self.search_space_keys

        return self.best_params_dict