import json
import numpy as np
import jax.numpy as jnp

def load_camera_data(filepath: str) -> dict:
    """
    Loads and preprocesses the camera dataset for the HBMNL model.
    
    This function reads the raw JSON data, extracts the choice occasions
    for each respondent, reshapes the covariates, and adjusts the observed 
    choices to be 0-indexed for JAX/TFP categorical distributions.
    
    Args:
        filepath: The path to the JSON file containing the camera data.
        
    Returns:
        A dictionary containing:
            - 'X': Reshaped covariate array of shape (total_obs, n_alts, n_params).
            - 'y': Adjusted 0-indexed observed choices array of shape (total_obs,).
            - 'unit_idx': Array mapping each observation to a specific unit.
            - 'n_units': Total number of unique respondents.
            - 'n_params': Number of covariate parameters.
    """
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    
    X_list, y_list, unit_indices = [], [], []
    n_alts = 5  # Canon, Sony, Nikon, Panasonic, Fuji
    
    for i, respondent in enumerate(raw_data):
        X_raw = np.array(respondent['X'])
        y_raw = np.array(respondent['y'])
        
        n_choice_occasions = len(y_raw)
        n_params = X_raw.shape[1]
        
        # Reshape X: (Obs, Alts, Params)
        X_reshaped = X_raw.reshape((n_choice_occasions, n_alts, n_params))
        
        # Adjust Y: 0-based indexing for Liesel/JAX Categorical
        y_adjusted = y_raw - 1
        
        X_list.append(X_reshaped)
        y_list.append(y_adjusted)
        unit_indices.extend([i] * n_choice_occasions)

    return {
        "X": jnp.array(np.concatenate(X_list, axis=0)),
        "y": jnp.array(np.concatenate(y_list, axis=0)),
        "unit_idx": jnp.array(unit_indices),
        "n_units": len(raw_data),
        "n_params": X_list[0].shape[2]
    }