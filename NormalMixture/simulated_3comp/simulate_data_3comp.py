import numpy as np
import pandas as pd
import os
import json
import jax.numpy as jnp

def generate_mixture_simulated_data(
        n_units=2000, n_obs=100, n_alts=4, n_components=3, n_params=None,
        n_demos=2, custom_pvec=None, custom_indicators=None, 
        custom_mu_k=None, seed=123):
    
    np.random.seed(seed)

    if n_params is None:
        n_params = n_alts

    if n_params < n_alts - 1:
        raise ValueError(f"n_params ({n_params}) must be at least n_alts - 1")

    n_ascs = n_alts - 1
    n_continuous = n_params - n_ascs

    # Generate Demographic Data (Z)
    Z = np.random.normal(0, 1, size=(n_units, n_demos))
    Z = Z - np.mean(Z, axis=0)

    # Generate TRUE Global Parameters (Delta)
    Delta_true = np.random.normal(0, 0.5, size=(n_demos, n_params))

    # Generate TRUE Mixture Probabilities
    if custom_pvec is not None:
        true_pvec = np.array(custom_pvec)
        true_pvec = true_pvec / np.sum(true_pvec)
    else:
        raw_p = np.random.uniform(0.5, 2.0, n_components)
        true_pvec = raw_p / np.sum(raw_p)

    # --- NEW: Generate or Assign TRUE Mixture Means (mu_k) ---
    if custom_mu_k is not None:
        true_mu_k = np.array(custom_mu_k)
        if true_mu_k.shape != (n_components, n_params):
            raise ValueError(f"custom_mu_k must have shape ({n_components}, {n_params})")
    else:
        true_mu_k = np.random.normal(0, 2.0, size=(n_components, n_params))
        
    # Generate TRUE Mixture Covariance Matrices (Sigma_k)
    true_Sigma_k = np.zeros((n_components, n_params, n_params))
    for k in range(n_components):
        true_Sigma_k[k] = np.diag(np.random.uniform(0.5, 2.0, n_params))

    # Generate Individual-Level Parameters (beta_i)
    beta_true = np.zeros((n_units, n_params))
    if custom_indicators is not None:
        true_indicators = np.array(custom_indicators)
    else:
        true_indicators = np.random.choice(n_components, size=n_units, p=true_pvec)

    for i in range(n_units):
        k = true_indicators[i]
        mu_i = Z[i] @ Delta_true + true_mu_k[k]
        beta_true[i] = np.random.multivariate_normal(mu_i, true_Sigma_k[k])

    # Generate Choice Data (X, y)
    X_list, y_list, unit_idx_list = [], [], []

    for i in range(n_units):
        for t in range(n_obs):
            X_it = np.zeros((n_alts, n_params))
            for a in range(1, n_alts):
                X_it[a, a - 1] = 1.0 # Set ASCs

            if n_continuous > 0:
                X_it[:, n_ascs:] = np.random.uniform(1.0, 5.0, size=(n_alts, n_continuous))

            U_it = X_it @ beta_true[i]
            exp_U = np.exp(U_it - np.max(U_it)) # Log-sum-exp trick for numerical stability
            probs = exp_U / np.sum(exp_U)
            y_it = int(np.random.choice(n_alts, p=probs))

            X_list.append(X_it)
            y_list.append(y_it)
            unit_idx_list.append(i)

    return {
        "X": jnp.array(X_list),
        "y": jnp.array(y_list),
        "Z": jnp.array(Z),
        "unit_idx": jnp.array(unit_idx_list),
        "n_units": n_units,
        "n_params": n_params,
        "n_demos": n_demos,
        "K": n_components,
        "n_alts": n_alts,
        "TRUE_DELTA": Delta_true,
        "TRUE_BETA": beta_true,
        "TRUE_PVEC": true_pvec,
        "TRUE_MU_K": true_mu_k,
        "TRUE_SIGMA_K": true_Sigma_k,
        "TRUE_INDICATORS": true_indicators
    }

# --- SAVE TO JSON LOGIC ---
def save_to_json(data, filename="sim_data.json"):
    """Converts arrays to lists and saves to JSON."""
    
    def convert_recursive(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        return obj

    serializable_data = convert_recursive(data)
    
    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)
    print(f"Data successfully saved to {filename}")

if __name__ == "__main__":
    print("Simulating data for 3 Latent Segments...")
    
    # Define 3 distinct consumer profiles based on 4 parameters: [ASC_1, ASC_2, ASC_3, Price]
    
    custom_mu = [
        # Segment 0: "Brand Loyalists" (Weight: ~40%)
        # Characteristics: High baseline utility for Alt 1 (2.0), moderate price sensitivity (-0.5). 
        # They routinely pick Alt 1 unless it is heavily overpriced.
        [ 2.0, -1.0, -1.0, -0.5], 
        
        # Segment 1: "Bargain Hunters" (Weight: ~35%)
        # Characteristics: Highly price-sensitive (-3.0), no strong brand preferences (negative ASCs).
        # They consistently choose the cheapest option available.
        [-1.0, -1.0, -1.0, -3.0], 
        
        # Segment 2: "Premium / Quality Seekers" (Weight: ~25%)
        # Characteristics: Almost completely price-insensitive (-0.1), strong preference for Alt 3 (1.0).
        # They view Alt 3 as the premium choice and are willing to pay for it.
        [ 0.5,  0.5,  1.0, -0.1]  
    ]
    
    sim_data = generate_mixture_simulated_data(
        n_units=300, 
        n_obs=30, 
        n_alts=4, 
        n_components=3, 
        n_demos=2, 
        custom_pvec=[0.40, 0.35, 0.25], 
        custom_mu_k=custom_mu,
        seed=101
    )
    
    # 1. Get the absolute path of the folder containing this .py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Attach the filename to that specific folder path
    save_path = os.path.join(script_dir, "simulated_hmnl_data_3comp.json")
    
    # 3. Save the data to that exact path
    save_to_json(sim_data, save_path)
    