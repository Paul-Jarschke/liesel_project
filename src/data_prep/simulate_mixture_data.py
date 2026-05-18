import numpy as np
import pandas as pd
import os
import json
import jax.numpy as jnp

def generate_mixture_simulated_data(
        n_units=2000, n_obs=100, n_alts=4, n_components=2, n_params=None,
        n_demos=2, custom_pvec=None, custom_indicators=None, seed=123):
    
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

    # Generate TRUE Global & Mixture Parameters
    Delta_true = np.random.normal(0, 0.5, size=(n_demos, n_params))

    if custom_pvec is not None:
        true_pvec = np.array(custom_pvec)
        true_pvec = true_pvec / np.sum(true_pvec)
    else:
        raw_p = np.random.uniform(0.5, 2.0, n_components)
        true_pvec = raw_p / np.sum(raw_p)

    true_mu_k = np.random.normal(0, 2.0, size=(n_components, n_params))
    true_Sigma_k = np.zeros((n_components, n_params, n_params))
    for k in range(n_components):
        true_Sigma_k[k] = np.diag(np.random.uniform(0.5, 2.0, n_params))

    # Generate Individual-Level Parameters
    beta_true = np.zeros((n_units, n_params))
    if custom_indicators is not None:
        true_indicators = np.array(custom_indicators)
    else:
        true_indicators = np.random.choice(n_components, size=n_units, p=true_pvec)

    for i in range(n_units):
        k = true_indicators[i]
        mu_i = Z[i] @ Delta_true + true_mu_k[k]
        beta_true[i] = np.random.multivariate_normal(mu_i, true_Sigma_k[k])

    # Generate Choice Data
    X_list, y_list, unit_idx_list = [], [], []

    for i in range(n_units):
        for t in range(n_obs):
            X_it = np.zeros((n_alts, n_params))
            for a in range(1, n_alts):
                X_it[a, a - 1] = 1.0

            if n_continuous > 0:
                X_it[:, n_ascs:] = np.random.uniform(1.0, 5.0, size=(n_alts, n_continuous))

            U_it = X_it @ beta_true[i]
            exp_U = np.exp(U_it - np.max(U_it))
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
    
    # 1. Ensure the target directory exists before trying to write to it
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)
    print(f"Data successfully saved to:\n{os.path.abspath(filename)}")

if __name__ == "__main__":
    import os

    # 1. Define all simulation parameters explicitly
    N_UNITS = 500
    N_OBS = 30
    N_ALTS = 4
    N_COMPS = 5
    N_DEMOS = 2
    SEED = 101
    CUSTOM_PVEC = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("Simulating data...")
    sim_data = generate_mixture_simulated_data(
        n_units=N_UNITS, 
        n_obs=N_OBS, 
        n_alts=N_ALTS, 
        n_components=N_COMPS,
        n_demos=N_DEMOS, 
        custom_pvec=CUSTOM_PVEC, 
        seed=SEED
    )
    
    # 2. Construct a dynamic filename containing the input parameters
    # E.g., "sim_data_U300_O30_A4_K2_D2_seed101.json"
    pvec_str = "".join([str(int(p * 100)) for p in CUSTOM_PVEC]) if CUSTOM_PVEC else "random"
    filename = f"sim_data_U{N_UNITS}_O{N_OBS}_A{N_ALTS}_K{N_COMPS}_D{N_DEMOS}_pvec{pvec_str}_seed{SEED}.json"
    
    # 3. Get the absolute path of this script (src/data_prep/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 4. Step back two levels to find the project root (liesel_project/)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # 5. Build the target path
    save_path = os.path.join(project_root, "data", "simulated", filename)
    
    # 6. Save the data
    save_to_json(sim_data, save_path)