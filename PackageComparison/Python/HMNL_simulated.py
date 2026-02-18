import time
# Start timing script
start_time = time.time()

import datetime
import pickle
import json
import os

import jax.numpy as jnp
import numpy as np

import liesel.model as lsl
import liesel.goose as gs
from tensorflow_probability.substrates.jax.experimental import distributions as tfde
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

# ====================
# 1. FACTORY FUNCTIONS
# ====================
def make_wishart(df, scale_tril):
    return tfd.WishartTriL(
        df=df, 
        scale_tril=scale_tril, 
        input_output_cholesky=True, 
        validate_args=False
    )

def make_mvn_precision(loc, precision_factor):
    return tfde.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc,
        precision_factor=tf.linalg.LinearOperatorLowerTriangular(precision_factor),
        validate_args=False
    )

# ===============
# 2. DATA LOADING 
# ===============
def load_simulated_data(filepath):
    with open(filepath, 'r') as f:
        full_json = json.load(f)
    
    # Access the list of respondents under 'dataset'
    raw_data = full_json['dataset']
    
    X_list = []
    y_list = []
    unit_indices = []
    
    for i, respondent in enumerate(raw_data):
        # Simulated data X is already structured as (Obs, Alts, Params)
        # Shape: (n_choice_occasions, 5, 5)
        X_structure = np.array(respondent['X'])
        y_raw = np.array(respondent['y'])
        
        # Determine dimensions
        n_choice_occasions = len(y_raw)
        
        # ADJUST Y: 0-based indexing (simulated data is 1-based)
        y_adjusted = y_raw - 1
        
        X_list.append(X_structure)
        y_list.append(y_adjusted)
        unit_indices.extend([i] * n_choice_occasions)

    return {
        "X": jnp.array(np.concatenate(X_list, axis=0)),
        "y": jnp.array(np.concatenate(y_list, axis=0)),
        "unit_idx": jnp.array(unit_indices),
        "n_units": len(raw_data),
        "n_params": X_list[0].shape[2]
    }

# Get the path of the current script
script_location = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the json file relative to the script
json_path = os.path.join(script_location, "..", "Data", "simulated_data.json")

# Load the data
data_dict = load_simulated_data(json_path)

k_dim = int(data_dict["n_params"]) 
n_units = int(data_dict["n_units"])

print(f"Data Loaded: {n_units} units, {k_dim} parameters.")
print(f"Total Observations: {data_dict['y'].shape[0]}")

# =================
# 3. MODEL BUILDING
# =================

# Constants
nu_val = float(k_dim + 3)
V_inv_val = jnp.linalg.inv(nu_val * jnp.eye(k_dim))
Vinv_chol_val = jnp.linalg.cholesky(V_inv_val)
A_val = 0.01
zero_loc_val = jnp.zeros(k_dim)

# Parameters
sigma_inv_chol = lsl.Var.new_param(
    value=jnp.eye(k_dim),
    distribution=lsl.Dist(make_wishart, df=nu_val, scale_tril=Vinv_chol_val),
    name="sigma_inv_chol"
)

sigma_inv_chol_latent = sigma_inv_chol.transform(
    tfb.FillScaleTriL(), 
    name="sigma_inv_chol_latent"
)

mu_prec_factor = lsl.Var.new_calc(
    lambda L: jnp.sqrt(A_val) * L, 
    L=sigma_inv_chol, 
    name="mu_prec_factor"
)

mu = lsl.Var.new_param(
    value=jnp.zeros(k_dim),
    distribution=lsl.Dist(make_mvn_precision, loc=zero_loc_val, precision_factor=mu_prec_factor),
    name="mu"
)

beta_i = lsl.Var.new_param(
    value=jnp.zeros((n_units, k_dim)),
    distribution=lsl.Dist(make_mvn_precision, loc=mu, precision_factor=sigma_inv_chol),
    name="beta_i"
)

# Observation Model
X_var = lsl.Var.new_obs(data_dict["X"], name="X_obs")
idx_var = lsl.Var.new_obs(data_dict["unit_idx"], name="idx_obs")

beta_expanded = lsl.Var.new_calc(
    lambda b, idx: b[idx], 
    b=beta_i, 
    idx=idx_var, 
    name="beta_expanded"
)

logits = lsl.Var.new_calc(
    lambda x, b: jnp.einsum("nij,nj->ni", x, b), 
    x=X_var, 
    b=beta_expanded, 
    name="logits"
)

y_var = lsl.Var.new_obs(
    data_dict["y"], 
    distribution=lsl.Dist(tfd.Categorical, logits=logits), 
    name="y"
)

model = lsl.Model([y_var])

# ============
# 4. INFERENCE
# ============
eb = gs.EngineBuilder(seed=123, num_chains=4)
eb.set_model(gs.LieselInterface(model))
eb.set_initial_values(model.state)

eb.add_kernel(gs.NUTSKernel(["mu"]))
eb.add_kernel(gs.NUTSKernel(["sigma_inv_chol_latent"], mm_diag=False))
eb.add_kernel(gs.NUTSKernel(["beta_i"]))

# Duration
eb.set_duration(warmup_duration=1000, posterior_duration=10000)

print("Starting Sampling...")
engine = eb.build()
engine.sample_all_epochs()

results = engine.get_results()
samples = results.get_posterior_samples()

# ===============
# 5. SAVE RESULTS
# ===============

# Stop the timer
end_time = time.time()
total_seconds = end_time - start_time
formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
print(f"Script finished in {formatted_time} (H:M:S)")

# ---------------------------------------------------------
# Dynamic Filename Logic
# ---------------------------------------------------------
# Calculate total samples (Chains * Posterior Duration)
n_chains_actual = samples['mu'].shape[0]
n_iter_actual = samples['mu'].shape[1]
total_samples_count = n_chains_actual * n_iter_actual

data_to_save = {
    "results": results,
    "samples": samples,
    "runtime_seconds": total_seconds,
    "runtime_formatted": formatted_time,
    "n_samples": total_samples_count
}

# Construct directory and filename
export_dir = os.path.join(script_location, "..", "Data")
os.makedirs(export_dir, exist_ok=True)

filename = f"liesel_output_simulated_{total_samples_count}_samples.pkl"
liesel_data_path = os.path.join(export_dir, filename)

# Save
with open(liesel_data_path, "wb") as f:
    pickle.dump(data_to_save, f)

print(f"Saved {filename} ({total_samples_count} total draws across {n_chains_actual} chains).")