import time
import datetime
import json
import os

import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd

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
def load_margarine_data(filepath):
    """Loads and preprocesses the margarine dataset."""
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    
    X_list, y_list, Z_list, unit_indices = [], [], [], []
    
    n_alts_orig = 10
    kept_alts_1based = np.array([1, 2, 3, 4, 5, 7])
    kept_alts_0based = kept_alts_1based - 1 
    p = len(kept_alts_0based) 
    y_mapping = {val: idx for idx, val in enumerate(kept_alts_1based)}
    unit_idx_counter = 0
    
    for respondent in raw_data:
        X_raw = np.array(respondent['X'])
        y_raw = np.array(respondent['y'])
        
        if y_raw.ndim == 0:
            y_raw = np.atleast_1d(y_raw)
        
        n_choice_occasions_orig = len(y_raw)
        n_params_orig = X_raw.shape[1]
        X_reshaped = X_raw.reshape((n_choice_occasions_orig, n_alts_orig, n_params_orig))
        
        valid_occasions = np.isin(y_raw, kept_alts_1based)
        y_filtered = y_raw[valid_occasions]
        X_filtered = X_reshaped[valid_occasions]
        
        n_obs = len(y_filtered)
        if n_obs < 5:
            continue
            
        Z_raw = np.array(respondent.get('Z_raw', [1.0, 1.0]))
        Z_list.append(Z_raw)
        y_recoded = np.array([y_mapping[val] for val in y_filtered])
        X_new = np.zeros((n_obs, p, p))
        
        for r in range(n_obs):
            orig_prices = X_filtered[r, kept_alts_0based, 9]
            log_prices = np.log(orig_prices)
            X_new[r, 1:, 0:5] = np.eye(5)
            X_new[r, :, 5] = log_prices
            
        X_list.append(X_new)
        y_list.append(y_recoded)
        unit_indices.extend([unit_idx_counter] * n_obs)
        unit_idx_counter += 1

    Z_arr = np.array(Z_list)
    Z_arr[:, 0] = np.log(Z_arr[:, 0])
    
    Z_intercept = np.ones((Z_arr.shape[0], 1))
    Z_final = np.concatenate([Z_intercept, Z_arr], axis=1)

    return {
        "X": jnp.array(np.concatenate(X_list, axis=0)),
        "y": jnp.array(np.concatenate(y_list, axis=0)),
        "Z": jnp.array(Z_final),
        "unit_idx": jnp.array(unit_indices),
        "n_units": unit_idx_counter,
        "n_params": p,
        "n_z": Z_final.shape[1]
    }

# =================
# 3. MODEL BUILDING
# =================
def build_hmnl_model(data_dict, A_val=0.01):
    """Constructs the Hierarchical MNL model graph using Liesel."""
    k_dim = int(data_dict["n_params"]) 
    n_units = int(data_dict["n_units"])
    n_z = int(data_dict["n_z"])

    # Constants
    nu_val = float(k_dim + 3)
    V_inv_val = jnp.linalg.inv(nu_val * jnp.eye(k_dim))
    Vinv_chol_val = jnp.linalg.cholesky(V_inv_val)

    # Precision matrix prior
    sigma_inv_chol = lsl.Var.new_param(
        value=jnp.eye(k_dim),
        distribution=lsl.Dist(make_wishart, df=nu_val, scale_tril=Vinv_chol_val),
        name="sigma_inv_chol"
    )

    sigma_inv_chol_latent = sigma_inv_chol.transform(
        tfb.FillScaleTriL(), 
        name="sigma_inv_chol_latent"
    )

    # Delta prior
    delta_prec_factor = lsl.Var.new_calc(
        lambda L: jnp.sqrt(A_val) * L, 
        L=sigma_inv_chol, 
        name="delta_prec_factor"
    )

    Delta = lsl.Var.new_param(
        value=jnp.zeros((n_z, k_dim)),
        distribution=lsl.Dist(
            make_mvn_precision, 
            loc=jnp.zeros(k_dim), 
            precision_factor=delta_prec_factor
        ),
        name="Delta"
    )

    # Beta mean and individual Betas
    Z_var = lsl.Var.new_obs(data_dict["Z"], name="Z_obs")
    beta_mean = lsl.Var.new_calc(
        lambda Z, D: jnp.dot(Z, D),
        Z=Z_var,
        D=Delta,
        name="beta_mean"
    )

    beta_i = lsl.Var.new_param(
        value=jnp.zeros((n_units, k_dim)),
        distribution=lsl.Dist(make_mvn_precision, loc=beta_mean, precision_factor=sigma_inv_chol),
        name="beta_i"
    )

    # Observation Level
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

    return lsl.Model([y_var])

# ============
# 4. INFERENCE
# ============
def run_inference(model, warmup=1000, posterior=10000, seed=123):
    """Sets up the MCMC engine and samples from the model."""
    eb = gs.EngineBuilder(seed=seed, num_chains=1)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    # Kernels for sampling
    eb.add_kernel(gs.NUTSKernel(["sigma_inv_chol_latent"], mm_diag=False))
    eb.add_kernel(gs.NUTSKernel(["Delta"])) 
    eb.add_kernel(gs.NUTSKernel(["beta_i"]))

    # Duration
    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting Sampling...")
    engine = eb.build()
    engine.sample_all_epochs()

    return engine.get_results().get_posterior_samples()

def print_results_tables(samples, k_dim):
    """Generates and prints Tables 5.1 and 5.2 from the posterior samples."""
    
    param_names = ["Blue Bonnet", "Fleischmanns", "House", "Generic", "Shed Spread", "LogPrice"]
    demo_names = ["Intercept", "log(Income)", "Family size"]

    # ==========================================
    # REPRODUCING TABLE 5.2: Posterior of Delta
    # ==========================================
    delta_draws = samples["Delta"].reshape(-1, len(demo_names), k_dim)

    delta_mean = np.mean(delta_draws, axis=0)
    delta_std = np.std(delta_draws, axis=0)

    print("\n=== TABLE 5.2: Posterior distribution of Delta ===")
    delta_df = pd.DataFrame(index=demo_names, columns=param_names)

    for i in range(len(demo_names)):
        for j in range(k_dim):
            delta_df.iloc[i, j] = f"{delta_mean[i, j]: .2f} ({delta_std[i, j]:.2f})"

    print(delta_df)
    print("\n")

    # ===============================================
    # REPRODUCING TABLE 5.1: Covariance / Correlation
    # ===============================================
    latent_samples = samples["sigma_inv_chol_latent"].reshape(-1, int(k_dim * (k_dim + 1) / 2))
    bijector = tfb.FillScaleTriL()

    def latent_to_cov(latent_vec):
        L = bijector.forward(latent_vec)
        precision_mat = L @ L.T
        return jnp.linalg.inv(precision_mat)

    v_latent_to_cov = jax.vmap(latent_to_cov)
    cov_draws = np.array(v_latent_to_cov(latent_samples)) 

    n_draws = cov_draws.shape[0]
    
    std_draws = np.sqrt(np.diagonal(cov_draws, axis1=1, axis2=2))
    outer_stds = std_draws[:, :, None] * std_draws[:, None, :]
    corr_draws = cov_draws / outer_stds

    std_mean = np.mean(std_draws, axis=0)
    std_sd = np.std(std_draws, axis=0)

    corr_mean = np.mean(corr_draws, axis=0)
    corr_sd = np.std(corr_draws, axis=0)

    print("=== TABLE 5.1: Correlations and standard deviations of betas ===")
    table_5_1_df = pd.DataFrame(index=param_names, columns=param_names)

    for i in range(k_dim):
        for j in range(k_dim):
            if i == j:
                table_5_1_df.iloc[i, j] = f"{std_mean[i]:.2f} ({std_sd[i]:.2f})"
            elif j > i:
                table_5_1_df.iloc[i, j] = f"{corr_mean[i, j]:.2f} ({corr_sd[i, j]:.2f})"
            else:
                table_5_1_df.iloc[i, j] = ""

    print(table_5_1_df)
    print("\n")


if __name__ == "__main__":
    start_time = time.time()
    
    # Setup paths and load data
    script_location = os.path.abspath(os.getcwd())
    json_path = os.path.join(script_location, "PackageComparison", "Data", "margarine_data_new.json")
    
    print("Loading data...")
    data_dict = load_margarine_data(json_path)
    
    # EXTRACT k_dim HERE SO IT CAN BE PASSED TO THE PRINT FUNCTION
    k_dim = int(data_dict["n_params"])
    
    print(f"Data Loaded: {data_dict['n_units']} units, {k_dim} parameters, {data_dict['n_z']} demographic vars")
    print(f"Total Observations: {data_dict['y'].shape[0]}")
    
    # Build model and run MCMC
    print("\nBuilding model...")
    model = build_hmnl_model(data_dict)
    
    samples = run_inference(model, warmup=1000, posterior=10000)
    
    # Generate tables
    print("\nGenerating Output Tables...")
    print_results_tables(samples, k_dim)
    
    # Timing
    end_time = time.time()
    formatted_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print(f"Script finished in {formatted_time} (H:M:S)")