import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import liesel.model as lsl
import liesel.goose as gs
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.experimental import distributions as tfde
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf


# --- Helper Functions ---

def _make_wishart(df: float, scale_tril: jnp.ndarray):
    """
    Creates a Wishart distribution parameterized by its Cholesky factor.
    
    Args:
        df: Degrees of freedom for the Wishart distribution.
        scale_tril: Lower triangular Cholesky factor of the scale matrix.
        
    Returns:
        A TensorFlow Probability WishartTriL distribution object.
    """
    return tfd.WishartTriL(
        df=df, 
        scale_tril=scale_tril, 
        input_output_cholesky=True, 
        validate_args=False
    )

def _make_mvn_precision(loc: jnp.ndarray, precision_factor: jnp.ndarray):
    """
    Creates a Multivariate Normal distribution parameterized by its precision factor.
    
    Args:
        loc: The mean vector of the distribution.
        precision_factor: The lower triangular Cholesky factor of the precision matrix.
        
    Returns:
        A TFP experimental MVN distribution object.
    """
    return tfde.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc, 
        precision_factor=tf.linalg.LinearOperatorLowerTriangular(precision_factor), 
        validate_args=False
    )


# --- Main Model Builder ---

def build_hbmnl_model(data_dict: dict, A_val: float = 0.01) -> lsl.Model:
    """
    Constructs the Hierarchical Bayesian Multinomial Logit (HBMNL) model.
    
    This model accounts for unobserved heterogeneity among units (e.g., consumers) 
    and optionally incorporates demographic data (Z) to shift baseline utilities.
    
    Args:
        data_dict: A dictionary containing:
            - 'X': Covariates (features) for the choice tasks.
            - 'y': Observed choices.
            - 'n_params': Number of parameters/coefficients.
            - 'n_units': Number of unique decision-makers.
            - 'unit_idx': Array mapping each observation to a specific unit.
            - 'Z' (optional): Demographic coavriates of the decision-making units.
        A_val: Prior scaling parameter for the baseline utility mean. Defaults to 0.01.
        
    Returns:
        A compiled Liesel Model ready to be passed to the Goose engine.
    """
    
    # Extract basic dimensions
    k_dim = int(data_dict["n_params"])   # Number of choice parameters
    n_units = int(data_dict["n_units"])  # Number of unique decision-makers
    has_Z = data_dict.get("Z") is not None # Check if demographic data is provided

    # Set up priors for the Wishart distribution (nu = k + 3, V_inv = (nu * I)^-1)
    nu_val = float(k_dim + 3)
    V_inv_val = jnp.linalg.inv(nu_val * jnp.eye(k_dim))
    Vinv_chol_val = jnp.linalg.cholesky(V_inv_val)

    # 1. Unobserved Heterogeneity Covariance (Sigma)
    # We parameterize using the inverse Cholesky factor for computational stability
    sigma_inv_chol = lsl.Var.new_param(
        value=jnp.eye(k_dim),
        distribution=lsl.Dist(_make_wishart, df=nu_val, scale_tril=Vinv_chol_val),
        name="sigma_inv_chol"
    )
    
    # Apply a bijector to ensure the sampled matrix remains a valid lower-triangular Cholesky factor
    sigma_inv_chol_latent = sigma_inv_chol.transform(tfb.FillScaleTriL(), name="sigma_inv_chol_latent")
    
    # Scale the precision factor for the population mean prior
    mu_prec_factor = lsl.Var.new_calc(lambda L: jnp.sqrt(A_val) * L, L=sigma_inv_chol, name="mu_prec_factor")

    # 2. Baseline Utilities (Mu or Delta)
    # If demographics (Z) exist, baseline utility is shifted by Z * Delta. 
    # Otherwise, it's just a global mean (Mu) applied to everyone.
    if has_Z:
        n_demos = data_dict["Z"].shape[1]
        Z_var = lsl.Var.new_obs(data_dict["Z"], name="Z_obs")
        
        global_param = lsl.Var.new_param(
            value=jnp.zeros((n_demos, k_dim)),
            distribution=lsl.Dist(_make_mvn_precision, loc=jnp.zeros(k_dim), precision_factor=mu_prec_factor),
            name="Delta"
        )
        
        # Compute the location (mean) for each unit based on their demographics
        loc_var = lsl.Var.new_calc(lambda z, delta: jnp.dot(z, delta), z=Z_var, delta=global_param, name="beta_loc")
    else:
        global_param = lsl.Var.new_param(
            value=jnp.zeros(k_dim),
            distribution=lsl.Dist(_make_mvn_precision, loc=jnp.zeros(k_dim), precision_factor=mu_prec_factor),
            name="mu"
        )
        
        # Global mean is identical for all units
        loc_var = global_param 

    # 3. Unit-Level Coefficients (Beta)
    # Each unit draws its specific parameters from the population distribution
    beta_i = lsl.Var.new_param(
        value=jnp.zeros((n_units, k_dim)),
        distribution=lsl.Dist(_make_mvn_precision, loc=loc_var, precision_factor=sigma_inv_chol),
        name="beta_i"
    )

    # 4. Likelihood / Choice Model
    X_var = lsl.Var.new_obs(data_dict["X"], name="X_obs")
    idx_var = lsl.Var.new_obs(data_dict["unit_idx"], name="idx_obs")
    
    # Map unit-level betas up to the observation level (one row per choice task)
    beta_expanded = lsl.Var.new_calc(lambda b, idx: b[idx], b=beta_i, idx=idx_var, name="beta_expanded")
    
    # Calculate utility logits: jnp.einsum handles the batched dot product of X and Beta
    logits = lsl.Var.new_calc(lambda x, b: jnp.einsum("nij,nj->ni", x, b), x=X_var, b=beta_expanded, name="logits")
    
    # Observed categorical choices depend on the calculated logits
    y_var = lsl.Var.new_obs(data_dict["y"], distribution=lsl.Dist(tfd.Categorical, logits=logits), name="y")

    return lsl.Model([y_var])# --- Inference Runner ---


def run_inference_hbmnl(model: lsl.Model, data_dict: dict, chains: int = 1, warmup: int = 1000, posterior: int = 5000, seed: int = 123):
    """
    Sets up and runs the Liesel/Goose MCMC engine for the HBMNL model.
    
    This function configures a block-wise NUTS (No-U-Turn Sampler) strategy. 
    By updating the parameters in a specific hierarchical order, we avoid 
    the pathological geometry often associated with joint sampling in 
    hierarchical models, leading to much better mixing and fewer divergences.
    
    Args:
        model: The compiled Liesel Model object (returned by build_hbmnl_model).
        data_dict: The dataset dictionary (used here to dynamically check for demographics).
        chains: Number of independent MCMC chains to run. Defaults to 1.
        warmup: Number of warmup (burn-in) steps for NUTS adaptation. Defaults to 1000.
        posterior: Number of posterior samples to draw per chain after warmup. Defaults to 10000.
        seed: Random seed for reproducibility. Defaults to 123.
        
    Returns:
        A tuple containing:
            - The full EngineResults object (contains traces, diagnostics, etc.).
            - A dictionary of just the extracted posterior samples for convenience.
    """
    
    # 1. Initialize the Goose Engine Builder
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    # Dynamically determine the name of the baseline utility parameter
    global_param_name = "Delta" if data_dict.get("Z") is not None else "mu"
    
    # 2. Kernel Configuration
    
    # Step A: Update the unobserved heterogeneity covariance (latent Cholesky factor)
    # mm_diag=True only used a diagonal mass matrix, which reduces computational costs when compared to using a full matrix
    eb.add_kernel(gs.NUTSKernel(["sigma_inv_chol_latent"], mm_diag=True))
    
    # Step B: Update the population-level means / demographic shifts
    eb.add_kernel(gs.NUTSKernel([global_param_name])) 
    
    # Step C: Update the unit-level (individual consumer) coefficients
    eb.add_kernel(gs.NUTSKernel(["beta_i"]))

    # 3. Finalize and Run Engine
    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print(f"Starting NUTS Sampling...")
    print(f" - Targeting global parameter: {global_param_name}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")
    
    engine = eb.build()
    engine.sample_all_epochs()

    return engine.get_results(), engine.get_results().get_posterior_samples()