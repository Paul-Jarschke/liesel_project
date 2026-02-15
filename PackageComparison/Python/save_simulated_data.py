import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import json
import os

def simulate_data(seed=42):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    
    n_units, n_obs_small, n_obs_large, n_alts, n_params = 100, 5, 50, 5, 5 
    counts = jnp.array([n_obs_small] * 50 + [n_obs_large] * 50)
    
    true_mu = jnp.array([1.0, -1.0, 0.0, 0.0, -3.0])
    true_Sigma = 3.0 * jnp.eye(n_params)
    true_Sigma = true_Sigma.at[3, 4].set(1.5)
    true_Sigma = true_Sigma.at[4, 3].set(1.5)
    
    beta_dist = tfd.MultivariateNormalFullCovariance(loc=true_mu, covariance_matrix=true_Sigma)
    betas = beta_dist.sample(seed=k1, sample_shape=(n_units,))
    
    X_list, y_list = [], []
    for i in range(n_units):
        n_i, beta_i = counts[i], betas[i]
        k2, sk = jax.random.split(k2)
        price = jax.random.uniform(sk, shape=(n_i, n_alts), minval=-1.5, maxval=0.0)
        
        D = jnp.zeros((n_i, n_alts, n_params))
        for a in range(4): D = D.at[:, a, a].set(1.0)
        D = D.at[:, :, 4].set(price)
        
        logits = jnp.einsum('ijk,k->ij', D, beta_i)
        k3, sk = jax.random.split(k3)
        choices = tfd.Categorical(logits=logits).sample(seed=sk)
        
        X_list.append(D.tolist())
        y_list.append((choices + 1).tolist())
        
    return X_list, y_list, true_mu.tolist(), true_Sigma.tolist(), betas.tolist()

if __name__ == "__main__":
    X_list, y_list, mu, sigma, betas = simulate_data()
    
    # Structure data
    formatted_data = []
    for i in range(len(y_list)):
        formatted_data.append({
            "y": y_list[i],
            "X": X_list[i]
        })
        
    full_output = {
        "dataset": formatted_data,
        "truth": {"mu": mu, "sigma": sigma, "betas": betas}
    }

    # --- PATH FIX ---
    # 1. Get the absolute path of THIS script file (PackageComparison/Python/script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go UP one level from Python to PackageComparison, then DOWN into Data
    output_dir = os.path.join(script_dir, "..", "Data")
    
    # 3. Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Set final path
    output_path = os.path.join(output_dir, "simulated_data.json")
    
    with open(output_path, "w") as f:
        json.dump(full_output, f)

    # Print absolute path so you can verify exactly where it went
    print(f"File saved to: {os.path.abspath(output_path)}")