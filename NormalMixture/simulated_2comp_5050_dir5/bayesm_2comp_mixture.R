# ==============================================================================
# REPLICATION: Hierarchical MNL — Simulated Data — 2-Component Mixture
# Matches Python DGP: n_units=300, n_obs=30, n_alts=4, n_demos=2
#
# OUTPUT: Saves mu_draws_df, cov_draws_df, pvec_draws_df as DataFrames
#         compatible with pyreadr / load_and_format_draws() in the
#         marginal-density comparison script.
# ==============================================================================

library(jsonlite)
library(bayesm)
library(this.path)

# ── 1. Load data ───────────────────────────────────────────────────────────────
cat("Loading simulated data...\n")
script_dir <- this.path::here()
data_path <- file.path(script_dir, "simulated_hmnl_data.json")
raw <- fromJSON(data_path, simplifyVector = TRUE)

# Scalars
n_units <- raw$n_units
n_params <- raw$n_params
n_alts <- raw$n_alts
n_demos <- raw$n_demos
K_true <- raw$K

n_obs <- length(raw$y) / n_units # 30 observations per unit

cat(sprintf(
    "Loaded: %d units | %d obs/unit | %d alts | %d params | %d demos\n",
    n_units, as.integer(n_obs), n_alts, n_params, n_demos
))

# ── 2. Reconstruct lgtdata ─────────────────────────────────────────────────────
# jsonlite parses 3D arrays such that dim 1 is n_obs, dim 2 is n_alts, dim 3 is n_params.
# We aperm() to (n_alts, n_units * n_obs, n_params) so n_alts varies fastest.
X_aperm <- aperm(raw$X, c(2, 1, 3))
X_all <- matrix(X_aperm, ncol = n_params, byrow = FALSE)

y_all <- as.integer(unlist(raw$y)) + 1L # 0-indexed → 1-indexed

lgtdata <- vector("list", n_units)
rows_per_unit <- as.integer(n_obs) * n_alts

for (i in seq_len(n_units)) {
    row_start <- (i - 1L) * rows_per_unit + 1L
    row_end <- i * rows_per_unit
    obs_start <- (i - 1L) * as.integer(n_obs) + 1L
    obs_end <- i * as.integer(n_obs)

    lgtdata[[i]] <- list(
        y = y_all[obs_start:obs_end],
        X = X_all[row_start:row_end, , drop = FALSE]
    )
}

# ── 3. Z matrix (already centred by Python DGP) ───────────────────────────────
Z <- matrix(unlist(raw$Z), nrow = n_units, ncol = n_demos, byrow = FALSE)
cat("Z column means (should be ~0):", round(colMeans(Z), 6), "\n")

# ── 4. True parameters for post-run comparison ────────────────────────────────
TRUE_DELTA <- matrix(unlist(raw$TRUE_DELTA), nrow = n_demos, ncol = n_params, byrow = FALSE)
TRUE_MU_K <- matrix(unlist(raw$TRUE_MU_K), nrow = K_true, ncol = n_params, byrow = FALSE)
TRUE_PVEC <- unlist(raw$TRUE_PVEC)
TRUE_BETA <- matrix(unlist(raw$TRUE_BETA), nrow = n_units, ncol = n_params, byrow = FALSE)

param_names <- paste0("Param_", seq_len(n_params) - 1)
demo_names <- paste0("Demo_", seq_len(n_demos) - 1)

cat("\nTrue mixture probabilities:\n")
print(TRUE_PVEC)
cat("\nTrue Delta (n_demos x n_params):\n")
print(round(TRUE_DELTA, 4))

# ── 5. Model configuration ─────────────────────────────────────────────────────
n_comp <- 2
R_total <- 41000
burn_in <- 1000
keep_every <- 4

data_list <- list(p = n_alts, lgtdata = lgtdata, Z = Z)

# Prior — matched to Python build_mixture_hmnl_model
#   Delta   ~ N(0, Ad^{-1}),   Ad = 0.01 * I
#   Sigma_k^{-1} ~ W(nu, V^{-1}),  nu = n_params + 3,  V = nu * I
#   mu_k | Sigma_k ~ N(0, Sigma_k / Amu),  Amu = 0.01
#   pvec ~ Dir(5, 5)
Prior <- list(
    ncomp = n_comp,
    Ad    = 0.01 * diag(n_demos * n_params),
    nu    = n_params + 3,
    V     = (n_params + 3) * diag(n_params),
    Amu   = 0.01,
    a     = rep(5, n_comp)
)

Mcmc <- list(R = R_total, keep = keep_every, nprint = 500)

# ── 6. Run model ───────────────────────────────────────────────────────────────
cat("\nRunning rhierMnlRwMixture (2-component, Z included)...\n")
set.seed(101)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ── 7. Post-processing — discard burn-in ──────────────────────────────────────
cat("\nProcessing posterior samples...\n")
thinned_burn_in <- burn_in %/% keep_every
R_draws <- length(out$nmix$compdraw)
keep_idx <- seq(thinned_burn_in + 1L, R_draws)
n_samples <- length(keep_idx)
cat(sprintf(
    "Total thinned draws: %d | Retained after burn-in: %d\n",
    R_draws, n_samples
))

# ── 8. Extract mu and cov per component ───────────────────────────────────────
mu_list <- vector("list", n_comp)
cov_list <- vector("list", n_comp)

for (k in seq_len(n_comp)) {
    # mu: n_samples x n_params
    mu_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) x[[k]]$mu)
    )[keep_idx, , drop = FALSE]

    # cov: recover from Cholesky root (rooti); n_samples x n_params^2
    cov_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) as.vector(chol2inv(x[[k]]$rooti)))
    )[keep_idx, , drop = FALSE]
}

# ── 9. Flatten horizontally (mirrors margarine export in bayesm_output_*.R) ───
#   mu_draws_df  : (n_samples, K * n_params)        — readable by load_and_format_draws()
#   cov_draws_df : (n_samples, K * n_params^2)       — readable by load_and_format_draws()
#   pvec_draws_df: (n_samples, K)
mu_flat <- do.call(cbind, mu_list) # (n_samples, K * n_params)
cov_flat <- do.call(cbind, cov_list) # (n_samples, K * n_params^2)

# Mixture weights — keep_idx on rows already correct
pvec_raw <- out$nmix$probdraw
if (nrow(pvec_raw) == n_comp) pvec_raw <- t(pvec_raw) # ensure (R_draws x K)
pvec_flat <- pvec_raw[keep_idx, , drop = FALSE] # (n_samples, K)

# Convert to DataFrames for clean pyreadr import
mu_draws_df <- as.data.frame(mu_flat)
cov_draws_df <- as.data.frame(cov_flat)
pvec_draws_df <- as.data.frame(pvec_flat)

# Give columns sensible names (not required by load_and_format_draws, but helpful)
colnames(mu_draws_df) <- paste0(
    rep(paste0("Comp_", seq_len(n_comp)), each = n_params), "_",
    rep(param_names, times = n_comp)
)
colnames(pvec_draws_df) <- paste0("Comp_", seq_len(n_comp))

# ── 10. Delta draws ────────────────────────────────────────────────────────────
delta_draws_df <- as.data.frame(out$Deltadraw[keep_idx, , drop = FALSE])

# ── 11. Unit-level beta draws — (n_units x n_samples x n_params) ──────────────
beta_reordered <- aperm(out$betadraw[, , keep_idx, drop = FALSE], c(1, 3, 2))

# ── 12. Posterior summary (quick sanity check) ────────────────────────────────
cat("\n=== Posterior Mixture Weights ===\n")
print(round(colMeans(pvec_draws_df), 4))
cat("True pvec:", round(TRUE_PVEC, 4), "\n")

cat("\n=== Component Means (mu_k) vs Truth ===\n")
for (k in seq_len(n_comp)) {
    post_mean <- colMeans(mu_list[[k]])
    dists <- sapply(seq_len(K_true), function(j) sum((post_mean - TRUE_MU_K[j, ])^2))
    best_j <- which.min(dists)
    cat(sprintf("\n-- Component %d (matched to True Component %d) --\n", k, best_j))
    comparison <- rbind(
        Posterior_Mean = round(post_mean, 4),
        True_Value     = round(TRUE_MU_K[best_j, ], 4),
        Abs_Diff       = round(abs(post_mean - TRUE_MU_K[best_j, ]), 4)
    )
    colnames(comparison) <- param_names
    print(comparison)
}

# ── 13. Export ─────────────────────────────────────────────────────────────────
export_dir <- file.path(script_dir, "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_simulated_2comp.RData")

# Save DataFrames + ground truth for plotting
save(
    mu_draws_df, # (n_samples, K * n_params)        ← marginal densities
    cov_draws_df, # (n_samples, K * n_params^2)       ← marginal densities
    pvec_draws_df, # (n_samples, K)                    ← mixture weights
    delta_draws_df, # (n_samples, n_demos * n_params)   ← Delta recovery
    beta_reordered, # (n_units, n_samples, n_params)    ← individual betas
    TRUE_MU_K, # (K_true, n_params)                ← ground truth
    TRUE_PVEC, # (K_true,)                         ← ground truth
    TRUE_DELTA, # (n_demos, n_params)               ← ground truth
    n_samples,
    file = save_path
)

cat(sprintf(
    "\nSUCCESS: %d posterior samples saved to\n  %s\n",
    n_samples, save_path
))
cat("\nObjects saved:\n")
cat("  mu_draws_df   : shape (", n_samples, ",", n_comp * n_params, ")\n")
cat("  cov_draws_df  : shape (", n_samples, ",", n_comp * n_params^2, ")\n")
cat("  pvec_draws_df : shape (", n_samples, ",", n_comp, ")\n")
cat("  delta_draws_df: shape (", n_samples, ",", n_demos * n_params, ")\n")
cat("  beta_reordered: shape (", n_units, "x", n_samples, "x", n_params, ")\n")
