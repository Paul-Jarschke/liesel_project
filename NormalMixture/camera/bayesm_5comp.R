# ==============================================================================
# REPLICATION: Hierarchical MNL for Camera Dataset - 5-Component Mixture
# ==============================================================================

library(this.path)
library(bayesm)

# SETUP & DATA LOADING
print("Loading camera data...")
data(camera)

lgtdata <- camera
n_units <- length(lgtdata)
p <- 5 # 5 alternatives: Canon, Sony, Nikon, Panasonic, Fuji
k_dim <- ncol(lgtdata[[1]]$X) # 10 parameters
n_comp <- 5 # number of mixture components

print(paste("Loaded", n_units, "households with", k_dim, "parameters."))

# MODEL CONFIGURATION
R_total <- 41000
burn_in <- 1000
keep_every <- 4

data_list <- list(p = p, lgtdata = lgtdata)

# Prior Config — 5-component mixture
# A is still the prior precision on Delta (intercept); remains 1x1 when Z = NULL.
# nu / V govern the Wishart prior on each component's Sigma.
# alphaDraw (Dirichlet) is initialised uniformly by bayesm when omitted.
Prior <- list(
    ncomp = n_comp,
    A     = matrix(0.01), # 1×1 — no Z covariates
    nu    = k_dim + 3,
    V     = (k_dim + 3) * diag(k_dim)
)

Mcmc <- list(R = R_total, keep = keep_every, nprint = 500)

# RUN MODEL
print("Running rhierMnlRwMixture (5-component mixture)...")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# POST-PROCESSING
print("Processing posterior samples...")
R_draws <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

param_names <- c(
    "Canon", "Sony", "Nikon", "Panasonic",
    "Pixels", "Zoom", "Video", "Swivel", "Wifi", "Price"
)

# --------------------------------------------------------------------------
# Extract per-component mu and Sigma for all 5 components
#
# out$nmix$compdraw[[r]] is a list of length n_comp.
# Each element has:  $mu  (k_dim vector)  and  $rooti  (upper-triangular Cholesky of Sigma^{-1})
# --------------------------------------------------------------------------

# mu_comp_draws  : list of length n_comp, each a (n_samples × k_dim) matrix
# cov_comp_draws : list of length n_comp, each a (n_samples × k_dim^2) matrix (vectorised Sigma)

mu_comp_draws <- vector("list", n_comp)
cov_comp_draws <- vector("list", n_comp)

for (comp in seq_len(n_comp)) {
    # --- mu ---
    mu_mat <- do.call(
        rbind,
        lapply(out$nmix$compdraw, function(draw) draw[[comp]]$mu)
    )
    mu_comp_draws[[comp]] <- mu_mat[keep_idx, , drop = FALSE]
    colnames(mu_comp_draws[[comp]]) <- param_names

    # --- Sigma (recover from inverse-Cholesky root) ---
    cov_mat <- do.call(
        rbind,
        lapply(out$nmix$compdraw, function(draw) {
            as.vector(chol2inv(draw[[comp]]$rooti))
        })
    )
    cov_comp_draws[[comp]] <- cov_mat[keep_idx, , drop = FALSE]
}

# Mixture-weight draws  (n_samples × n_comp)
# out$nmix$probdraw is an (R_draws × n_comp) matrix
prob_draws <- out$nmix$probdraw[keep_idx, , drop = FALSE]
colnames(prob_draws) <- paste0("comp", seq_len(n_comp))

# --------------------------------------------------------------------------
# Aggregate convenience objects (weighted average mu across components)
# Useful for downstream comparison with single-component / Python results.
# --------------------------------------------------------------------------
mu_draws_avg <- Reduce(
    "+",
    mapply(
        function(mu_mat, w_col) {
            sweep(mu_mat, 1, prob_draws[, w_col], "*")
        },
        mu_comp_draws, seq_len(n_comp),
        SIMPLIFY = FALSE
    )
)
colnames(mu_draws_avg) <- param_names
mu_draws_df <- as.data.frame(mu_draws_avg)

# Delta fallback (same logic as before)
if (!is.null(out$Deltadraw) && length(out$Deltadraw) > 0) {
    delta_matrix <- matrix(out$Deltadraw, ncol = k_dim, byrow = TRUE)
    delta_draws <- delta_matrix[keep_idx, , drop = FALSE]
    colnames(delta_draws) <- param_names
} else {
    delta_draws <- mu_draws_avg
}
delta_draws_df <- as.data.frame(delta_draws)

# Unit-level betas  (n_units × k_dim × n_samples)
beta_kept <- out$betadraw[, , keep_idx, drop = FALSE]
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # → (Units, Draws, Parameters)

# --------------------------------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------------------------------
script_dir <- this.path::here()
export_dir <- file.path(dirname(script_dir), "camera", "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

file_name <- "bayesm_output_camera_5comp.RData"
save_path <- file.path(export_dir, file_name)

save(mu_draws_df, # weighted-average mu (n_samples × k_dim)
    delta_draws_df, # Delta or mu fallback
    mu_comp_draws, # list[n_comp] of (n_samples × k_dim) — per-component mu
    cov_comp_draws, # list[n_comp] of (n_samples × k_dim^2) — per-component Sigma
    prob_draws, # (n_samples × n_comp) mixture weights
    beta_reordered, # (Units × Draws × Parameters)
    n_samples,
    n_comp,
    file = save_path
)

print(paste("SUCCESS: Saved to", save_path))
