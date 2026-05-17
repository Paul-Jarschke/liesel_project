# ==============================================================================
# REPLICATION: Hierarchical MNL — Simulated Data — Mixture Model
#
# OUTPUT: Saves mu_draws_df, cov_draws_df, pvec_draws_df as DataFrames
#         compatible with pyreadr / load_and_format_draws() in the
#         marginal-density comparison script.
# ==============================================================================

library(jsonlite)
library(bayesm)
library(this.path)

# ── 0. User Configuration ──────────────────────────────────────────────────────

# Define all model and MCMC parameters here before running the script
DATA_FILENAME <- "sim_data_U300_O30_A4_K2_D2_pvec5050_seed101.json"

# Define parametes to match the liesel implementation

N_COMP <- 2 # Number of mixture components
R_TOTAL <- 41000 # Total MCMC draws
BURN_IN <- 1000 # Burn-in draws to discard
KEEP_EVERY <- 4 # Thinning factor (keep 1 in every 4 draws)
RANDOM_SEED <- 101 # Seed for reproducibility


# ── 1. Load data ───────────────────────────────────────────────────────────────
cat(sprintf("\n=== Loading Data ===\n"))

# Dynamically find the script's directory and step back to the project root
script_dir <- this.path::here()
project_root <- normalizePath(file.path(script_dir, "..", "..", ".."))

# Point to the central simulated data directory
data_dir <- file.path(project_root, "data", "simulated")
data_path <- file.path(data_dir, DATA_FILENAME)

cat(sprintf("Reading data from: %s\n", DATA_FILENAME))
raw <- fromJSON(data_path, simplifyVector = TRUE)

# Scalars
n_units <- raw$n_units
n_params <- raw$n_params
n_alts <- raw$n_alts
n_demos <- raw$n_demos
K_true <- raw$K

n_obs <- length(raw$y) / n_units # observations per unit

cat(sprintf(
    " - Units (N): %d\n - Obs/Unit:  %d\n - Alts (A):  %d\n - Params (P): %d\n - Demos (D): %d\n - True Comp: %d\n",
    n_units, as.integer(n_obs), n_alts, n_params, n_demos, K_true
))

# ── 2. Reconstruct lgtdata ─────────────────────────────────────────────────────
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

# ── 3. Z matrix (Demographics) ─────────────────────────────────────────────────
Z <- matrix(unlist(raw$Z), nrow = n_units, ncol = n_demos, byrow = FALSE)

# ── 4. True parameters for post-run comparison ────────────────────────────────
TRUE_DELTA <- matrix(unlist(raw$TRUE_DELTA), nrow = n_demos, ncol = n_params, byrow = FALSE)
TRUE_MU_K <- matrix(unlist(raw$TRUE_MU_K), nrow = K_true, ncol = n_params, byrow = FALSE)
TRUE_PVEC <- unlist(raw$TRUE_PVEC)
TRUE_BETA <- matrix(unlist(raw$TRUE_BETA), nrow = n_units, ncol = n_params, byrow = FALSE)

param_names <- paste0("Param_", seq_len(n_params) - 1)
demo_names <- paste0("Demo_", seq_len(n_demos) - 1)


# ── 5. Model configuration ─────────────────────────────────────────────────────
data_list <- list(p = n_alts, lgtdata = lgtdata, Z = Z)

# Prior — matched to Python Liesel Model
Prior <- list(
    ncomp = N_COMP,
    Ad    = 0.01 * diag(n_demos * n_params),
    nu    = n_params + 3,
    V     = (n_params + 3) * diag(n_params),
    Amu   = 0.01,
    a     = rep(5, N_COMP)
)

Mcmc <- list(R = R_TOTAL, keep = KEEP_EVERY, nprint = 5000)


# ── 6. Run model ───────────────────────────────────────────────────────────────
cat(sprintf("\n=== Starting rhierMnlRwMixture (%d-Component) ===\n", N_COMP))
set.seed(RANDOM_SEED)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)


# ── 7. Post-processing — discard burn-in ──────────────────────────────────────
cat("\n=== Processing Posterior Samples ===\n")
thinned_burn_in <- BURN_IN %/% KEEP_EVERY
R_draws <- length(out$nmix$compdraw)
keep_idx <- seq(thinned_burn_in + 1L, R_draws)
n_samples <- length(keep_idx)

cat(sprintf(
    "Total thinned draws: %d | Retained after burn-in: %d\n",
    R_draws, n_samples
))

# ── 8. Extract mu and cov per component ───────────────────────────────────────
mu_list <- vector("list", N_COMP)
cov_list <- vector("list", N_COMP)

for (k in seq_len(N_COMP)) {
    mu_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) x[[k]]$mu)
    )[keep_idx, , drop = FALSE]

    cov_list[[k]] <- t(
        sapply(out$nmix$compdraw, function(x) as.vector(chol2inv(x[[k]]$rooti)))
    )[keep_idx, , drop = FALSE]
}

# ── 9. Flatten horizontally ───────────────────────────────────────────────────
mu_flat <- do.call(cbind, mu_list)
cov_flat <- do.call(cbind, cov_list)

pvec_raw <- out$nmix$probdraw
if (nrow(pvec_raw) == N_COMP) pvec_raw <- t(pvec_raw)
pvec_flat <- pvec_raw[keep_idx, , drop = FALSE]

mu_draws_df <- as.data.frame(mu_flat)
cov_draws_df <- as.data.frame(cov_flat)
pvec_draws_df <- as.data.frame(pvec_flat)

colnames(mu_draws_df) <- paste0(
    rep(paste0("Comp_", seq_len(N_COMP)), each = n_params), "_",
    rep(param_names, times = N_COMP)
)
colnames(pvec_draws_df) <- paste0("Comp_", seq_len(N_COMP))

# ── 10. Delta & Beta draws ─────────────────────────────────────────────────────
delta_draws_df <- as.data.frame(out$Deltadraw[keep_idx, , drop = FALSE])
beta_reordered <- aperm(out$betadraw[, , keep_idx, drop = FALSE], c(1, 3, 2))


# ── 11. Posterior Summary Tables ───────────────────────────────────────────────
cat("\n=== pvec: Posterior Summary ===\n")
cat("Posterior Mean :", round(colMeans(pvec_draws_df), 4), "\n")
cat("True pvec      :", round(TRUE_PVEC, 4), "\n")

cat("\n=== Global Parameters (Baseline Means mu_k) Summary Tables ===\n")
for (k in seq_len(N_COMP)) {
    post_mean <- colMeans(mu_list[[k]])

    # Simple Hungarian-style matching to find the closest true component
    dists <- sapply(seq_len(K_true), function(j) sum((post_mean - TRUE_MU_K[j, ])^2))
    best_j <- which.min(dists)

    cat(sprintf("\n--- MCMC Component %d (Mapped to True Component %d) ---\n", k - 1, best_j - 1))
    comparison <- data.frame(
        Parameter      = param_names,
        Posterior_Mean = round(post_mean, 4),
        True_Value     = round(TRUE_MU_K[best_j, ], 4),
        Diff_Abs       = round(abs(post_mean - TRUE_MU_K[best_j, ]), 4)
    )
    print(comparison, row.names = FALSE)
}


# ── 12. Export ─────────────────────────────────────────────────────────────────
cat("\n=== Exporting Data ===\n")
export_dir <- file.path(script_dir, "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

export_filename <- sprintf("bayesm_output_simulated_%dcomp.RData", N_COMP)
save_path <- file.path(export_dir, export_filename)

save(
    mu_draws_df,
    cov_draws_df,
    pvec_draws_df,
    delta_draws_df,
    beta_reordered,
    TRUE_MU_K,
    TRUE_PVEC,
    TRUE_DELTA,
    n_samples,
    file = save_path
)

cat(sprintf("Saved %d-component draws → %s\n", N_COMP, save_path))
cat("   mu_draws_df   :", sprintf("(%d, %d)", n_samples, N_COMP * n_params), "\n")
cat("   cov_draws_df  :", sprintf("(%d, %d)", n_samples, N_COMP * n_params^2), "\n")
cat("   pvec_draws_df :", sprintf("(%d, %d)", n_samples, N_COMP), "\n")
