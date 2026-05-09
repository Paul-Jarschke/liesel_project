# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: 5 Normal Components (K=5) with Raw X Variables
# MCMC:  25 000 total iterations | 5 000 burn-in | 20 000 posterior samples
# ==============================================================================

library(this.path)
library(bayesm)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & FILTERING
# ---------------------------------------------------------------------------
data(margarine)
cat("Loading and filtering margarine data...\n")

select <- c(1, 2, 3, 4, 5, 7) # Six brands kept (Parkay Stick = base)
chPr_raw <- as.matrix(margarine$choicePrice)
demos_raw <- as.matrix(margarine$demos)

# ---------------------------------------------------------------------------
# 2. CHOICE DATA PREPROCESSING (RAW SCALE)
#    chPr: col1 = hhid | col2 = choice | cols 3-8 = raw log prices
# ---------------------------------------------------------------------------
chPr <- cbind(
    chPr_raw[, 1], # Household ID
    chPr_raw[, 2], # Choice indicator
    log(chPr_raw[, 2 + select]) # Raw log prices (no standardisation)
)

# Keep only selected brands; recode brand 7 → 6 for contiguous indexing
chPr <- chPr[chPr[, 2] %in% select, , drop = FALSE]
chPr[chPr[, 2] == 7, 2] <- 6

# ---------------------------------------------------------------------------
# 3. CONSTRUCT LGTDATA  (households with >= 5 purchase occasions)
# ---------------------------------------------------------------------------
hhid_list <- unique(chPr[, 1])
lgtdata <- list()
keep_hhids <- c()
p <- length(select) # 6 alternatives
ind <- 1

for (i in seq_along(hhid_list)) {
    hh_data <- chPr[chPr[, 1] == hhid_list[i], , drop = FALSE]
    nobs <- nrow(hh_data)

    if (nobs >= 5) {
        y <- hh_data[, 2]
        # createX: Xa = raw log prices; INT = TRUE adds brand intercepts; base = 1
        X <- createX(
            p = p,
            na = 1,
            Xa = hh_data[, 3:8, drop = FALSE],
            nd = NULL,
            Xd = NULL,
            INT = TRUE,
            base = 1
        )
        lgtdata[[ind]] <- list(y = as.integer(y), X = X)
        keep_hhids[ind] <- hhid_list[i]
        ind <- ind + 1
    }
}

cat(sprintf("Kept %d households with 5+ observations.\n", length(lgtdata)))

# ---------------------------------------------------------------------------
# 4. CONSTRUCT HIERARCHY MATRIX Z  (centred)
# ---------------------------------------------------------------------------
Z_filtered <- NULL
for (id in keep_hhids) {
    row_data <- demos_raw[demos_raw[, 1] == id, c(2, 5), drop = FALSE]
    Z_filtered <- rbind(Z_filtered, row_data)
}

Z <- matrix(NA, nrow = nrow(Z_filtered), ncol = 2)
Z[, 1] <- log(Z_filtered[, 1]) # log(income)
Z[, 2] <- Z_filtered[, 2] # family size

# Centre each column (required by bayesm)
Z[, 1] <- Z[, 1] - mean(Z[, 1])
Z[, 2] <- Z[, 2] - mean(Z[, 2])
Z <- as.matrix(Z)

# ---------------------------------------------------------------------------
# 5. MODEL CONFIGURATION & PRIORS
#    A_mu = 1/16 → diffuse prior precision suited for raw (non-standardised) vars
# ---------------------------------------------------------------------------
nvar <- p # 5 brand intercepts + 1 price coefficient = 6
nz <- ncol(Z) # 2 demographic regressors
A_mu <- 1 / 16 # Diffuse prior precision for raw-scale variables
K <- 5 # Number of mixture components

Prior <- list(
    ncomp = K, # K = 5 Normal components
    A     = A_mu * diag(nz + 1), # 3x3 prior precision on Delta
    nu    = nvar + 3, # = 9  (degrees of freedom for Sigma)
    V     = (nvar + 3) * diag(nvar) # 6x6 prior scale for Sigma
)

# ---------------------------------------------------------------------------
# 6. MCMC SETTINGS
# ---------------------------------------------------------------------------
R_total <- 200000
burn_in <- 10000
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)
Mcmc <- list(R = R_total, keep = 10, nprint = 500)

# ---------------------------------------------------------------------------
# 7. RUN MODEL
# ---------------------------------------------------------------------------
cat(sprintf("Running rhierMnlRwMixture with K=%d components...\n", K))
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 8. POST-PROCESSING  — discard burn-in, account for thinning (Multi-Component)
# ---------------------------------------------------------------------------
cat("Processing posterior samples for multi-component mixture...\n")

# Determine how many thinned draws to discard
thinned_burn_in <- burn_in %/% Mcmc$keep

R_draws <- length(out$nmix$compdraw)
keep_idx <- (thinned_burn_in + 1):R_draws

n_samples <- length(keep_idx)
cat(sprintf("Thinned posterior samples retained after burn-in: %d\n", n_samples))

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# --- Mixture Probabilities (prob: dims = n_samples x K) ---
raw_probs <- out$nmix$probdraw
if (nrow(raw_probs) == K) {
    raw_probs <- t(raw_probs) # Transpose if bayesm returned it as K x R
}
pvec_draws <- raw_probs[keep_idx, ]
pvec_draws_df <- as.data.frame(pvec_draws)
colnames(pvec_draws_df) <- paste0("Comp_", 1:K)

# --- Mu & Sigma (Flattened horizontally for pyreadr compatibility) ---
mu_draws <- NULL
cov_draws <- NULL

for (k in 1:K) {
    # Extract mu for component k
    mu_k <- t(sapply(out$nmix$compdraw, function(x) x[[k]]$mu))[keep_idx, ]

    # Extract and invert Cholesky rooti to get covariance for component k
    cov_k <- t(sapply(out$nmix$compdraw, function(x) as.vector(chol2inv(x[[k]]$rooti))))[keep_idx, ]

    # cbind appends the matrices horizontally
    mu_draws <- cbind(mu_draws, mu_k)
    cov_draws <- cbind(cov_draws, cov_k)
}

mu_draws_df <- as.data.frame(mu_draws)
cov_draws_df <- as.data.frame(cov_draws)

# --- Delta (upper-level demographic coefficients) ---
delta_draws <- out$Deltadraw[keep_idx, ]
delta_draws_df <- as.data.frame(delta_draws)

# --- Beta (household-level coefficients; dims: nvar x n_samples x nhh) ---
beta_kept <- out$betadraw[, , keep_idx]
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # → nvar x n_samples x nhh

# ---------------------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------------------
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_5comp_rawX.RData")

# Save DataFrames rather than arrays so pyreadr imports them flawlessly
save(pvec_draws_df, mu_draws_df, delta_draws_df, beta_reordered, cov_draws_df, n_samples,
    file = save_path
)

cat(sprintf("SUCCESS: %d posterior samples saved to %s\n", n_samples, save_path))
