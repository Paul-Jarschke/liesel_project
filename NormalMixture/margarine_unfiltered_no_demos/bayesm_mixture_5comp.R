# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: 5 Normal Components (K=5) | Raw X Variables | NO Demographics (Z)
# MCMC:  60 000 total iterations | 10 000 burn-in | thinning = 10
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
# 3. CONSTRUCT LGTDATA  (all households, no purchase-count filter)
# ---------------------------------------------------------------------------
hhid_list <- unique(chPr[, 1])
lgtdata <- list()
p <- length(select) # 6 alternatives
ind <- 1

for (i in seq_along(hhid_list)) {
    hh_data <- chPr[chPr[, 1] == hhid_list[i], , drop = FALSE]

    y <- hh_data[, 2]
    # drop = FALSE ensures single-row households stay as matrices,
    # preventing the ncol() == NULL crash inside createX
    X <- createX(
        p    = p,
        na   = 1,
        Xa   = hh_data[, 3:8, drop = FALSE],
        nd   = NULL,
        Xd   = NULL,
        INT  = TRUE,
        base = 1
    )
    lgtdata[[ind]] <- list(y = as.integer(y), X = X)
    ind <- ind + 1
}

cat(sprintf("Total households included: %d\n", length(lgtdata)))

# ---------------------------------------------------------------------------
# 4. MODEL CONFIGURATION & PRIORS  (no Z → no Delta, no A prior)
#
#    Without Z, the upper level is simply:
#        beta_i ~ sum_k pi_k * N(mu_k, Sigma_k)
#    The prior on each mu_k is:  mu_k ~ N(mubar, (Amu)^{-1} * I)
#
#    Amu  = 1/16  → diffuse prior precision for raw-scale variables
#    mubar defaults to 0 inside rhierMnlRwMixture when not supplied
# ---------------------------------------------------------------------------
nvar <- p # 5 brand intercepts + 1 price coefficient = 6
K <- 5 # Number of mixture components

Prior <- list(
    ncomp = K, # K = 5 Normal components
    Amu   = 1 / 16, # Scalar prior precision on each mu_k (replaces A when Z absent)
    nu    = nvar + 3, # = 9  (degrees of freedom for Sigma)
    V     = (nvar + 3) * diag(nvar) # 6x6 prior scale for Sigma
)

# ---------------------------------------------------------------------------
# 5. MCMC SETTINGS
#    NOTE: Z is omitted from data_list entirely
# ---------------------------------------------------------------------------
R_total <- 60000
burn_in <- 10000
data_list <- list(p = p, lgtdata = lgtdata)
Mcmc <- list(R = R_total, keep = 10, nprint = 500)

# ---------------------------------------------------------------------------
# 6. RUN MODEL
# ---------------------------------------------------------------------------
cat(sprintf("Running rhierMnlRwMixture with K=%d components (no demographics)...\n", K))
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 7. POST-PROCESSING  — discard burn-in, account for thinning
# ---------------------------------------------------------------------------
cat("Processing posterior samples for multi-component mixture...\n")

thinned_burn_in <- burn_in %/% Mcmc$keep
R_draws <- length(out$nmix$compdraw)
keep_idx <- (thinned_burn_in + 1):R_draws
n_samples <- length(keep_idx)

cat(sprintf("Thinned posterior samples retained after burn-in: %d\n", n_samples))

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# --- Mixture Probabilities (pvec_draws: dims = n_samples x K) ---
raw_probs <- out$nmix$probdraw
if (nrow(raw_probs) == K) {
    raw_probs <- t(raw_probs) # Transpose if bayesm returned it as K x R
}
pvec_draws <- raw_probs[keep_idx, ]
pvec_draws_df <- as.data.frame(pvec_draws)
colnames(pvec_draws_df) <- paste0("Comp_", 1:K)

# --- Mu & Sigma per component (flattened horizontally) ---
mu_draws <- NULL
cov_draws <- NULL

for (k in 1:K) {
    # mu for component k across all posterior draws
    mu_k <- t(sapply(out$nmix$compdraw, function(x) x[[k]]$mu))[keep_idx, ]

    # Covariance for component k (recovered from inverse Cholesky root)
    cov_k <- t(sapply(out$nmix$compdraw, function(x) as.vector(chol2inv(x[[k]]$rooti))))[keep_idx, ]

    mu_draws <- cbind(mu_draws, mu_k)
    cov_draws <- cbind(cov_draws, cov_k)
}

mu_draws_df <- as.data.frame(mu_draws)
cov_draws_df <- as.data.frame(cov_draws)

# --- Beta (household-level coefficients; dims: nvar x n_samples x nhh) ---
beta_kept <- out$betadraw[, , keep_idx]
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # → nvar x n_samples x nhh

# ---------------------------------------------------------------------------
# 8. EXPORT  (delta_draws omitted — no upper-level regression)
# ---------------------------------------------------------------------------
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_5comp_rawX_noZ.RData")
save(pvec_draws_df, mu_draws_df, beta_reordered, cov_draws_df, n_samples,
    file = save_path
)

cat(sprintf("SUCCESS: %d posterior samples saved to %s\n", n_samples, save_path))
