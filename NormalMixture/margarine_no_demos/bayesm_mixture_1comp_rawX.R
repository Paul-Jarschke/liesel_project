# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: Single Normal Component (K=1) | Raw X Variables | NO Demographics (Z)
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
# 3. CONSTRUCT LGTDATA  (households with >= 5 purchase occasions)
# ---------------------------------------------------------------------------
hhid_list <- unique(chPr[, 1])
lgtdata <- list()
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
            Xa = hh_data[, 3:8],
            nd = NULL,
            Xd = NULL,
            INT = TRUE,
            base = 1
        )
        lgtdata[[ind]] <- list(y = as.integer(y), X = X)
        ind <- ind + 1
    }
}

cat(sprintf("Kept %d households with 5+ observations.\n", length(lgtdata)))

# ---------------------------------------------------------------------------
# 4. MODEL CONFIGURATION & PRIORS  (no Z → no Delta, no A prior)
#
#    Without Z, the upper level is simply:
#        beta_i ~ N(mu, Sigma)
#    The prior on mu is:  mu ~ N(mubar, (Amu)^{-1} * I)
#
#    Amu  = 1/16  → diffuse prior precision suited for raw-scale variables
#    mubar defaults to 0 inside rhierMnlRwMixture when not supplied
# ---------------------------------------------------------------------------
nvar <- p # 5 brand intercepts + 1 price coefficient = 6

Prior <- list(
    ncomp = 1, # K = 1 Normal component
    Amu   = 1 / 16, # Scalar prior precision on mu (replaces A when Z absent)
    nu    = nvar + 3, # = 9  (degrees of freedom for Sigma)
    V     = (nvar + 3) * diag(nvar) # 6x6 prior scale for Sigma
)

# ---------------------------------------------------------------------------
# 5. MCMC SETTINGS
#    R_total = 60 000  |  burn_in = 10 000  |  keep = 10 (thinning)
#    → 5 000 thinned posterior draws total; 1 000 retained after burn-in
# ---------------------------------------------------------------------------
R_total <- 60000
burn_in <- 10000

# NOTE: Z is omitted from data_list entirely
data_list <- list(p = p, lgtdata = lgtdata)
Mcmc <- list(R = R_total, keep = 1, nprint = 500)

# ---------------------------------------------------------------------------
# 6. RUN MODEL
# ---------------------------------------------------------------------------
cat("Running rhierMnlRwMixture (no demographics)...\n")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 7. POST-PROCESSING  — discard burn-in, account for thinning
# ---------------------------------------------------------------------------
cat("Processing posterior samples...\n")

thinned_burn_in <- burn_in %/% Mcmc$keep
R_draws <- length(out$nmix$compdraw)
keep_idx <- (thinned_burn_in + 1):R_draws
n_samples <- length(keep_idx)

cat(sprintf("Thinned posterior samples retained after burn-in: %d\n", n_samples))

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# --- Mu (mixture mean of the random-coefficient distribution) ---
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))[keep_idx, ]
colnames(mu_draws) <- param_names
mu_draws_df <- as.data.frame(mu_draws)

# --- Beta (household-level coefficients) ---
beta_kept <- out$betadraw[, , keep_idx]
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # → nvar x n_samples x nhh

# --- Sigma (covariance matrix recovered from inverse Cholesky root) ---
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))
cov_draws_mat <- t(sapply(cov_list, as.vector))[keep_idx, ]
cov_draws_df <- as.data.frame(cov_draws_mat)

# ---------------------------------------------------------------------------
# 8. EXPORT  (delta_draws omitted — no upper-level regression)
# ---------------------------------------------------------------------------
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_1comp_rawX_noZ.RData")
save(mu_draws_df, beta_reordered, cov_draws_df, n_samples,
    file = save_path
)

cat(sprintf("SUCCESS: %d posterior samples saved to %s\n", n_samples, save_path))
