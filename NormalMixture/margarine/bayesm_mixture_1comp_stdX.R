# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: Single Normal Component (K=1) with STANDARDISED Price Variables
# MCMC:  25 000 total iterations | 5 000 burn-in | 20 000 posterior samples
# ==============================================================================

library(this.path)
library(bayesm)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & FILTERING
# ---------------------------------------------------------------------------
data(margarine)
cat("Loading and filtering margarine data...\n")

select <- c(1, 2, 3, 4, 5, 7)
chPr_raw <- as.matrix(margarine$choicePrice)
demos_raw <- as.matrix(margarine$demos)

# ---------------------------------------------------------------------------
# 2. CHOICE DATA PREPROCESSING (GLOBALLY STANDARDISED PRICE)
#    A single mean and SD are computed pooling all brands and occasions,
#    preserving cross-brand relative price levels.
# ---------------------------------------------------------------------------
log_prices <- log(chPr_raw[, 2 + select])

global_mean <- mean(as.vector(log_prices))
global_sd <- sd(as.vector(log_prices))
std_prices <- (log_prices - global_mean) / global_sd

cat(sprintf("Global log-price mean: %.4f  |  SD: %.4f\n", global_mean, global_sd))
cat(sprintf(
    "After std — mean: %.6f  |  SD: %.6f\n",
    mean(as.vector(std_prices)), sd(as.vector(std_prices))
))

chPr <- cbind(
    chPr_raw[, 1], # Household ID
    chPr_raw[, 2], # Choice indicator
    std_prices # Globally standardised log prices
)

# Keep only selected brands; recode brand 7 → 6 for contiguous indexing
chPr <- chPr[chPr[, 2] %in% select, , drop = FALSE]
chPr[chPr[, 2] == 7, 2] <- 6
# ─────────────────────────────────────────────────────────────────────────────

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
        X <- createX(
            p    = p,
            na   = 1,
            Xa   = hh_data[, 3:8], # Standardised log prices
            nd   = NULL,
            Xd   = NULL,
            INT  = TRUE,
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
Z[, 1] <- log(Z_filtered[, 1]) - mean(log(Z_filtered[, 1])) # centred log(income)
Z[, 2] <- Z_filtered[, 2] - mean(Z_filtered[, 2]) # centred family size
Z <- as.matrix(Z)

# ---------------------------------------------------------------------------
# 5. MODEL CONFIGURATION & PRIORS
#    A_mu = 0.01 matches Rossi et al. default for standardised data.
#    With unit-variance predictors the prior is no longer needed to be
#    as diffuse as 1/16; 0.01 is the value used in the book's replication.
# ---------------------------------------------------------------------------
nvar <- p # 5 brand intercepts + 1 price coefficient = 6
nz <- ncol(Z) # 2 demographic regressors
A_mu <- 1 / 16 # Prior precision for standardised-scale variables

Prior <- list(
    ncomp = 1,
    A     = A_mu * diag(nz + 1), # 3x3
    nu    = nvar + 3, # = 9
    V     = (nvar + 3) * diag(nvar) # 6x6
)

# ---------------------------------------------------------------------------
# 6. MCMC SETTINGS
# ---------------------------------------------------------------------------
R_total <- 25000
burn_in <- 5000
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)
Mcmc <- list(R = R_total, keep = 1, nprint = 500)

# ---------------------------------------------------------------------------
# 7. RUN MODEL
# ---------------------------------------------------------------------------
cat("Running rhierMnlRwMixture...\n")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 8. POST-PROCESSING  — discard burn-in, retain 20 000 posterior samples
# ---------------------------------------------------------------------------
cat("Processing posterior samples...\n")

R_draws <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws # 5001 … 25000  → 20 000 draws
n_samples <- length(keep_idx)
cat(sprintf("Posterior samples retained: %d\n", n_samples))

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# Mu
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))[keep_idx, ]
colnames(mu_draws) <- param_names
mu_draws_df <- as.data.frame(mu_draws)

# Delta
delta_draws <- out$Deltadraw[keep_idx, ]
delta_draws_df <- as.data.frame(delta_draws)

# Beta
beta_kept <- out$betadraw[, , keep_idx]
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

# Sigma
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))
cov_draws_mat <- t(sapply(cov_list, as.vector))[keep_idx, ]
cov_draws_df <- as.data.frame(cov_draws_mat)

# ---------------------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------------------
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_1comp_stdX.RData")
save(mu_draws_df, delta_draws_df, beta_reordered, cov_draws_df, n_samples,
    file = save_path
)

cat(sprintf("SUCCESS: %d posterior samples saved to %s\n", n_samples, save_path))
