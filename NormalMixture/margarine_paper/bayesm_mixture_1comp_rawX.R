# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: Single Normal Component (K=1) with Raw X Variables
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
            Xa = hh_data[, 3:8],
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
#    Per paper p.150: "Given that we center the Z variables..."
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

Prior <- list(
    ncomp = 1, # K = 1 Normal component
    A     = A_mu * diag(nz + 1), # 3x3 prior precision on Delta
    nu    = nvar + 3, # = 9  (degrees of freedom for Sigma)
    V     = (nvar + 3) * diag(nvar) # 6x6 prior scale for Sigma
)

# ---------------------------------------------------------------------------
# 6. MCMC SETTINGS
#    R_total = 25 000  |  burn_in = 5 000  |  posterior samples = 20 000
#    keep = 1 so every draw is stored; burn-in is discarded in post-processing
# ---------------------------------------------------------------------------
R_total <- 60000
burn_in <- 10000
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)
Mcmc <- list(R = R_total, keep = 10, nprint = 500)

# ---------------------------------------------------------------------------
# 7. RUN MODEL
# ---------------------------------------------------------------------------
cat("Running rhierMnlRwMixture...\n")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 8. POST-PROCESSING  — discard burn-in, account for thinning
# ---------------------------------------------------------------------------
cat("Processing posterior samples...\n")

# Determine how many thinned draws to discard
# Using integer division (%/%) ensures a clean index
thinned_burn_in <- burn_in %/% Mcmc$keep

R_draws <- length(out$nmix$compdraw) # This will be 1500
keep_idx <- (thinned_burn_in + 1):R_draws # This will be 501:1500

n_samples <- length(keep_idx) # This will be 1000
cat(sprintf("Thinned posterior samples retained after burn-in: %d\n", n_samples))

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# --- Mu (mixture mean of the random-coefficient distribution) ---
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))[keep_idx, ]
colnames(mu_draws) <- param_names
mu_draws_df <- as.data.frame(mu_draws)

# --- Delta (upper-level demographic coefficients) ---
delta_draws <- out$Deltadraw[keep_idx, ]
delta_draws_df <- as.data.frame(delta_draws)

# --- Beta (household-level coefficients; dims: nvar x nhh x n_samples) ---
beta_kept <- out$betadraw[, , keep_idx]
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # → nvar x n_samples x nhh

# --- Sigma (covariance matrix recovered from inverse Cholesky root) ---
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))
cov_draws_mat <- t(sapply(cov_list, as.vector))[keep_idx, ]
cov_draws_df <- as.data.frame(cov_draws_mat)

# ---------------------------------------------------------------------------
# 9. EXPORT
# ---------------------------------------------------------------------------
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save_path <- file.path(export_dir, "bayesm_output_margarine_1comp_rawX.RData")
save(mu_draws_df, delta_draws_df, beta_reordered, cov_draws_df, n_samples,
    file = save_path
)

cat(sprintf("SUCCESS: %d posterior samples saved to %s\n", n_samples, save_path))
