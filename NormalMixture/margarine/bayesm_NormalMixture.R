# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# Section 5.5.3 - Normal Mixture Prior with K=5 components
# Key adaptations per paper:
#   - A_mu = 1/16 (not 1/100) for intercept prior
#   - nu = nvar+3, V = nu*I for Sigma_k prior (relatively diffuse)
#   - Standardize X variables
#   - ncomp = 5 (five-component mixture)
# ==============================================================================
library(this.path)
library(bayesm)

# 1. SETUP & DATA LOADING
data(margarine)
print("Preprocessing margarine data...")
select <- c(1, 2, 3, 4, 5, 7)
chPr_raw <- as.matrix(margarine$choicePrice)
demos_raw <- as.matrix(margarine$demos)

# 2. CHOICE DATA PREPROCESSING
chPr <- cbind(
    chPr_raw[, 1],
    chPr_raw[, 2],
    log(chPr_raw[, 2 + select])
)
chPr <- chPr[chPr[, 2] %in% select, , drop = FALSE]
chPr[chPr[, 2] == 7, 2] <- 6

# 3. CONSTRUCT LGTDATA (313 households)
hhid_list <- unique(chPr[, 1])
lgtdata <- list()
keep_hhids <- c()
p <- length(select)
ind <- 1

for (i in 1:length(hhid_list)) {
    hh_data <- chPr[chPr[, 1] == hhid_list[i], , drop = FALSE]
    nobs <- nrow(hh_data)
    if (nobs >= 5) {
        y <- hh_data[, 2]
        X <- createX(
            p = p,
            na = 1, Xa = hh_data[, 3:8],
            nd = NULL, Xd = NULL,
            INT = TRUE, base = 1
        )

        # --- PAPER SEC 5.5.3: Standardize X variables ---
        # "For this reason, we advocate standardizing the X variables."
        # Standardize each column of X (zero mean, unit variance)
        X <- scale(X)
        # -------------------------------------------------

        lgtdata[[ind]] <- list(y = as.integer(y), X = X)
        keep_hhids[ind] <- hhid_list[i]
        ind <- ind + 1
    }
}
print(paste("Kept", length(lgtdata), "households with 5+ observations."))

# 4. CONSTRUCT HIERARCHY MATRIX Z
Z_filtered <- NULL
for (id in keep_hhids) {
    row_data <- demos_raw[demos_raw[, 1] == id, c(2, 5), drop = FALSE]
    Z_filtered <- rbind(Z_filtered, row_data)
}
Z <- matrix(NA, nrow = nrow(Z_filtered), ncol = 2)
Z[, 1] <- log(Z_filtered[, 1]) # Log Income
Z[, 2] <- Z_filtered[, 2] # Family Size

# De-mean Z (matches paper's intent and bayesm requirements)
Z[, 1] <- Z[, 1] - mean(Z[, 1])
Z[, 2] <- Z[, 2] - mean(Z[, 2])
Z <- as.matrix(Z)

# 5. MODEL CONFIGURATION
# nvar = number of beta parameters per household = ncol(X)
# For 6 alternatives with intercepts + 1 price: nvar = 6
nvar <- p # = 6

# --- PAPER SEC 5.5.3: Prior specifications ---
# A_mu = 1/16 (paper advocates this over 1/100 = 0.01 for standardized X)
# "We then set A_mu to 1/16 or so rather than 1/100"
A_mu_val <- 1 / 16

# nu = nvar + 3, V = nu * I  (relatively diffuse prior on Sigma_k)
# "We will set the prior on Sigma to be relatively diffuse
#  by setting nu to nvar+3 and V = vI"
nu_val <- nvar + 3
V_val <- nu_val * diag(nvar)

# Number of mixture components: K = 5
# (paper compares K=1 and K=5; we run K=5 as the richer model)
ncomp_val <- 5
# ---------------------------------------------

# Z has 2 columns -> Delta is nvar x (nZ+1) = 6x3 (intercept + 2 demos)
# A is the prior precision on vec(Delta): (nZ+1) x (nZ+1) matrix
nZ <- ncol(Z) # = 2
A_Delta <- A_mu_val * diag(nZ + 1) # 3x3, same scale as A_mu

R_total <- 41000
burn_in <- 1000
keep_every <- 1

data_list <- list(p = p, lgtdata = lgtdata, Z = Z)

Prior <- list(
    ncomp = ncomp_val,
    A     = A_Delta, # Prior precision on Delta (includes mu via Z)
    nu    = nu_val, # Degrees of freedom for Sigma_k
    V     = V_val # Scale matrix for Sigma_k
)

Mcmc <- list(
    R      = R_total,
    keep   = keep_every,
    nprint = 500
)

# 6. RUN MODEL
print(paste0(
    "Running rhierMnlRwMixture | K=", ncomp_val,
    " | A_mu=1/", round(1 / A_mu_val),
    " | nu=", nu_val
))
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# 7. POST-PROCESSING
print("Processing posterior samples...")
R_draws <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

param_names <- c(
    "Blue Bonnett", "Fleischmanns", "House",
    "Generic", "Shed Spread Tub", "LogPrice"
)

# Extract Mu (component means) — shape: [n_samples x nvar]
mu_draws <- t(sapply(
    out$nmix$compdraw[keep_idx],
    function(x) x[[1]]$mu # first (only active) component mean
))
colnames(mu_draws) <- param_names

# Extract Delta (demographic coefficients) — shape: [n_samples x nvar*(nZ+1)]
delta_draws <- out$Deltadraw[keep_idx, ]

# Extract Beta (household-level) — shape: [nvar x nhh x n_samples]
beta_kept <- out$betadraw[, , keep_idx]

# Extract Covariance matrices (invert stored rooti = upper Chol of Sigma^{-1})
# chol2inv(rooti) recovers Sigma_k
cov_list <- lapply(
    out$nmix$compdraw[keep_idx],
    function(x) chol2inv(x[[1]]$rooti)
)
cov_draws_mat <- t(sapply(cov_list, as.vector))
cov_draws_df <- as.data.frame(cov_draws_mat)

# Extract component weights (pvec) — shape: [n_samples x ncomp]
pvec_draws <- t(sapply(
    out$nmix$compdraw[keep_idx],
    function(x) out$nmix$pvec[keep_idx, ] # already a matrix
))

# 8. SAVE OUTPUT
export_dir <- file.path(getwd(), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

mu_draws_df <- as.data.frame(mu_draws)
delta_draws_df <- as.data.frame(delta_draws)
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # [nvar x n_samples x nhh]

# Also save pvec (mixture weights) for diagnostics
pvec_draws_df <- as.data.frame(out$nmix$pvec[keep_idx, ])

file_name <- "bayesm_output_margarine_K5_standardized.RData"
save_path <- file.path(export_dir, file_name)

save(
    mu_draws_df, delta_draws_df, beta_reordered,
    cov_draws_df, pvec_draws_df, n_samples,
    file = save_path
)
print(paste("SUCCESS: Saved to", save_path))
