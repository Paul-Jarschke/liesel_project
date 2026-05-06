# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: Single Normal Component (K=1) with Raw X Variables
# ==============================================================================

library(this.path)
library(bayesm)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & FILTERING
# ---------------------------------------------------------------------------
data(margarine)
cat("Loading and filtering margarine data...\n")

select <- c(1, 2, 3, 4, 5, 7) # Six brands kept (Parkay Stick is base)
chPr_raw <- as.matrix(margarine$choicePrice)
demos_raw <- as.matrix(margarine$demos)

# ---------------------------------------------------------------------------
# 2. CHOICE DATA PREPROCESSING (RAW SCALE)
#    Building chPr: col1=hhid, col2=choice, cols 3-8=raw log prices
# ---------------------------------------------------------------------------
chPr <- cbind(
    chPr_raw[, 1], # Household ID
    chPr_raw[, 2], # Choice indicator
    log(chPr_raw[, 2 + select]) # RAW log prices (No Standardization)
)

# Keep selected brands and recode brand 7 to 6 for continuity
chPr <- chPr[chPr[, 2] %in% select, , drop = FALSE]
chPr[chPr[, 2] == 7, 2] <- 6

# ---------------------------------------------------------------------------
# 3. CONSTRUCT LGTDATA
# ---------------------------------------------------------------------------
hhid_list <- unique(chPr[, 1])
lgtdata <- list()
keep_hhids <- c()
p <- length(select)

ind <- 1
for (i in seq_along(hhid_list)) {
    hh_data <- chPr[chPr[, 1] == hhid_list[i], , drop = FALSE]
    nobs <- nrow(hh_data)

    if (nobs >= 5) {
        y <- hh_data[, 2]
        # createX builds the design matrix
        # Xa = raw log prices; INT = TRUE adds brand intercepts
        X <- createX(
            p = p,
            na = 1,
            Xa = hh_data[, 3:8], # Columns 3-8 are log prices
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
# 4. CONSTRUCT HIERARCHY MATRIX Z (CENTERED)
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

# Center Z columns
z_means <- colMeans(Z)
Z[, 1] <- Z[, 1] - z_means[1]
Z[, 2] <- Z[, 2] - z_means[2]

# ---------------------------------------------------------------------------
# 5. MODEL CONFIGURATION & PRIORS
#    Prior A (A_mu) is set to 1/16 for raw variables to be diffuse.
# ---------------------------------------------------------------------------
nvar <- p # 5 intercepts + 1 price coefficient
nz <- ncol(Z)
A_mu <- 1 / 16 # Diffuse prior precision for non-standardized data

Prior <- list(
    ncomp = 1, # K=1 for the grey line replication
    A     = A_mu * diag(nz + 1),
    nu    = nvar + 3, # 9
    V     = (nvar + 3) * diag(nvar)
)

Mcmc <- list(R = 40000, keep = 10, nprint = 500)
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)

# ---------------------------------------------------------------------------
# 6. RUN MODEL
# ---------------------------------------------------------------------------
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 7. POST-PROCESSING & EXPORT
# ---------------------------------------------------------------------------
R_draws <- length(out$nmix$compdraw)
# mu_draws: (R x 6)
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
# cov_draws: (R x 36) Recover Sigma via inverse Cholesky
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))
cov_draws_mat <- t(sapply(cov_list, as.vector))

mu_draws_df <- as.data.frame(mu_draws)
cov_draws_df <- as.data.frame(cov_draws_mat)

# Save to be read by your Python plotting script
export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save(
    mu_draws_df,
    cov_draws_df,
    file = file.path(export_dir, "bayesm_output_margarine_1comp_rawX.RData")
)

cat("SUCCESS: Data saved for Python post-processing.\n")
