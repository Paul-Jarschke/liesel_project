# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# MODEL: Ten Normal Components (K=10) with Raw X Variables
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
# ---------------------------------------------------------------------------
chPr <- cbind(
    chPr_raw[, 1],
    chPr_raw[, 2],
    log(chPr_raw[, 2 + select]) # RAW log prices
)

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
        X <- createX(p = p, na = 1, Xa = hh_data[, 3:8], nd = NULL, Xd = NULL, INT = TRUE, base = 1)
        lgtdata[[ind]] <- list(y = as.integer(y), X = X)
        keep_hhids[ind] <- hhid_list[i]
        ind <- ind + 1
    }
}

# ---------------------------------------------------------------------------
# 4. CONSTRUCT HIERARCHY MATRIX Z (CENTERED)
# ---------------------------------------------------------------------------
Z_filtered <- NULL
for (id in keep_hhids) {
    row_data <- demos_raw[demos_raw[, 1] == id, c(2, 5), drop = FALSE]
    Z_filtered <- rbind(Z_filtered, row_data)
}

Z <- matrix(NA, nrow = nrow(Z_filtered), ncol = 2)
Z[, 1] <- log(Z_filtered[, 1])
Z[, 2] <- Z_filtered[, 2]

z_means <- colMeans(Z)
Z[, 1] <- Z[, 1] - z_means[1]
Z[, 2] <- Z[, 2] - z_means[2]

# ---------------------------------------------------------------------------
# 5. MODEL CONFIGURATION & PRIORS
# ---------------------------------------------------------------------------
nvar <- p
nz <- ncol(Z)
A_mu <- 1 / 16
ncomp <- 10 # <--- SET TO 10 COMPONENTS

Prior <- list(
    ncomp = ncomp,
    A     = A_mu * diag(nz + 1),
    nu    = nvar + 3,
    V     = (nvar + 3) * diag(nvar)
)

Mcmc <- list(R = 40000, keep = 10, nprint = 500)
data_list <- list(p = p, lgtdata = lgtdata, Z = Z)

# ---------------------------------------------------------------------------
# 6. RUN MODEL
# ---------------------------------------------------------------------------
set.seed(123)
cat("\nRunning 10-component hierarchical MNL model...\n")
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# ---------------------------------------------------------------------------
# 7. POST-PROCESSING & EXPORT (Dynamic for K=10)
# ---------------------------------------------------------------------------
cat("Extracting parameters for all 10 components...\n")

R_draws <- length(out$nmix$compdraw)
pvec_draws_df <- as.data.frame(out$nmix$probdraw)

mu_mat <- matrix(0, nrow = R_draws, ncol = ncomp * nvar)
cov_mat <- matrix(0, nrow = R_draws, ncol = ncomp * (nvar * nvar))

for (r in 1:R_draws) {
    mu_r <- numeric(0)
    cov_r <- numeric(0)

    for (k in 1:ncomp) {
        comp <- out$nmix$compdraw[[r]][[k]]
        mu_r <- c(mu_r, comp$mu)
        Sigma_k <- chol2inv(comp$rooti)
        cov_r <- c(cov_r, as.vector(Sigma_k))
    }

    mu_mat[r, ] <- mu_r
    cov_mat[r, ] <- cov_r
}

mu_draws_df <- as.data.frame(mu_mat)
cov_draws_df <- as.data.frame(cov_mat)

export_dir <- file.path(dirname(this.path()), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

save(
    pvec_draws_df,
    mu_draws_df,
    cov_draws_df,
    file = file.path(export_dir, "bayesm_output_margarine_10comp_rawX.RData")
)

cat("SUCCESS: Data saved for Python post-processing.\n")
