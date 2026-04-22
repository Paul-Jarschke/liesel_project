# ==============================================================================
# REPLICATION: Hierarchical MNL (Rossi et al. 2006 Margarine Example)
# Matches Python Implementation (Liesel) by using centered/de-meaned demographic
# variables unlike explained in the paper (p.)
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
    X <- createX(p = p, na = 1, Xa = hh_data[, 3:8], nd = NULL, Xd = NULL, INT = TRUE, base = 1)
    lgtdata[[ind]] <- list(y = as.integer(y), X = X)
    keep_hhids[ind] <- hhid_list[i]
    ind <- ind + 1
  }
}

print(paste("Kept", length(lgtdata), "households with 5+ observations."))

# 4. CONSTRUCT HIERARCHY MATRIX Z (313 rows)
Z_filtered <- NULL
for (id in keep_hhids) {
  row_data <- demos_raw[demos_raw[, 1] == id, c(2, 5), drop = FALSE]
  Z_filtered <- rbind(Z_filtered, row_data)
}

Z <- matrix(NA, nrow = nrow(Z_filtered), ncol = 2)
Z[, 1] <- log(Z_filtered[, 1]) # Log Income
Z[, 2] <- Z_filtered[, 2] # Family Size

# --- CRITICAL FIX: De-mean Z to satisfy bayesm and match paper's intent ---
Z[, 1] <- Z[, 1] - mean(Z[, 1])
Z[, 2] <- Z[, 2] - mean(Z[, 2])
# --------------------------------------------------------------------------

Z <- as.matrix(Z)

# 5. MODEL CONFIGURATION
R_total <- 41000
burn_in <- 1000
keep_every <- 1

data_list <- list(p = p, lgtdata = lgtdata, Z = Z)
Prior <- list(ncomp = 1, A = 0.01 * diag(3), nu = 6 + 3, V = (6 + 3) * diag(6))
Mcmc <- list(R = R_total, keep = keep_every, nprint = 500)

# 6. RUN MODEL
print("Running rhierMnlRwMixture...")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# 7. POST-PROCESSING
print("Processing posterior samples...")
R_draws <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

param_names <- c("Blue Bonnett", "Fleischmanns", "House", "Generic", "Shed Spread Tub", "LogPrice")

# Extract Mu (Baseline Intercepts)
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))[keep_idx, ]
colnames(mu_draws) <- param_names

# Extract Delta (Demographic Coefficients)
delta_draws <- out$Deltadraw[keep_idx, ]

# Extract Beta (Individual Household Coefficients)
beta_kept <- out$betadraw[, , keep_idx]

# Extract Covariance Matrices
# bayesm stores 'rooti' (inverse Cholesky root). We use chol2inv to get the Covariance matrix.
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))

# Flatten each 6x6 matrix into a vector so it saves cleanly as a 2D dataframe
cov_draws_mat <- t(sapply(cov_list, as.vector))[keep_idx, ]
cov_draws_df <- as.data.frame(cov_draws_mat)
# ----------------------------------------

# 8. SAVE OUTPUT
# Create the Data folder inside your current working directory
export_dir <- file.path(getwd(), "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

mu_draws_df <- as.data.frame(mu_draws)
delta_draws_df <- as.data.frame(delta_draws)
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

file_name <- "bayesm_output_margarine_demeaned.RData"
save_path <- file.path(export_dir, file_name)

# Save all data, including the covariance draws
save(mu_draws_df, delta_draws_df, beta_reordered, cov_draws_df, n_samples, file = save_path)
print(paste("SUCCESS: Saved to", save_path))
