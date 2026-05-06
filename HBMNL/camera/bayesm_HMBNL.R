# ==============================================================================
# REPLICATION: Hierarchical MNL for Camera Dataset
# ==============================================================================

library(this.path)
library(bayesm)

# SETUP & DATA LOADING
print("Loading camera data...")
data(camera)

# The camera dataset is already a list of lists in the exact 'lgtdata' format
# expected by rhierMnlRwMixture. No manual createX() looping is required
lgtdata <- camera

n_units <- length(lgtdata)
p <- 5 # 5 alternatives: Canon, Sony, Nikon, Panasonic, Fuji
k_dim <- ncol(lgtdata[[1]]$X) # 10 parameters

print(paste("Loaded", n_units, "households with", k_dim, "parameters."))

# MODEL CONFIGURATION
# Matches the Python MCMC specs: 40,000 posterior + 1,000 warmup
R_total <- 41000
burn_in <- 1000
keep_every <- 4

# Data List (Notice Z is omitted since there are no demographic covariates)
data_list <- list(p = p, lgtdata = lgtdata)

# Prior Config to match Liesel setup
# Since there is no Z matrix, bayesm implicitly models an intercept
# We supply a 1x1 A matrix for that intercept variance
Prior <- list(
    ncomp = 1,
    A = matrix(0.01),
    nu = k_dim + 3,
    V = (k_dim + 3) * diag(k_dim)
)

Mcmc <- list(R = R_total, keep = keep_every, nprint = 500)

# RUN MODEL
print("Running rhierMnlRwMixture...")
set.seed(123)
out <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)

# POST-PROCESSING
print("Processing posterior samples...")
R_draws <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

param_names <- c("Canon", "Sony", "Nikon", "Panasonic", "Pixels", "Zoom", "Video", "Swivel", "Wifi", "Price")

# Extract the global mean vector (mu) safely
mu_list <- lapply(out$nmix$compdraw, function(x) x[[1]]$mu)
mu_matrix <- do.call(rbind, mu_list)
mu_draws <- mu_matrix[keep_idx, , drop = FALSE] # drop=FALSE prevents vector conversion
colnames(mu_draws) <- param_names

# Extract and calculate the Covariance matrices from the inverse Cholesky roots
cov_list <- lapply(out$nmix$compdraw, function(x) chol2inv(x[[1]]$rooti))

# Stack them row by row, subset by the burn-in index, and convert to dataframe
cov_draws <- do.call(rbind, lapply(cov_list, as.vector))
cov_draws_df <- as.data.frame(cov_draws[keep_idx, , drop = FALSE])

# Because Z = NULL, Deltadraw is often empty/dropped by bayesm
# In this homogenous setup, the mixture mean (Mu) IS the Delta
if (!is.null(out$Deltadraw) && length(out$Deltadraw) > 0) {
    # If bayesm happens to return it, coerce safely
    delta_matrix <- matrix(out$Deltadraw, ncol = length(param_names), byrow = TRUE)
    delta_draws <- delta_matrix[keep_idx, , drop = FALSE]
    colnames(delta_draws) <- param_names
} else {
    # Fallback: Just duplicate mu_draws so the downstream Python comparisons work
    delta_draws <- mu_draws
}

beta_kept <- out$betadraw[, , keep_idx, drop = FALSE]

# SAVE OUTPUT
script_dir <- this.path::here()

# Append "Data" to the export directory path
export_dir <- file.path(dirname(script_dir), "camera", "Data")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

mu_draws_df <- as.data.frame(mu_draws)
delta_draws_df <- as.data.frame(delta_draws)

# Reorder betadraw array to match the Python Liesel (Units, Parameters, Draws) logic if needed
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

file_name <- "bayesm_output_camera.RData"
save_path <- file.path(export_dir, file_name)
save(mu_draws_df, delta_draws_df, cov_draws_df, beta_reordered, n_samples, file = save_path)
print(paste("SUCCESS: Saved to", save_path))
