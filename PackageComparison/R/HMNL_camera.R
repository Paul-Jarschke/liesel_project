# ==============================================================================
# 1. SETUP & MODEL EXECUTION
# ==============================================================================
library(this.path)
library(bayesm)
data(camera)

# Configuration ---
R_total    <- 11000   # Total iterations    (including burn-in!)
burn_in    <- 1000    # Burn-in to discard
keep_every <- 1       # Thinning parameter  (No thinning)

# Prepare Data Lists
data_list  <- list(lgtdata = camera, p = 5)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = R_total, keep = keep_every, nprint = 500)

# Run Model with parameter estimation
print("Running rhierMnlRwMixture...")
out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)


# ==============================================================================
# 2. PROCESS DRAWS
# ==============================================================================
print("Processing posterior samples...")

# Define Sample Indices
# bayesm returns all R draws, burn-in must be removed manually
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)         # This is the number that goes in filename

# Dimensions
n_units  <- dim(out$betadraw)[1]      # 332
n_params <- dim(out$betadraw)[2]      # 10
param_names <- c("Canon", "Sony", "Nikon", "Panasonic", "Pixels", "Zoom",
                 "Video", "Swivel", "Wifi", "Price")


# ------------------------------------------------------------------------------
# Process Mu ("Population-level" parameters)
# ------------------------------------------------------------------------------
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# ------------------------------------------------------------------------------
# Process Beta (Unit-level parameters)
# ------------------------------------------------------------------------------
# Subset draws to remove burn-in
beta_kept <- out$betadraw[,,keep_idx]

# Create Index Columns
beta_df_indices <- data.frame(
  Unit_ID = rep(1:n_units, times = n_samples),
  Draw_ID = rep(1:n_samples,  each  = n_units)
)

# Reshape 3D Array to 2D Matrix
# beta_kept is [Units, Params, Draws]
# Reorder to [Units, Draws, Params] so we can flatten the first two dims
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

# Flatten
beta_mat <- matrix(beta_reordered, nrow = n_units * n_samples, ncol = n_params)
colnames(beta_mat) <- param_names

# Combine Indices and Data
final_beta_df <- cbind(beta_df_indices, beta_mat)

# ==============================================================================
# 3. SAVE OUTPUT
# ==============================================================================

# Determine the directory where THIS script is located
# Works for source(), RStudio, and command line
script_dir <- this.path::here()
print(paste("Script directory determined as:", script_dir))

# Construct path: .../liesel_project/PackageComparison/Data
export_dir <- file.path(dirname(script_dir), "Data")

# Create directory if it doesn't exist
if (!dir.exists(export_dir)) {
  dir.create(export_dir, recursive = TRUE)
  print(paste("Created directory:", export_dir))
}

# Prepare Data
mu_draws_df   <- as.data.frame(mu_final)
beta_draws_df <- final_beta_df
n_samples_val <- as.integer(n_samples) 

# Save
file_name <- paste0("R_draws_", n_samples, "_samples.RData")
save_path <- file.path(export_dir, file_name)

save(mu_draws_df, beta_draws_df, n_samples_val, file = save_path)

print(paste("SUCCESS: Saved to", save_path))