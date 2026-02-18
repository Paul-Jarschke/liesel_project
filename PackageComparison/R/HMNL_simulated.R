# ==============================================================================
# 1. SETUP & MODEL EXECUTION
# ==============================================================================
library(this.path)
library(bayesm)
library(jsonlite)

# Determine script and data paths
script_dir <- this.path::here()
json_path <- file.path(dirname(script_dir), "Data", "simulated_data.json")

print(paste("Loading simulated data from:", json_path))

# Load Data
json_data <- jsonlite::read_json(json_path, simplifyVector = TRUE)
raw_dataset <- json_data$dataset

# Construct 'lgtdata' list for bayesm
# bayesm expects a list where each element has:
#   $y: vector of choices
#   $X: matrix of dimension (n_obs * n_alts) x n_params
lgtdata <- list()
n_units <- nrow(raw_dataset)

for (i in 1:n_units) {
  y_raw <- raw_dataset$y[[i]]
  X_raw <- raw_dataset$X[[i]] # Shape is [n_obs, n_alts, n_params]
  
  # Check dimensions
  n_obs <- dim(X_raw)[1]
  n_alts <- dim(X_raw)[2]
  n_params <- dim(X_raw)[3]
  
  # Flatten X for bayesm
  # We need to stack the 'n_alts' rows for each observation.
  # The JSON X_raw is [obs, alt, param].
  # We iterate over observations and bind the alt x param matrices.
  X_list_i <- lapply(1:n_obs, function(r) X_raw[r,,])
  X_mat <- do.call(rbind, X_list_i)
  
  lgtdata[[i]] <- list(y = y_raw, X = X_mat)
}

# Configuration ---
R_total    <- 41000   # Total iterations    (including burn-in!)
burn_in    <- 1000    # Burn-in to discard
keep_every <- 1       # Thinning parameter  (No thinning)

# Prepare Data Lists
# p = 5 alternatives (from simulated data generation)
data_list  <- list(lgtdata = lgtdata, p = 5)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = R_total, keep = keep_every, nprint = 500)

# Run Model with parameter estimation
print("Running rhierMnlRwMixture on simulated data...")
out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)


# ==============================================================================
# 2. PROCESS DRAWS
# ==============================================================================
print("Processing posterior samples...")

# Define Sample Indices
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)         # This is the number that goes in filename

# Dimensions
n_units  <- dim(out$betadraw)[1]      
n_params <- dim(out$betadraw)[2]      
# Parameter names based on save_simulated_data.py logic:
# First 4 are Brands, 5th is Price
param_names <- c("Brand1", "Brand2", "Brand3", "Brand4", "Price")


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
file_name <- paste0("bayesm_output_simulated_", n_samples, "_samples.RData")
save_path <- file.path(export_dir, file_name)

save(mu_draws_df, beta_draws_df, n_samples_val, file = save_path)

print(paste("SUCCESS: Saved to", save_path))