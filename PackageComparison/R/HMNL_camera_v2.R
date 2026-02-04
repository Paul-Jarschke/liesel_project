# ==============================================================================
# 1. SETUP & MODEL EXECUTION
# ==============================================================================
library(bayesm)
data(camera)

# --- Configuration ---
R_total    <- 41000   # Total iterations
burn_in    <- 1000    # Burn-in to discard
keep_every <- 1       # Thinning parameter

# Prepare Data Lists
data_list  <- list(lgtdata = camera, p = 5)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = R_total, keep = keep_every, nprint = 500)

# Run Model (Always runs fresh, no caching)
print("Running rhierMnlRwMixture...")
out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)

# ==============================================================================
# 2. PROCESS DRAWS
# ==============================================================================
print("Processing posterior samples...")

# Define Sample Indices
# Note: bayesm returns all R draws; we must manually remove burn-in
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)         # This is the number that goes in filename

# Dimensions
n_units  <- dim(out$betadraw)[1]      # 332
n_params <- dim(out$betadraw)[2]      # 10
param_names <- c("Canon", "Sony", "Nikon", "Panasonic", "Pixels", "Zoom",
                 "Video", "Swivel", "Wifi", "Price")

# ------------------------------------------------------------------------------
# A. Process Mu (Population Means)
# ------------------------------------------------------------------------------
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# ------------------------------------------------------------------------------
# B. Process Beta (Individual Part-Worths)
# ------------------------------------------------------------------------------
# 1. Subset draws to remove burn-in
beta_kept <- out$betadraw[,,keep_idx]

# 2. Create Index Columns
# We want a long format: Unit 1 all draws, then Unit 2 all draws?
# OR Unit 1 Draw 1, Unit 2 Draw 1?
# The standard logic here is usually stacking Units within Draws or vice versa.
# Here we align with the matrix flattening order below.
beta_df_indices <- data.frame(
  Unit_ID = rep(1:n_units, times = n_samples),
  Draw_ID = rep(1:n_samples,  each  = n_units)
)

# 3. Reshape 3D Array to 2D Matrix
# beta_kept is [Units, Params, Draws]
# We reorder to [Units, Draws, Params] so we can flatten the first two dims
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

# 4. Flatten
beta_mat <- matrix(beta_reordered, nrow = n_units * n_samples, ncol = n_params)
colnames(beta_mat) <- param_names

# 5. Combine Indices and Data
final_beta_df <- cbind(beta_df_indices, beta_mat)

# ==============================================================================
# 3. SAVE OUTPUT WITH DYNAMIC FILENAME
# ==============================================================================
export_dir <- file.path("PackageComparison", "Data") 
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

# Create the final list object
output_object <- list(
  mu_draws   = mu_final,
  beta_draws = final_beta_df,
  n_samples  = n_samples
)

# Construct Filename with number of samples
file_name <- paste0("R_draws_", n_samples, "_samples.rds")
save_path <- file.path(export_dir, file_name)

# Save
saveRDS(output_object, save_path)

print("------------------------------------------------")
print(paste("Processing Complete."))
print(paste("File saved to:", save_path))
print(paste("Total valid samples:", n_samples))