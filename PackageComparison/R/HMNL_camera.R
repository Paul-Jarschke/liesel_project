# ==============================================================================
# 1. SETUP & MODELING
# ==============================================================================
library(bayesm)
data(camera)

# Settings (1000 Burn-in + 2000 Samples)
data_list  <- list(lgtdata = camera, p = 5)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = 10000, keep = 1, nprint = 0) 

# Run or Load
cache_dir <- file.path("PackageComparison", "R", "model_cache")
if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
model_path <- file.path(cache_dir, "HMNL_camera_model.rds")

if (file.exists(model_path)) {
  print("Loading saved model...")
  out <- readRDS(model_path)
} else {
  print("Running rhierMnlRwMixture...")
  out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)
  saveRDS(out, model_path)
}

# ==============================================================================
# 2. EXTRACT RAW DRAWS
# ==============================================================================
print("Extracting full posterior samples...")

# Define Indices (Strict 1000 Burn-in)
burn_in  <- 1000
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_kept   <- length(keep_idx)          # Should be 2000
n_units  <- dim(out$betadraw)[1]      # 332
n_params <- dim(out$betadraw)[2]      # 10

param_names <- c("Canon", "Sony", "Nikon", "Panasonic", "Pixels", "Zoom", "Video", "Swivel", "Wifi", "Price")

# ------------------------------------------------------------------------------
# A. EXPORT POPULATION MEAN DRAWS (MU)
# ------------------------------------------------------------------------------
# Structure: Rows = Draws (2000), Cols = Parameters (10)
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# ------------------------------------------------------------------------------
# B. EXPORT INDIVIDUAL BETA DRAWS (FULL DISTRIBUTION)
# ------------------------------------------------------------------------------
# Raw Structure: (Units, Params, Draws)
# Target Structure: CSV with Columns [Unit_ID, Draw_ID, Param1, Param2...Param10]
# This creates a file with roughly 332 * 2000 = 664,000 rows.

# 1. Subset to keep only valid draws
beta_kept <- out$betadraw[,,keep_idx] 

# 2. Reshape efficiently
# We want rows to represent (Unit x Draw) combinations
# aperm reorders to (Draws, Units, Params) so we can flatten the first two
beta_reordered <- aperm(beta_kept, c(3, 1, 2)) 

# 3. Flatten to Matrix (Rows = Draws*Units, Cols = Params)
beta_flat <- matrix(beta_reordered, ncol = n_params)
colnames(beta_flat) <- param_names

# 4. Create Index Columns for reconstruction in Python
# Rep 1..332 each repeated 2000 times? No, because of aperm order (Draws outer, Units inner)
# Order is: Draw1-Unit1, Draw1-Unit2 ... Draw2-Unit1 ...
unit_ids <- rep(1:n_units, each = n_kept) # Wrong for the aperm above?
# Let's double check order: aperm(c(3,1,2)) -> Dimension 1 is Draw, Dim 2 is Unit.
# Matrix fills by column (Dim 1 varies fastest). 
# So the order in `beta_flat` varies by Draw first? No, R fills columns first.
# Actually, let's just use explicit reshaping to be safe.

# SAFE RESHAPING APPROACH:
# Melt to long format -> (Unit, Param, Draw)
dimnames(beta_kept) <- list(NULL, param_names, NULL)
# We handle this manually to ensure CSV structure is perfect:
beta_df <- data.frame(
  Unit_ID = rep(1:n_units, times = n_kept),     # 1, 2, ... 332, 1, 2 ...
  Draw_ID = rep(1:n_kept,  each  = n_units)      # 1, 1, ... 1,   2, 2 ...
)
# We need to fill the parameters. 
# We need the data ordered by Draw (outer) then Unit (inner)
beta_reordered <- aperm(beta_kept, c(1, 3, 2)) # (Units, Draws, Params)
# Now flatten preserving the last dimension (Params)
beta_mat <- matrix(beta_reordered, nrow = n_units * n_kept, ncol = n_params)
colnames(beta_mat) <- param_names

# Combine
final_beta_df <- cbind(beta_df, beta_mat)

# ------------------------------------------------------------------------------
# 3. SAVE TO DISK
# ------------------------------------------------------------------------------
export_dir <- file.path("PackageComparison", "Data") 
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

write.csv(mu_final,      file.path(export_dir, "R_mu_draws.csv"),   row.names = FALSE)
write.csv(final_beta_df, file.path(export_dir, "R_beta_draws.csv"), row.names = FALSE)

print("Export Complete.")
print(paste("Saved R_mu_draws.csv   :", nrow(mu_final), "draws"))
print(paste("Saved R_beta_draws.csv :", nrow(final_beta_df), "rows (Units x Draws)"))