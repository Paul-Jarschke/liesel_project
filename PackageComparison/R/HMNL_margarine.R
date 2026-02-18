# ==============================================================================
# 1. SETUP & MODEL EXECUTION
# ==============================================================================
library(this.path)
library(bayesm)

data(margarine)
cp <- margarine$choicePrice

# Configuration ---
R_total    <- 41000   # Total iterations
burn_in    <- 1000    # Burn-in
keep_every <- 1       # Thinning

# ------------------------------------------------------------------------------
# Data Preprocessing (Constructing lgtdata for margarine)
# ------------------------------------------------------------------------------
print("Preprocessing margarine data...")
hhids <- unique(cp$hhid)
n_units <- length(hhids)
lgtdata <- list()
n_alts  <- 10
n_params <- 10

for (i in 1:n_units) {
  id <- hhids[i]
  subset_dat <- cp[cp$hhid == id, ]
  
  y <- subset_dat$choice
  n_obs_i <- nrow(subset_dat)
  X_i <- matrix(0, nrow = n_obs_i * n_alts, ncol = n_params)
  
  for (r in 1:n_obs_i) {
    prices <- as.numeric(subset_dat[r, 3:12])
    mat_r <- matrix(0, nrow = n_alts, ncol = n_params)
    mat_r[1:9, 1:9] <- diag(9)
    mat_r[, 10] <- prices
    X_i[((r - 1) * n_alts + 1):(r * n_alts), ] <- mat_r
  }
  lgtdata[[i]] <- list(y = y, X = X_i)
}

# Prepare Lists
data_list  <- list(lgtdata = lgtdata, p = 10)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = R_total, keep = keep_every, nprint = 500)

# Run Model
print("Running rhierMnlRwMixture on margarine data...")
out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)


# ==============================================================================
# 2. PROCESS DRAWS
# ==============================================================================
print("Processing posterior samples...")

# Define Sample Indices
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

# Dimensions
n_units  <- dim(out$betadraw)[1]      
n_params <- dim(out$betadraw)[2] 

# Parameter Names
# 9 ASCs + 1 Price. We name them generically or try to map brands if known.
# Using generic numbering to match the X matrix construction.
param_names <- c("ASC_1", "ASC_2", "ASC_3", "ASC_4", "ASC_5", 
                 "ASC_6", "ASC_7", "ASC_8", "ASC_9", "Price")


# ------------------------------------------------------------------------------
# Process Mu ("Population-level" parameters)
# ------------------------------------------------------------------------------
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# ------------------------------------------------------------------------------
# Process Beta (Unit-level parameters)
# ------------------------------------------------------------------------------
beta_kept <- out$betadraw[,,keep_idx]

# Create Index Columns
beta_df_indices <- data.frame(
  Unit_ID = rep(1:n_units, times = n_samples),
  Draw_ID = rep(1:n_samples,  each  = n_units)
)

# Reshape [Units, Params, Draws] -> [Units, Draws, Params]
beta_reordered <- aperm(beta_kept, c(1, 3, 2))

# Flatten
beta_mat <- matrix(beta_reordered, nrow = n_units * n_samples, ncol = n_params)
colnames(beta_mat) <- param_names

# Combine
final_beta_df <- cbind(beta_df_indices, beta_mat)

# ==============================================================================
# 3. SAVE OUTPUT
# ==============================================================================
script_dir <- this.path::here()
export_dir <- file.path(dirname(script_dir), "Data")

if (!dir.exists(export_dir)) {
  dir.create(export_dir, recursive = TRUE)
}

mu_draws_df   <- as.data.frame(mu_final)
beta_draws_df <- final_beta_df
n_samples_val <- as.integer(n_samples)

file_name <- paste0("bayesm_output_margarine_", n_samples, "_samples.RData")
save_path <- file.path(export_dir, file_name)

save(mu_draws_df, beta_draws_df, n_samples_val, file = save_path)

print(paste("SUCCESS: Saved to", save_path))