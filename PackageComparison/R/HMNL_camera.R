# ==============================================================================
# 1. SETUP & DATA
# ==============================================================================
library(bayesm)
library(ggplot2)
library(corrplot)

# Load dataset
data(camera)

# Prepare Data Structure for bayesm
# (This transforms the list-of-lists into the format bayesm needs)
data_list  <- list(lgtdata = camera, p = 5)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = 2000, nprint = 0) # Reduced nprint to keep console clean

# ==============================================================================
# 2. MODELING
# ==============================================================================

# 1. Define the specific path inside the R folder
# This creates the path: PackageComparison/R/model_cache
cache_dir <- file.path("PackageComparison", "R", "model_cache")

# 2. Safety Check: Create the directory if it doesn't exist
if (!dir.exists(cache_dir)) {
  # recursive = TRUE ensures it builds the whole path if parts are missing
  dir.create(cache_dir, recursive = TRUE)
  print(paste("Created directory:", cache_dir))
}

# 3. Define the full file path
model_path <- file.path(cache_dir, "HMNL_camera_model.rds")

# 4. Load or Run
if (file.exists(model_path)) {
  print(paste("Loading saved model from:", model_path))
  out <- readRDS(model_path)
} else {
  print("Running rhierMnlRwMixture (this may take a moment)...")
  out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)
  saveRDS(out, model_path)
  print(paste("Model saved to:", model_path))
}

# ==============================================================================
# 3. ANALYSIS & VISUALIZATION
# ==============================================================================

# A. SETUP ---------------------------------------------------------------------
library(ggplot2)
library(reshape2)
library(corrplot)

# 1. Extract Draws (Discard Burn-in)
# 'out' contains the raw MCMC draws. We need to process them.
R_draws  <- length(out$nmix$compdraw)
burn_in  <- 0.2 * R_draws
keep_idx <- (burn_in + 1):R_draws

# 2. Define Parameter Names
# Based on the camera dataset: 5 Brands + 5 Features
# (Check colnames(camera[[1]]$X) to confirm order if unsure)
param_names <- c("Canon", "Sony", "Nikon", "Panasonic", "Pixels", "Zoom", "Video", "Swivel", "Wifi", "Price")

# 3. Extract Individual Betas (Heterogeneity)
# Dimensions: (Units x Params x Draws). We take the mean over draws for each unit.
beta_ind_means <- apply(out$betadraw[,,keep_idx], c(1,2), mean)
colnames(beta_ind_means) <- param_names


# B. POPULATION PREFERENCES (Global Means) -------------------------------------
# Extract Mu draws (Population Mean)
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# Calculate Summary Stats for Plotting
mu_summary <- data.frame(
  Parameter = factor(param_names, levels = param_names),
  Mean = colMeans(mu_final),
  Lower = apply(mu_final, 2, quantile, probs = 0.025),
  Upper = apply(mu_final, 2, quantile, probs = 0.975)
)

# Plot 1: Population Means (The "Caterpillar" Plot)
p1 <- ggplot(mu_summary, aes(x = Parameter, y = Mean)) +
  geom_point(size = 3, color = "#2E86C1") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "#2E86C1") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Average Consumer Preferences (Population Means)",
       subtitle = "Points are posterior means; bars are 95% credible intervals",
       y = "Utility Value", x = "")
print(p1)


# C. HETEROGENEITY (Distribution of Preferences) -------------------------------
# Plot 2: Boxplots of Individual Betas
# Reshape for ggplot
df_beta <- melt(beta_ind_means)
colnames(df_beta) <- c("Respondent", "Parameter", "Utility")

p2 <- ggplot(df_beta, aes(x = Parameter, y = Utility)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7, outlier.shape = 1, outlier.alpha = 0.3) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Heterogeneity in Preferences",
       subtitle = "Distribution of individual-level parameters across respondents",
       y = "Utility Value", x = "")
print(p2)


# D. CORRELATIONS (Trade-offs) -------------------------------------------------
# We calculate the correlation of the *individual* betas to see how prefs co-vary.
cor_matrix <- cor(beta_ind_means)

# Plot 3: Correlation Heatmap
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", diag = FALSE, addCoef.col = "black", 
         number.cex = 0.6, tl.cex = 0.8,
         title = "Preference Correlations", mar=c(0,0,2,0))


# E. PREDICTIVE FIT (Hit Rate) -------------------------------------------------
# Predict the choice for each task based on the individual's estimated beta.

hits <- 0
total_obs <- 0

# Loop over every respondent
for (i in seq_along(camera)) {
  y_true <- camera[[i]]$y
  X_raw  <- camera[[i]]$X
  beta_i <- beta_ind_means[i, ] # Use this person's specific beta
  
  # The data is stacked. We need to process it task-by-task.
  n_alts  <- 5 # 5 brands per choice task
  n_tasks <- length(y_true)
  
  for (t in 1:n_tasks) {
    # Isolate the 5 alternatives for this specific task
    row_start <- (t-1)*n_alts + 1
    row_end   <- t*n_alts
    X_task    <- X_raw[row_start:row_end, ]
    
    # Calculate Utility: U = X * beta
    utils <- X_task %*% beta_i
    
    # Prediction: The alternative with the highest utility
    pred_choice <- which.max(utils)
    
    if (pred_choice == y_true[t]) {
      hits <- hits + 1
    }
    total_obs <- total_obs + 1
  }
}

hit_rate <- round(hits / total_obs, 4) * 100
cat("\n========================================\n")
cat(" MODEL PERFORMANCE \n")
cat("========================================\n")
cat("Total Observations Evaluated:", total_obs, "\n")
cat("Correct Predictions:", hits, "\n")
cat("In-Sample Hit Rate:", hit_rate, "%\n")
cat("Random Chance Baseline: 20.00 %\n")
cat("========================================\n")

# ==============================================================================
# 4. EXPORT FOR PYTHON COMPARISON
# ==============================================================================

# Ensure beta_ind_means is calculated (as per previous script)
# Dimensions: 332 rows (people) x 10 columns (parameters)
beta_ind_means <- apply(out$betadraw[,,keep_idx], c(1,2), mean)
colnames(beta_ind_means) <- param_names

# Define path (same logic as before)
export_dir <- file.path("PackageComparison", "Data") 
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

# Save to CSV
write.csv(beta_ind_means, file.path(export_dir, "R_beta_estimates.csv"), row.names = FALSE)
print("R estimates exported to Data/R_beta_estimates.csv")

# ==============================================================================
# 5. DETAILED EXPORT (Run in R)
# ==============================================================================

# A. Export Individual Standard Deviations (Uncertainty)
# ------------------------------------------------------------------------------
# We need to know if R and Python agree on how "sure" they are about each person.
beta_ind_sds <- apply(out$betadraw[,,keep_idx], c(1,2), sd)
colnames(beta_ind_sds) <- param_names
write.csv(beta_ind_sds, file.path(export_dir, "R_beta_sds.csv"), row.names = FALSE)

# B. Export Global Population Means (Mu)
# ------------------------------------------------------------------------------
# This compares the "average consumer" profile between the two models.
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_mean  <- colMeans(mu_draws[keep_idx, ])
# Save as a simple CSV with one row
write.csv(t(mu_mean), file.path(export_dir, "R_mu_means.csv"), row.names = FALSE)

print("Detailed exports (SDs and Global Means) saved to Data folder.")