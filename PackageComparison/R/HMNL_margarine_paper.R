# ==========================
# 1. SETUP & MODEL EXECUTION
# ==========================
library(this.path)
library(bayesm)

data(margarine)

# Configuration
R_total    <- 41000   # Total iterations
burn_in    <- 1000    # Burn-in
keep_every <- 1       # Thinning

# Data Preprocessing (Constructing lgtdata & Z for margarine)
print("Preprocessing margarine data...")

select = c(1:5,7)  ## select brands
chPr = as.matrix(margarine$choicePrice)

# make sure to log prices
chPr = cbind(chPr[,1], chPr[,2], log(chPr[,2+select]))
demos = as.matrix(margarine$demos[,c(1,2,5)])

# remove obs for other alts
chPr = chPr[chPr[,2] <= 7,]
chPr = chPr[chPr[,2] != 6,]

# recode choice
chPr[chPr[,2] == 7, 2] = 6

hhidl = levels(as.factor(chPr[,1]))
lgtdata = list()
nlgt = length(hhidl)
p = length(select)  ## number of choice alts (6)

ind = 1
for (i in 1:nlgt) {
  nobs = sum(chPr[,1]==hhidl[i])
  if(nobs >= 5) {
    data = chPr[chPr[,1]==hhidl[i],]
    y = data[,2]
    names(y) = NULL

    # createX generates alternative specific intercepts (INT=TRUE, base=1)
    X = createX(p=p, na=1, Xa=data[,3:8], nd=NULL, Xd=NULL, INT=TRUE, base=1)
    lgtdata[[ind]] = list(y=y, X=X, hhid=hhidl[i])
    ind = ind + 1
  }
}

# extract demos corresponding to hhs in lgtdata
Z = NULL
nlgt = length(lgtdata)
for(i in 1:nlgt){
  Z = rbind(Z, demos[demos[,1]==lgtdata[[i]]$hhid, 2:3])
}

# take log of income and family size and demean
Z = log(Z)
Z[,1] = Z[,1] - mean(Z[,1])
Z[,2] = Z[,2] - mean(Z[,2])

# Prepare Lists
data_list  <- list(p = p, lgtdata = lgtdata, Z = Z)
prior_list <- list(ncomp = 1)
mcmc_list  <- list(R = R_total, keep = keep_every, nprint = 500)

# Run Model
print("Running rhierMnlRwMixture on margarine data...")
out <- rhierMnlRwMixture(Data = data_list, Prior = prior_list, Mcmc = mcmc_list)


# ================
# 2. PROCESS DRAWS
# ================
print("Processing posterior samples...")

# Define Sample Indices
R_draws  <- length(out$nmix$compdraw)
keep_idx <- (burn_in + 1):R_draws
n_samples <- length(keep_idx)

# Dimensions
n_units  <- dim(out$betadraw)[1]
n_params <- dim(out$betadraw)[2]

# Parameter Names
# Since p=6 and base=1, we have 5 Brands (for alternatives 2 to 6) and 1 logged Price variable
param_names <- c("Blue Bonnett", "Fleischmanns", "House", "Generic", "Shead Spread Tub", "LogPrice")


# Process Mu ("Population-level" base parameters)
mu_draws <- t(sapply(out$nmix$compdraw, function(x) x[[1]]$mu))
mu_final <- mu_draws[keep_idx, ]
colnames(mu_final) <- param_names

# Process Beta (Unit-level parameters)
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


# Process Delta (Demographic coefficients matrix)
# Since Z is included in this model, rhierMnlRwMixture produces Deltadraw
delta_draws_df <- as.data.frame(out$Deltadraw[keep_idx, ])

# ==============
# 3. SAVE OUTPUT
# ==============
script_dir <- this.path::here()
export_dir <- file.path(dirname(script_dir), "Data")

if (!dir.exists(export_dir)) {
  dir.create(export_dir, recursive = TRUE)
}

mu_draws_df   <- as.data.frame(mu_final)
beta_draws_df <- final_beta_df
n_samples_val <- as.integer(n_samples)

file_name <- paste0("bayesm_output_margarine_paper_", n_samples, "_samples.RData")
save_path <- file.path(export_dir, file_name)

# Now additionally saving delta_draws_df
save(mu_draws_df, beta_draws_df, delta_draws_df, n_samples_val, file = save_path)

print(paste("SUCCESS: Saved to", save_path))