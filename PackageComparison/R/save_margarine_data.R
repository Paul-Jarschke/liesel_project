library(bayesm)
library(jsonlite)

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# Directory of this script
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))

# Project root = parent of PackageComparison
project_root <- normalizePath(file.path(script_dir, ".."))

# Data directory
data_dir <- file.path(project_root, "Data")

# Output path
output_path <- file.path(data_dir, "margarine_data.json")

# ==============================================================================
# 2. DATA PROCESSING
# ==============================================================================
# Load bayesm margarine data
data(margarine)
# choicePrice contains: hhid, choice, and 10 price columns
cp <- margarine$choicePrice

# Get unique households
hhids <- unique(cp$hhid)
n_units <- length(hhids)
print(paste("Processing", n_units, "households..."))

# Prepare list for JSON
lgtdata_list <- list()

for (i in 1:n_units) {
  id <- hhids[i]
  # Subset data for this household
  subset_dat <- cp[cp$hhid == id, ]

  # 1. Choice vector
  y <- subset_dat$choice

  # 2. Design Matrix X
  # We have 10 alternatives.
  # We want a model with 9 ASCs (Alternative Specific Constants) and 1 generic Price.
  # Total parameters (K) = 10.
  # The 10th alternative is the base reference for ASCs.

  n_obs_i <- nrow(subset_dat)
  n_alts  <- 10
  n_params <- 10 # 9 ASCs + 1 Price

  # Initialize stacked X matrix: (Obs * Alts) x Params
  X_i <- matrix(0, nrow = n_obs_i * n_alts, ncol = n_params)

  for (r in 1:n_obs_i) {
    # Extract prices for the 10 alternatives for this choice occasion
    # Prices are in columns 3 to 12
    prices <- as.numeric(subset_dat[r, 3:12])

    # Create J x K matrix for this observation
    # Cols 1-9: Identity matrix (ASC indicators)
    # Col 10:   Price vector
    mat_r <- matrix(0, nrow = n_alts, ncol = n_params)

    # Fill ASCs (Diag 1..9), leave 10th row 0 for base
    mat_r[1:9, 1:9] <- diag(9)

    # Fill Price (All rows)
    mat_r[, 10] <- prices

    # Place into stacked matrix
    start_row <- (r - 1) * n_alts + 1
    end_row   <- r * n_alts
    X_i[start_row:end_row, ] <- mat_r
  }

    lgtdata_list[[i]] <- list(y = y, X = X_i)
}

# ==============================================================================
# 3. SAVE OUTPUT
# ==============================================================================
# Convert to JSON and write to file
json_data <- toJSON(lgtdata_list, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

print(paste("Saved margarine data to", output_path))
print(paste("Number of parameters (K):", 10))