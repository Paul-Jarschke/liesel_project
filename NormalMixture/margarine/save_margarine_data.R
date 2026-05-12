library(bayesm)
library(jsonlite)

# =====================
# PATH CONFIGURATION
# =====================
# Get the directory where this script is located (works when using source())
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))

# Set the output path to be exactly in that same directory
output_path <- file.path(script_dir, "margarine_data.json")

# ==================
# DATA PROCESSING
# ==================
data(margarine)
cp <- margarine$choicePrice
demos <- margarine$demos # Load demographics

hhids <- unique(cp$hhid)
n_units <- length(hhids)
print(paste("Processing", n_units, "households..."))

lgtdata_list <- list()

for (i in 1:n_units) {
    id <- hhids[i]
    subset_dat <- cp[cp$hhid == id, ]

    # Extract demographics for this household (Income = col 2, Fam_Size = col 5)
    demo_subset <- demos[demos$hhid == id, ]
    if (nrow(demo_subset) > 0) {
        Z_raw <- as.numeric(demo_subset[1, c(2, 5)])
    } else {
        Z_raw <- c(NA, NA)
    }

    y <- subset_dat$choice
    n_obs_i <- nrow(subset_dat)
    n_alts <- 10
    n_params <- 10

    X_i <- matrix(0, nrow = n_obs_i * n_alts, ncol = n_params)

    for (r in 1:n_obs_i) {
        prices <- as.numeric(subset_dat[r, 3:12])
        mat_r <- matrix(0, nrow = n_alts, ncol = n_params)
        mat_r[1:9, 1:9] <- diag(9)
        mat_r[, 10] <- prices
        start_row <- (r - 1) * n_alts + 1
        end_row <- r * n_alts
        X_i[start_row:end_row, ] <- mat_r
    }

    # Append Z_raw to the list
    lgtdata_list[[i]] <- list(y = y, X = X_i, Z_raw = Z_raw)
}

# ==============
# 3. SAVE OUTPUT
# ==============
json_data <- toJSON(lgtdata_list, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

print(paste("Saved margarine data to", output_path))
print(paste("Number of parameters (K):", 10))
