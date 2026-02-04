library(bayesm)
library(jsonlite)

# Directory of this script
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))

# Project root = parent of PackageComparison
project_root <- normalizePath(file.path(script_dir, ".."))

# Existing Data directory
data_dir <- file.path(project_root, "Data")

# Output path for JSON file
output_path <- file.path(data_dir, "camera_data.json")

# Load bayesm camera data
data(camera)

# Convert to JSON and write to file
json_data <- toJSON(camera, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

# Visual parameter validation
print(paste("Number of parameters (K):", ncol(camera[[1]]$X)))
