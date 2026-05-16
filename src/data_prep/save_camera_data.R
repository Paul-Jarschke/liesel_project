# src/data_prep/save_camera_data.R
library(bayesm)
library(jsonlite)

# Path handling to go up two levels: src/data_prep -> src -> project_root -> data/processed
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))
project_root <- dirname(dirname(script_dir))
output_dir <- file.path(project_root, "data", "processed")

# Create the directory if it doesn't exist
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Create output path
output_path <- file.path(output_dir, "camera_data.json")

# Load the data
data(camera)

# Convert and write
json_data <- toJSON(camera, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

cat("Camera data successfully saved at:", output_path, "\n")
