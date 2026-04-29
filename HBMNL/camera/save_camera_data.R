library(bayesm)
library(jsonlite)

# Get the directory of the current script
# (Note: This works when 'Sourcing' the script in RStudio)
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))

# Define the output path (directly in the script directory)
output_path <- file.path(script_dir, "camera_data.json")

# Load the data
data(camera)

# Convert and write
json_data <- toJSON(camera, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

# Confirmation message
cat("File saved to:", output_path)
