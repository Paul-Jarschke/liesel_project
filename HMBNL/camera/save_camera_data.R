library(bayesm)
library(jsonlite)

# 1. Get the directory of the current script
# (Note: This works when 'Sourcing' the script in RStudio)
script_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile))

# 2. Define the output path (directly in the script directory)
output_path <- file.path(script_dir, "camera_data.json")

# 3. Load the data
data(camera)

# 4. Convert and write
json_data <- toJSON(camera, matrix = "rowmajor", auto_unbox = TRUE)
write(json_data, output_path)

# Confirmation message
cat("File saved to:", output_path)