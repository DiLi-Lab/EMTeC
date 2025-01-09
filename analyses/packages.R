# List of required packages
required_packages <- c(
  "ggplot2",
  "dplyr",
  "viridis",
  "tidyr",
  "broom",
  "gsubfn",
  "tidyverse",
  "boot",
  "broman",
  "rsample",
  "plotrix",
  "ggrepel",
  "ggcorrplot",
  "lmerTest",
  "patchwork",
  "cowplot",
  "yaml"
)

# Install any missing packages
install_missing_packages <- function(packages) {
  installed_packages <- rownames(installed.packages())
  missing_packages <- packages[!(packages %in% installed_packages)]
  
  if (length(missing_packages) > 0) {
    install.packages(missing_packages)
  }
}

install_missing_packages(required_packages)

# Load all the required packages
lapply(required_packages, library, character.only = TRUE)

# Optional: print a message indicating that all packages are loaded
cat("All required packages are installed and loaded.\n")
