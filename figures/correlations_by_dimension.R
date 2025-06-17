# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(scales)
library(viridis)
library(rstudioapi)  # For RStudio integration

# For RStudio integration (uncomment as needed)
library(httpgd)
hgd(port = 8067)
hgd_browse()

# --- Path Finding ---
# Get directory of current script
dir = dirname(getSourceEditorContext()$path)

# --- Data Loading and Preparation ---
# Load both CSV files
dot_data <- read.csv(paste0(dir, "/correlations_dot.csv"))
euclidean_data <- read.csv(paste0(dir, "/correlations_euclidean.csv"))

# Check column names in both dataframes
print("Dot data columns:")
print(names(dot_data))
print("Euclidean data columns:")
print(names(euclidean_data))

# Add a column to identify the correlation type
dot_data$distance_metric <- "Dot Product"
euclidean_data$distance_metric <- "Euclidean Distance"

# Using dplyr's bind_rows which is more flexible with column names
# It will fill missing columns with NA
data_df <- dplyr::bind_rows(dot_data, euclidean_data)

# Verify the combined dataset structure
print("Combined data structure:")
print(str(data_df))

# Process dimension column for proper ordering
data_df <- data_df %>%
  # Ensure dimension is numeric first
  mutate(dimension = as.numeric(as.character(dimension))) %>%
  # Convert to factor with proper order
  mutate(dimension = factor(dimension, levels = sort(unique(dimension))))

# Reshape data for faceting (convert to long format)
data_long <- data_df %>%
  # Convert from wide to long format
  pivot_longer(
    cols = c(pearson_correlation, spearman_correlation),
    names_to = "correlation_type", 
    values_to = "correlation_value"
  ) %>%
  # Create readable labels for metrics
  mutate(
    correlation_type = case_when(
      correlation_type == "pearson_correlation" ~ "Pearson",
      correlation_type == "spearman_correlation" ~ "Spearman",
      TRUE ~ correlation_type
    )
  )

# --- Plotting ---
# Create the faceted plot
final_plot <- ggplot(
    data_long, 
    aes(x = dimension, y = correlation_value, group = dataset, color = dataset)
  ) +
  # Add lines and points
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  # Create 2x2 grid of facets
  facet_grid(
    distance_metric ~ correlation_type,  # Row ~ Column format
    scales = "fixed"                     # Ensures shared axes
  ) +
  # Add labels
  labs(
    x = "Embedding Dimension",
    y = "Correlation Coefficient",
    title = "Contrastive Prediction of Binding Affinities"
  ) +
  # Set theme and styling
  theme_bw(base_size = 12) +
  theme(
    axis.title = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),  # No legend title
    legend.text = element_text(size = 12),
    strip.background = element_rect(fill = "lightgrey"),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.minor = element_blank()  # Remove minor grid for cleaner look
  ) +
  # Set color scheme
  scale_color_viridis_d(drop = FALSE) +
  # Set y-axis limits
  ylim(0.6, 1.0)  # Shared y-axis limits for all facets

# Print the final plot
print(final_plot)

# Optional: Save the plot
# ggsave("correlation_analysis.png", final_plot, width = 10, height = 8, dpi = 300)