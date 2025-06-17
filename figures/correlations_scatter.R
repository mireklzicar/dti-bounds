# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(scales)
library(viridis)
library(rstudioapi)  # For RStudio integration
library(httpgd)
hgd(port = 8067)
hgd_browse()


# --- Path Finding ---
# Get directory of current script
dir = dirname(getSourceEditorContext()$path)

# --- Data Loading ---
# Check if embedding samples files exist
dot_file_path <- paste0(dir, "/embedding_samples_dot.csv")
euclidean_file_path <- paste0(dir, "/embedding_samples_euclidean.csv")

# Function to read and process sample data
read_and_process_samples <- function(file_path, metric_name, value_col) {
  if (file.exists(file_path)) {
    samples_df <- read.csv(file_path)
    
    # Filter for only dimensions 2, 16, and 128
    target_dimensions <- c(2, 16, 128)
    samples_df <- samples_df %>% 
      filter(dimension %in% target_dimensions)
    
    # Process dimensions as factors with proper ordering
    samples_df <- samples_df %>%
      mutate(dimension = factor(dimension, levels = sort(unique(as.numeric(dimension)))))
    
    # Add metric identifier
    samples_df$distance_metric <- metric_name
    
    return(samples_df)
  } else {
    warning(paste("File not found:", file_path))
    return(NULL)
  }
}

# Read sample data if available
samples_dot_df <- read_and_process_samples(dot_file_path, "Dot Product", "dot_product")
samples_euclidean_df <- read_and_process_samples(euclidean_file_path, "Euclidean Distance", "euclidean_distance")

# --- Function to Calculate Metrics ---
calculate_metrics <- function(data, x_col) {
  # Formula for lm
  formula_str <- paste("binding_affinity ~", x_col)
  lm_model <- lm(as.formula(formula_str), data = data)
  predictions <- predict(lm_model, data)
  
  # Calculate error metrics
  mse <- mean((data$binding_affinity - predictions)^2)
  mae <- mean(abs(data$binding_affinity - predictions))
  
  # Calculate correlations
  pearson_cor <- cor(data[[x_col]], data$binding_affinity, method = "pearson")
  spearman_cor <- cor(data[[x_col]], data$binding_affinity, method = "spearman")
  
  # Return a data frame with metrics
  return(data.frame(
    pearson_cor = pearson_cor,
    spearman_cor = spearman_cor,
    mse = mse,
    mae = mae
  ))
}

# --- Create Scatter Plot Function ---
create_scatter_plot <- function(data, x_col, title) {
  # Get metrics for each dataset/dimension combination
  metrics_df <- data %>%
    group_by(dataset, dimension) %>%
    do(calculate_metrics(., x_col)) %>%
    ungroup()
  
  # Join metrics with sample data
  plot_data <- data %>%
    left_join(metrics_df, by = c("dataset", "dimension"))
  
  # Create the plot
  p <- ggplot(plot_data) +
    geom_point(aes(x = .data[[x_col]], y = binding_affinity), 
              alpha = 0.3, size = 0.8, color ="black") +
    facet_grid(dataset ~ dimension, scales = "free_y") + # Only free on y-axis
    xlim(0, 1) + # Restrict x-axis from 0 to 1
    labs(
      x = ifelse(x_col == "dot_product", "Dot Product", "Euclidean Distance"),
      y = "Binding Affinity",
      title = title
    ) +
    theme_bw() +
    theme(
      strip.text.y = element_text(angle = 0, hjust = 0),
      panel.spacing = unit(0.3, "lines"),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  # Add metrics as text annotations to each facet
  p <- p +
    geom_text(
      data = metrics_df,
      aes(
        x = -Inf, y = Inf,
        label = sprintf(
          "P: %.2f\nS: %.2f",
          pearson_cor, spearman_cor
        )
      ),
      hjust = -0.1, vjust = 1.2, size = 2.5
    )
  
  return(p)
}

# --- Create and Print Plots ---
# If we have dot product data, create the dot product plot
if (!is.null(samples_dot_df)) {
  dot_plot <- create_scatter_plot(
    samples_dot_df, 
    "dot_product", 
    "Dot Product Correlation by Dataset and Dimension"
  )
  print(dot_plot)
  ggsave("dot_correlation_facets.png", dot_plot, width = 12, height = 10, dpi = 300)
}

# If we have Euclidean data, create the Euclidean plot
if (!is.null(samples_euclidean_df)) {
  euclidean_plot <- create_scatter_plot(
    samples_euclidean_df, 
    "euclidean_distance", 
    "Euclidean Distance Correlation by Dataset and Dimension"
  )
  print(euclidean_plot)
  ggsave("euclidean_correlation_facets.png", euclidean_plot, width = 12, height = 10, dpi = 300)
}
