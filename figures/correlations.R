
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(scales)
library(rstudioapi)
library(httpgd)
hgd(port = 8067)
hgd_browse()

dir = dirname(getSourceEditorContext()$path)
samples_df <- read.csv(paste0(dir, "/embedding_samples_dot.csv"))

# Convert dimension to a factor with proper ordering
samples_df$dimension <- factor(samples_df$dimension, 
                              levels = sort(unique(as.numeric(samples_df$dimension))))

# Create a helper function to calculate metrics
calculate_metrics <- function(data) {
  # Calculate metrics
  lm_model <- lm(binding_affinity ~ dot_product, data = data)
  predictions <- predict(lm_model, data)
  
  # Calculate error metrics
  mse <- mean((data$binding_affinity - predictions)^2)
  mae <- mean(abs(data$binding_affinity - predictions))
  
  # Calculate correlations
  pearson_cor <- cor(data$dot_product, data$binding_affinity, method = "pearson")
  spearman_cor <- cor(data$dot_product, data$binding_affinity, method = "spearman")
  
  # Return a data frame with metrics
  return(data.frame(
    pearson_cor = pearson_cor,
    spearman_cor = spearman_cor,
    mse = mse,
    mae = mae
  ))
}

# Generate metrics for each dataset and dimension
metrics_df <- samples_df %>%
  group_by(dataset, dimension) %>%
  do(calculate_metrics(.)) %>%
  ungroup()

# Join with samples data
plot_data <- samples_df %>%
  left_join(metrics_df, by = c("dataset", "dimension"))

# Get the number of unique datasets and dimensions for the grid layout
datasets <- unique(plot_data$dataset)
dimensions <- levels(plot_data$dimension)

# Extract dataset type (before the underscore) for grouping
plot_data <- plot_data %>%
  mutate(dataset_type = str_extract(dataset, "^[^_]+"))

# Function to create a single plot
create_plot <- function(data) {
  # Get metrics for the plot
  metrics <- data %>%
    select(pearson_cor, spearman_cor, mse, mae) %>%
    distinct() %>%
    as.list()
  
  # Create the plot
  p <- ggplot(data, aes(x = dot_product, y = binding_affinity)) +
    geom_point(alpha = 0.3, size = 0.8, color = "darkblue") +
    #geom_smooth(method = "lm", color = "red", formula = y ~ x) +
    labs(
      subtitle = sprintf(
        "Pearson: %.3f, Spearman: %.3f\nMSE: %.3f, MAE: %.3f",
        metrics$pearson_cor, metrics$spearman_cor, metrics$mse, metrics$mae
      )
    ) +
    theme_minimal() +
    theme(
      plot.subtitle = element_text(size = 8),
      axis.title = element_text(size = 9),
      axis.text = element_text(size = 7)
    )
  
  return(p)
}

# 1. Create faceted plot by dataset (rows) and dimension (columns)
main_plot <- ggplot(plot_data, aes(x = dot_product, y = binding_affinity)) +
  geom_point(alpha = 0.3, size = 0.8) +
  #geom_smooth(method = "lm", color = "red", formula = y ~ x) +
  facet_grid(dataset ~ dimension, scales = "free") +
  labs(
    x = "Dot Product",
    y = "Binding Affinity",
    title = "Embedding Performance by Dataset and Dimension"
  ) +
  theme_bw() +
  theme(
    strip.text.y = element_text(angle = 0, hjust = 0),
    panel.spacing = unit(0.3, "lines")
  )

# Add metrics as text annotations to each facet
main_plot <- main_plot +
  geom_text(
    data = metrics_df,
    aes(
      x = -Inf, y = Inf,
      label = sprintf(
        "P: %.2f\nS: %.2f\nMSE: %.2f\nMAE: %.2f",
        pearson_cor, spearman_cor, mse, mae
      )
    ),
    hjust = -0.1, vjust = 1.2, size = 2.5
  )

# Print the main plot
ggsave("embedding_facet_plot.png", main_plot, width = 12, height = 10, dpi = 300)

# 2. Create a summary plot of correlations by dimension
summary_plot <- metrics_df %>%
  ggplot(aes(x = dimension, y = pearson_cor, group = dataset, color = dataset)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    x = "Dimension",
    y = "Pearson Correlation",
    title = "Correlation by Embedding Dimension"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("correlation_by_dimension.png", summary_plot, width = 10, height = 6, dpi = 300)

# 3. Heatmap of correlations
heatmap_plot <- metrics_df %>%
  ggplot(aes(x = dimension, y = dataset, fill = pearson_cor)) +
  geom_tile() +
  scale_fill_viridis_c(name = "Pearson\nCorrelation") +
  labs(
    x = "Dimension",
    y = "Dataset",
    title = "Correlation Heatmap by Dataset and Dimension"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 0))

ggsave("correlation_heatmap.png", heatmap_plot, width = 10, height = 8, dpi = 300)

# 4. Bonus: Advanced comparison plot
# Group by dataset type (e.g., BindingDB_IC50, BindingDB_Kd)
comparison_plot <- metrics_df %>%
  mutate(dataset_type = str_extract(dataset, "^[^_]+")) %>%
  ggplot(aes(x = dimension, y = pearson_cor, color = dataset)) +
  geom_line(aes(group = dataset), linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~dataset_type, scales = "free_y") +
  labs(
    x = "Dimension",
    y = "Pearson Correlation",
    title = "Correlation by Dataset Type and Dimension"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("comparison_by_dataset_type.png", comparison_plot, width = 12, height = 8, dpi = 300)

# Print a summary table of the metrics
summary_table <- metrics_df %>%
  arrange(desc(pearson_cor)) %>%
  select(dataset, dimension, pearson_cor, spearman_cor, mse, mae)

write.csv(summary_table, "embedding_metrics_summary.csv", row.names = FALSE)

# Print the table to console
print("Top performing embeddings by Pearson correlation:")
print(head(summary_table, 10))

print("Summary statistics by dimension:")
metrics_df %>%
  group_by(dimension) %>%
  summarize(
    mean_pearson = mean(pearson_cor),
    mean_spearman = mean(spearman_cor),
    mean_mse = mean(mse),
    mean_mae = mean(mae),
    n = n()
  ) %>%
  print()

print(main_plot)
