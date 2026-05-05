rm(list = ls())
gc()

library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(tidyr)
library(ggplot2)


# number of informative features for Friedman
N_INFORMATIVE = 7

# --------------------------------------------------
# Data loading


# Find all files containing "selection"
files_selection <- list.files(
    path = "./results",
    pattern = "selection.*\\.csv$",
    full.names = TRUE
)

# Read and combine
selection_all <- files_selection %>%
    map(~ read_csv(
        .x,
        col_types = cols(
            num_features = col_number(),
            num_selected = col_number(),
            sampling = col_factor(levels = c("none", "bootstrap")),
            result_type = col_factor(levels = c("weights", "rank", "subset"))
        )
    )) %>%
    bind_rows()


# Inspect result
print(selection_all)

# Find all files containing "scoring"
files_scoring <- list.files(
    path = "./results",
    pattern = "scoring-complete.*\\.csv$",
    full.names = TRUE
)

# Read and combine
scoring_all <- files_scoring %>%
    map(~ read_csv(.x,
                   col_types = cols(sampling = col_factor(levels = c("none",
                                                                     "bootstrap"))))) %>%
    bind_rows()

# Inspect result
print(scoring_all)

#------------------------------------------------------------------
# Plot pecentage of informative feature in top k features


model_data <- selection_all %>%
    filter(result_type %in% c("weights", "rank"))

parse_values <- function(s) {
    s %>%
        str_remove_all("\\[|\\]") %>%
        str_split(",") %>%
        `[[`(1) %>%
        str_trim() %>%
        as.numeric()
}
k_values <- c(8, 16, 32, 64, 128, 256)

score_one_row <- function(values_str, true_k, result_type) {
    vals <- parse_values(values_str)
    n    <- length(vals)

    if (result_type == "weights") {
        # Higher absolute weight = more important
        ranked_indices <- order(abs(vals), decreasing = TRUE)

    } else if (result_type == "rank") {
        # Lower rank value = more important (rank 1 is best)
        ranked_indices <- order(vals, decreasing = FALSE)
    }

    sapply(k_values, function(k) {
        k_eff     <- min(k, n)
        top_k_idx <- ranked_indices[seq_len(k_eff)]
        sum(top_k_idx <= true_k)
    })
}

score_cols <- paste0("score_k", k_values)

scores_matrix <- mapply(
    score_one_row,
    model_data$values,
    N_INFORMATIVE,
    as.character(model_data$result_type)
)

# scores_matrix: 4 rows × n_models columns
results <- model_data %>%
    select(name, dataset_name, num_features, result_type) %>%
    bind_cols(
        as.data.frame(t(scores_matrix)) %>%
            setNames(score_cols)
    )

results_long <- results %>%
    pivot_longer(
        cols      = starts_with("score_k"),
        names_to  = "k_label",
        values_to = "score"
    ) %>%
    mutate(k = as.integer(str_extract(k_label, "\\d+"))) %>%
    select(-k_label)

print(results)
print(results_long)


#----------------------------------------------------------------------
# percentage of correctly identified features
results_long <- results_long %>%
    mutate(pcif = score / N_INFORMATIVE * 100)
results_long <- results_long %>%
    mutate(k = factor(k, levels = c(0, 8, 16, 32, 64, 128, 256)))

ggplot(results_long, aes(x = k, y = pcif, color = name, group = name)) +
    stat_summary(fun = mean, geom = "line", linewidth = 1) +
    stat_summary(fun = mean, geom = "point", size = 2) +
    labs(
        title = "Feature Selection Performance",
        x = "Top-k selected features",
        y = "Percentage of correctly selected informative features",
        color = "Method"
    ) +
    theme_minimal() +
    coord_cartesian(ylim = c(0, 100))
