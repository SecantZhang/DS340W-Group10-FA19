library(tidyverse)

mse <- read.csv("output/cv_mse.csv", header=TRUE)
mse %>% ggplot(aes(x = validation_fold, y = MSE)) +
    geom_point(aes(col = track_name)) +
    geom_text(aes(label=MSE),hjust=0.5, vjust=-0.5) +
    labs(x = "Iteration / Validation Fold Number in k-Fold Cross Validation", 
         y = "Mean Squared Error Value") + 
    ggtitle("Mean Square Error of Different Tracks in Each CV Iteration")

mse_hist <- read.csv("output/mse_hist_marks_model.csv", header=TRUE)
mse_hist %>% ggplot(aes(x = Iteration, shape = ideas)) + 
    geom_point(aes(y = predicted, col = "Predicted")) + 
    geom_point(aes(y = avocado, col = "Avocado")) + 
    geom_point(aes(y = curr_impute, col = "Curr_impute")) + 
    facet_wrap(~Mark_id) + 
    labs(x = "Iteration / Validation Fold Number in k-Fold Cross Validation", 
         y = "Mean Squared Error Value")

mse_hist %>% select(-curr_impute) %>% 
    ggplot(aes(x = Iteration, shape = ideas)) + 
    geom_point(aes(y = predicted, col = "Predicted")) + 
    geom_point(aes(y = avocado, col = "Avocado")) + 
    facet_wrap(~Mark_id) + 
    labs(x = "Iteration / Validation Fold Number in k-Fold Cross Validation", 
         y = "Mean Squared Error Value")
