library(tidyverse)

mse <- read.csv("output/cv_mse.csv", header=TRUE)
mse %>% ggplot(aes(x = validation_fold, y = MSE)) +
    geom_point(aes(col = track_name)) +
    geom_text(aes(label=MSE),hjust=0.5, vjust=-0.5) +
    labs(x = "Iteration / Validation Fold Number in k-Fold Cross Validation", 
         y = "Mean Squared Error Value") + 
    ggtitle("Mean Square Error of Different Tracks in Each CV Iteration")
