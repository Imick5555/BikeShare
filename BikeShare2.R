library(tidyverse)
library(tidymodels)
library(patchwork)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(timetk)
library(dplyr)
library(glmnet)
library(dials)
library(rpart)
install.packages("rpart")
setwd("C://Users//Isaac//OneDrive//Documents//fall 2025 semester//STAT 348//BiekShare")

train <- vroom("train.csv")
test <- vroom("test.csv")

train_ud <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))



my_recipe <- recipe(count ~., data=train_ud) %>%
  step_mutate(weather= factor(ifelse(weather == "4", "3", weather),
                              levels=c("1","2","3"), 
                              labels=c("Clear", "Mist", "Light Snow"))) %>%
  step_mutate(season = factor(season, levels=c("1","2","3","4"), 
                              labels=c("Spring","Summer","Fall","Winter"))) %>%
  step_time(datetime, features="hour") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train_ud)

tree_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

head(train_ud, 5)


# penreg_model <- linear_reg(penalty=tune(),
#                            mixture=tune()) %>%
#   set_engine("glmnet")

tree_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod)
# 
tree_grid <- grid_regular(tree_depth(),
                               cost_complexity(),
                               min_n(),
                               levels = 5)
# 
folds <- vfold_cv(train_ud, v = 10, repeats=1)
# 
CV_results <- tree_workflow %>%
  tune_grid(resamples=folds,
            grid=tree_grid,
            metrics=metric_set(rmse, mae))
# 
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") 
# %>%
#   ggplot(data=., aes(x=tree_depth(), y=mean, color=factor(min_n()))) +
#   geom_line() +
#   facet_wrap(~ cost_complexity(), scales = "free_x")
# 
bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <-
  tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_ud)

final_wf %>%
  predict(new_data = test)



## Run all the steps on test data15
tree_predictions <- predict(final_wf, new_data = test)



tree_predictions


kaggle_submission <- tree_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(count = exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
