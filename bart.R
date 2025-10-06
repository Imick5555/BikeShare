library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(timetk)
library(dplyr)
library(glmnet)
library(dials)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(agua)
library(dbarts)

setwd("C://Users//Isaac//OneDrive//Documents//fall 2025 semester//STAT 348//BiekShare")

train <- vroom("train.csv")
test <- vroom("test.csv")

# h2o::h2o.init()

train_ud <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))



my_recipe <- recipe(count ~., data=train_ud) %>%
  step_mutate(weather= factor(ifelse(weather == "4", "3", weather),
                              levels=c("1","2","3"), 
                              labels=c("Clear", "Mist", "Light Snow"))) %>%
  step_mutate(season = factor(season, levels=c("1","2","3","4"), 
                              labels=c("Spring","Summer","Fall","Winter"))) %>%
  step_date(datetime, features = c("month", "year", "dow")) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train_ud)

bart_model <- parsnip::bart(trees=500) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("regression")

# auto_model <- auto_ml() %>%
# set_engine("h2o", max_runtime_secs=240, max_models=5) %>%
# set_mode("regression")
# 
bart_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model) %>%
  fit(data=train_ud)

# 
bart_grid <- grid_regular(trees(), levels = 5)
# 
folds <- vfold_cv(train_ud, v = 5, repeats=1)
#
CV_results <- bart_workflow %>%
  tune_grid(resamples=folds,
            grid=bart_grid,
            metrics=metric_set(rmse, mae))
# # 
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse")
# %>%
#   ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
#   geom_line()
# 
# bestTune <- CV_results %>%
#   select_best(metric="rmse")

# final_wf <-
#   bart_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=train_ud)
# 
bart_workflow %>%
  predict(new_data = test)

# final_wf %>%
#   predict(new_data = test)



## Run all the steps on test data15
bart_predictions <- predict(bart_workflow, new_data = test)



bart_predictions


kaggle_submission <- bart_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(count = exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

