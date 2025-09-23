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
  step_rm(datetime)%>%
  step_corr(all_numeric_predictors(), threshold=0.5) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train_ud)

my_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

pen_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)
# 
my_grid <- grid_regular(mtry(range=c(1, ncol(prepped_recipe) - 1)),
                     min_n(),
                     levels=5)
# 
folds <- vfold_cv(train_ud, v = 5, repeats=1)
# 
CV_results <- pen_workflow %>%
  tune_grid(resamples=folds,
            grid=my_grid,
            metrics=metric_set(rmse, mae))
# 
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()
# 
bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <-
  pen_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = test)



## Run all the steps on test data15
pen_predictions <- predict(final_wf, new_data = test)



pen_predictions


kaggle_submission <- pen_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(count = exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
