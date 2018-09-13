# Project_Kojak
Passion project from Metis Data Science Bootcamp

## Worklist
Here is my checklist of things to do:
* Engineer by meal nutritional statistics and implement in model
* Use final month as holdout set to test final models against
* Add R^2 into cross validation to check
* Prune model by replacing anomalies found in residuals plot?
* Retrain my model using the final regression chosen and run on holdout test set to get final R^2 and RMSE
* Determine forecasting vs prediction
* Engineer features (log and potentially lag back exogenous variables further)
* Start working on Yummly data to make recommendations based on calorie goals
* Integrate into flask app using tableau as well

## Forecasting Goals

* Predict tomorrow's weight based on today's weight and nutritional intake/activity
* Have feature to predict 1 week/1 month/X months in the future by inputting an expected calorie/macronutrient/activity goal (with confidence interval)
* Food recommender for the next meal/couple meals based on calorie/macronutrient goal and current caloric intake



* stacked models using prophet to feed into other regressors
* feed into gaussian NB with last ~3 weights of prophets prediction

* Gaussian Process regressors

For example, if trying to predict August 1st weight, take in normal exogenous variables plus prophets prediction for July 29, July 30, and July 31 weights.
