# Loading requisite packages
library(ggplot2)
library(glmnet)
library(dplyr)
library(reshape2)
library(car)
library(MASS)
library(olsrr)
library(caret)
library(lmtest)
library(FNN)
library(xgboost)
library(Metrics)
library(randomForest)
library(e1071)


#################################### DATASET CONFIGURATION ####################################

# Reading in the statcast_final dataset
statcast = read.csv("/Users/Jake/Desktop/STA 221/Project Data/statcast_final.csv", header = TRUE)

# Reordering columns in the statcast dataset
statcast = statcast[, c(names(statcast)[1:100], names(statcast)[102], names(statcast)[101])]

# Removing any variables that contain only missing values for pre-2024 seasons
statcast = statcast[, -c(3, 41, 52:59, 92:96, 99)]

# Standardizing all covariates
statcast_std = data.frame(cbind(statcast[, 1], statcast[, 2], scale(statcast[, 2:84]), statcast[, 85], statcast[, 86]))

# Renaming the Name, Injury, and next.war variables
names(statcast_std)[1] = "Name"
names(statcast_std)[2] = "Year"
names(statcast_std)[3] = "Year.std"
names(statcast_std)[86] = "inj"
names(statcast_std)[87] = "next.war"

# Splitting the standardized dataset into the main data and 2024 data
main = statcast_std[is.na(statcast_std$next.war) == FALSE, ]
dat24 = statcast_std[statcast_std$Year == 2024, ]

# Converting all variables in both datasets to numeric
main = main %>%
  mutate(across(where(is.character) & -Name, as.numeric))
dat24 = dat24 %>%
  mutate(across(where(is.character) & -Name, as.numeric))

# Separating the name and year labels from the data in the main dataset
labels = main[, c(1, 2)]
data = main[, -c(1, 2)]


#################################### EXPLORATORY DATA ANALYSIS ####################################

# plotting histogram of the response variable
hist(data$next.war, main = "", xlab = "next.war", col = "steelblue")

# Plotting a handful of pairwise scatterplots for some covariates vs. next.war
plot(data$next.war ~ data$WAR, xlab = "WAR", ylab = "next.war")
plot(data$next.war ~ data$woba, xlab = "woba", ylab = "next.war")
plot(data$next.war ~ data$player_age, xlab = "player_age", ylab = "next.war")
plot(data$next.war ~ data$sprint_speed, xlab = "sprint_speed", ylab = "next.war")

# Plotting heat map of correlation matrix for the data
cor_matrix = cor(data, use = "complete.obs")
heatmap_colors = rev(heat.colors(100))
heatmap(cor_matrix, main = "Heatmap of Pairwise Correlations", symm = TRUE,
        Rowv = NA, Colv = NA, margins = c(7, 4))

# Create dummy matrix for color scale
z = matrix(seq(-1, 1, length.out = 100), ncol = 1)

# Plot the color scale
par(fig = c(0.8, 0.9, 0.04, 1), new = TRUE)  # Adjust figure region
image(1, seq(-1, 1, length.out = 100), t(z),  # Use t(z) to ensure dimensions align
      col = heatmap_colors, axes = FALSE, main = "")
axis(4, at = seq(-1, 1, by = 0.2), labels = seq(-1, 1, by = 0.2), las = 1)
mtext("Correlation", side = 4, line = 2)


#################################### CREATING CANDIDATE OLS MODELS ####################################

# Using the full main dataset to create intercept, full additive models
# Full additive model has 85 terms, R-Squared Adj = 0.4353
n = length(main$Year.std)
intercept = lm(next.war ~ 1, data = data)
add_full = lm(next.war ~ ., data = data)
summary(add_full)
length(add_full$coefficients)

# Forward stepwise first-order model by AIC.
# 25 model terms, R-Squared Adj = 0.4389, many high VIF values.
forward1_AIC = stepAIC(intercept, scope=list(upper=add_full, lower = ~1),
                       direction="both", k=2, trace = FALSE)
summary(forward1_AIC)
length(forward1_AIC$coefficients)
vif(forward1_AIC)

# Full two-way interaction model of the forward1_AIC model
# 301 terms, R-Squared Adj = 0.4616, many high VIF values.
fw_dat = data[, c("WAR", "player_age", "avg.war", "b_ball", "bacon", "sprint_speed", "b_intent_walk",
                  "b_game", "f_strike_percent", "oz_contact_percent", "Year.std", "edge_percent", "woba",
                  "strikeout", "xiso", "xwobacon", "b_lob", "b_foul", "oz_swing_percent", "avg_hyper_speed",
                  "single", "babip", "b_total_bases", "flyballs_percent", "next.war")]
int_full = lm(next.war ~ .*., data = fw_dat)
summary(int_full)
length(int_full$coefficients)
vif(int_full)

# Testing the additive forward AIC model vs. the two-way interaction model; small p-value, we will include some interactions
anova(forward1_AIC, int_full)

# Forward stepwise regression from the additive model to the full interaction model
# 65 terms, R-Squared Adj = 0.4753, many high VIF values
forward2_AIC = stepAIC(forward1_AIC, scope=list(upper=int_full, lower = forward1_AIC),
                       direction="both", k=2, trace = FALSE)
summary(forward2_AIC)
length(forward2_AIC$coefficients)
vif(forward2_AIC, type = 'predictor')

# Creating simple linear regression models using war and war + avg.war + player_age as the regressors, to compare our full models to
war_mod = lm(next.war ~ WAR, data = data)
war_mod2 = lm(next.war ~ WAR + avg.war + player_age, data = data)

# Creating a test model that only contains highly significant additive and interaction terms
final_mod = lm(next.war ~ WAR + player_age + avg.war + b_ball + strikeout + xiso + woba + sprint_speed +
                f_strike_percent + edge_percent + Year.std + avg_hyper_speed + b_game + b_foul +
                double + b_intent_walk + WAR:f_strike_percent + avg.war:woba, data = data)
summary(final_mod)
length(final_mod$coefficients)
vif(final_mod, type = 'predictor')


#################################### CROSS-VALIDATING CANDIDATE OLS MODELS ####################################

# Creating a list of candidate OLS models
models = list(intercept, war_mod, war_mod2, forward1_AIC, int_full, forward2_AIC, final_mod)

# Creating empty dataframe to store the RMSE values for each model
rmse = data.frame(matrix(ncol = length(models) + 1, nrow = 6))
names(rmse) = c("test_year", "intercept", "war", "war2", "forward1_AIC", "int_full", "forward2_AIC", "final_mod")
rmse$test_year = c(2016, 2017, 2018, 2021, 2022, 2023)

# Iterating over each testing year in the main dataset (2016-2023)
# completing sequential cross-validation for each subsequent year, storing the RMSE in each year
options(warn = -1) 
for (y in c(2016, 2017, 2018, 2021, 2022, 2023)){
  
  # Creating the training and test sets for the given year
  train = main[main$Year < y, -c(1, 2)]
  test = main[main$Year == y, -c(1, 2)]
  
  # Training each model using the training set and making predictions for the following year
  intercept_train = lm(formula(intercept), data = train)
  intercept_predictions = predict(intercept_train, newdata = test[, -c(87)])
  war_train = lm(formula(war_mod), data = train)
  war_predictions = predict(war_train, newdata = test[, -c(87)])
  war2_train = lm(formula(war_mod2), data = train)
  war2_predictions = predict(war2_train, newdata = test[, -c(87)])
  fw1_train = lm(formula(forward1_AIC), data = train)
  fw1_predictions = predict(fw1_train, newdata = test[, -c(87)])
  int_train = lm(formula(int_full), data = train)
  int_predictions = predict(int_train, newdata = test[, -c(87)])
  fw2_train = lm(formula(forward2_AIC), data = train)
  fw2_predictions = predict(fw2_train, newdata = test[, -c(87)])
  final_train = lm(formula(final_mod), data = train)
  final_predictions = predict(final_train, newdata = test[, -c(87)])
  
  # Storing the actual observed next.war values from the test set
  actual = test$next.war
  
  # calculating the RMSE for each model in that given year
  rmse_intercept = sqrt(mean((intercept_predictions - actual)^2))
  rmse_war = sqrt(mean((war_predictions - actual)^2))
  rmse_war2 = sqrt(mean((war2_predictions - actual)^2))
  rmse_fw1 = sqrt(mean((fw1_predictions - actual)^2))
  rmse_int = sqrt(mean((int_predictions - actual)^2))
  rmse_fw2 = sqrt(mean((fw2_predictions - actual)^2))
  rmse_final = sqrt(mean((final_predictions - actual)^2))
  
  # Storing the RMSE for each model in the rmse table
  rmse$intercept[rmse$test_year == y] = rmse_intercept
  rmse$war[rmse$test_year == y] = rmse_war
  rmse$war2[rmse$test_year == y] = rmse_war2
  rmse$forward1_AIC[rmse$test_year == y] = rmse_fw1
  rmse$int_full[rmse$test_year == y] = rmse_int
  rmse$forward2_AIC[rmse$test_year == y] = rmse_fw2
  rmse$final_mod[rmse$test_year == y] = rmse_final
}

# Choosing the model with the lowest mean RMSE across all test splits; we will use the final_mod
sapply(rmse, mean)


#################################### ASSESSING FINAL OLS MODEL ####################################

# Residual plots. No major departures
par(mfrow = c(2, 2), mar = c(2.5, 2.5, 3, 1))
plot(final_mod, which = 1)
plot(final_mod, which = 2)
plot(final_mod, which = 4)
hist(final_mod$residuals, main = "Histogram of OLS Residuals")

# Running Durbin-Watson test for Lag-1 Autocorrelation; high P-Value = No Autocorrelation detected
durbinWatsonTest(final_mod)

# Printing VIF values for the model, moderately high multicollinearity for terms associated with WAR, avg.war, b_ball, woba, and f_strike_percent
vif(final_mod, type = 'predictor')


#################################### FINDING OPTIMAL LASSO MODEL ON FULL DATASET ####################################

# Defining list of lambda values that we want to assess the Lasso model for
lambda_vals = seq(0.01, 1, 0.0001)

# Creating a dataframe to store results
results = data.frame(lambda = lambda_vals, RMSE = NA)

# Creating Lasso dataframe using full dataset
lasso_data = data

# Ensuring data is ordered by Year
lasso_data = lasso_data[order(lasso_data$Year.std), ]

# Iterating over each lambda value
for (l in 1:length(lambda_vals)){
  
  # Setting lambda for this iteration
  lambda = lambda_vals[l]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this lambda
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(lasso_data$Year.std))){
    
    # Defining training and test sets based on the current year
    train_data = lasso_data[lasso_data$Year.std < unique(lasso_data$Year.std)[year], ]
    test_data = lasso_data[lasso_data$Year.std == unique(lasso_data$Year.std)[year], ]
    
    # Extracting predictors and response for training and test sets
    X_train = as.matrix(train_data[, -length(names(train_data))])
    y_train = train_data$next.war
    X_test = as.matrix(test_data[, -length(names(train_data))])
    y_test = test_data$next.war
    
    # Fitting Lasso model with current lambda
    lasso_model = glmnet(X_train, y_train, alpha = 1, lambda = lambda)
    
    # Predicting on the test set
    predictions = predict(lasso_model, X_test)
    
    # Calculating RMSE for the test set
    rmse_vals = c(rmse_vals, sqrt(mean((predictions - y_test)^2)))
  }
  
  # Storing the average RMSE for this lambda
  results$RMSE[l] = mean(rmse_vals)
}

# Finding the optimal lambda (lambda = 0.0209)
results[results$RMSE == min(results$RMSE), ]
lambda_opt1 = results$lambda[results$RMSE == min(results$RMSE)]

# Fitting this optimal additive Lasso model to the full dataset
X_full = as.matrix(lasso_data[, -length(names(lasso_data))])
y_full = lasso_data$next.war

# Fitting the Lasso model to the full dataset with the optimal lambda
lasso_model_full1 = glmnet(X_full, y_full, alpha = 1, lambda = lambda_opt1)

# Storing the coefficients for this model
lasso_coef1 = as.matrix(coef(lasso_model_full1))


#################################### FINDING OPTIMAL XGBOOST MODEL ON FULL DATASET ####################################

# Defining the xgboost dataset, ordering by Year
data_xgb = data
data_xgb = data_xgb[order(data_xgb$Year.std), ]

# Creating a dataframe with the mean RMSE for each set of tuning parameters
param_grid = expand.grid(
  max_depth = seq(1, 7, 1),
  eta = seq(0.01, 0.1, 0.01),
  subsample = seq(0.5, 1, 0.1),
  colsample_bytree = seq(0.5, 1, 0.1)
)
param_grid$RMSE = NA

# Iterating over all rows in the parameters dataset
for (r in 1:length(param_grid$max_depth)){
  
  # Setting the parameters for this iteration
  md_r = param_grid$max_depth[r]
  eta_r = param_grid$eta[r]
  subsample_r = param_grid$subsample[r]
  cs_r = param_grid$colsample_bytree[r]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this set of parameters
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_xgb$Year.std))){
    
    # Defining the training and testing datasets
    train_data = data_xgb[data_xgb$Year.std < unique(data_xgb$Year.std)[year], ]
    test_data = data_xgb[data_xgb$Year.std == unique(data_xgb$Year.std)[year], ]
    
    # Extracting features (X) and target variable (y)
    X_train = as.matrix(train_data[, -length(names(train_data))])
    y_train = train_data$next.war
    X_test = as.matrix(test_data[, -length(names(train_data))])
    y_test = test_data$next.war
    
    # Preparing the data in DMatrix format for XGBoost
    dtrain = xgb.DMatrix(data = X_train, label = y_train)
    dtest = xgb.DMatrix(data = X_test, label = y_test)
    
    # Setting parameters
    params = list(
      objective = "reg:squarederror",
      max_depth = md_r,
      eta = eta_r,
      subsample = subsample_r,
      colsample_bytree = cs_r)
    
    # Training the model
    nrounds = 100
    xgb_model = xgboost(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      verbose = 0)
    
    # Predicting on test data
    y_pred = predict(xgb_model, dtest)
    
    # Storing RMSE for this year in the rmse vector
    rmse = sqrt(mean((y_test - y_pred)^2))
    rmse_vals = c(rmse_vals, rmse)
  }
  
  # Storing the average RMSE for this set of parameters
  param_grid$RMSE[r] = mean(rmse_vals)
}

# Printing the combination of tuning parameters that correspond to the best RMSE model
param_grid[param_grid$RMSE == min(param_grid$RMSE), ]

# Setting parameters to the optimal values
md_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][1])
eta_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][2])
ss_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][3])
csbt_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][4])

# Fitting the XGBoost model to the full dataset
params = list(
  objective = "reg:squarederror",
  max_depth = md_opt,
  eta = eta_opt,
  subsample = ss_opt,
  colsample_bytree = csbt_opt)
xgb_data = xgb.DMatrix(data = as.matrix(data[, -length(names(data))]), label = data$next.war)
nrounds = 100
xgb_model = xgboost(
  params = params,
  data = xgb_data,
  nrounds = nrounds,
  verbose = 0)

# Extracting feature importance from this XGBoost model
importance = xgb.importance(model = xgb_model)
importance$Gain = round(importance$Gain, 4)

# Creating a subset based features with Gain > 0.0055
xgboost_terms = importance$Feature[importance$Gain > 0.0055]

# Plotting XGBoost Gain values
par(mfrow = c(1, 1), mar = c(10, 5, 2, 2))
importance = importance[order(-importance$Gain), ]
features = importance$Feature
values = importance$Gain
colors = ifelse(importance$Gain > 0.0055, "indianred2", "steelblue")
barplot(
  height = values,
  names.arg = features,
  las = 2,
  col = colors,
  main = "XGBoost Feature Importance by Gain",
  xlab = "",
  ylab = "Gain",
  cex.names = 0.8
)
legend(50, 0.2,
       legend = c("Included in Subset (Gain > 0.0055)", "Not Included in Subsset (Gain <= 0.0055)"),
       fill = c("indianred2", "steelblue")
)


#################################### FINDING OPTIMAL RANDOM FOREST MODEL ON FULL DATASET ####################################

# Defining the rf dataset, ordering by year
data_rf = data
data_rf = data_rf[order(data_rf$Year.std), ]

# Creating a dataframe with the mean RMSE for each set of tuning parameters
param_grid = expand.grid(
  ntree = c(300, 500, 700, 1000),
  mtry = c(2, 5, 9, 12)
)
param_grid$RMSE = NA

# Iterating over all rows in the parameters dataset
for (r in 1:length(param_grid$ntree)){
  
  # Setting the parameters for this iteration
  ntree_r = param_grid$ntree[r]
  mtry_r = param_grid$mtry[r]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this set of parameters
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_rf$Year.std))){
    
    # Defining the training and testing datasets
    train_data = data_rf[data_rf$Year.std < unique(data_rf$Year.std)[year], ]
    test_data = data_rf[data_rf$Year.std == unique(data_rf$Year.std)[year], ]
    
    # Training random forest model
    rf_model = randomForest(next.war ~ ., data = train_data, 
                            ntree = ntree_r,
                            mtry = mtry_r,
                            importance = FALSE)
    
    # Making predictions
    predictions = predict(rf_model, newdata = test_data)
    
    # Calculating RMSE, appending to vector
    rmse = sqrt(mean((predictions - test_data$next.war)^2))
    rmse_vals = c(rmse_vals, rmse)
  }
  
  # Storing the average RMSE for this set of parameters
  param_grid$RMSE[r] = mean(rmse_vals)
}

# Printing the combination of tuning parameters that correspond to the best RMSE model
param_grid[param_grid$RMSE == min(param_grid$RMSE), ]

# Setting parameters to the optimal values
nt_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][1])
mt_opt = as.numeric(param_grid[param_grid$RMSE == min(param_grid$RMSE), ][2])

# Fitting random forest model to full dataset
rf_model = randomForest(next.war ~ ., data = data, 
                        ntree = nt_opt,
                        mtry = mt_opt)

# Extracting feature importance
importance_values = as.data.frame(importance(rf_model))
importance_values$Feature = rownames(importance_values)

# Creating a subset of features where their node purity is at least 100
rf_terms = importance_values$Feature[importance_values$IncNodePurity > 100]

# Plotting IncNodePurity values
par(mfrow = c(1, 1), mar = c(10, 5, 2, 2))
importance_values = importance_values[order(-importance_values$IncNodePurity), ]
features = importance_values$Feature
values = importance_values$IncNodePurity
colors = ifelse(importance_values$IncNodePurity > 100, "indianred2", "steelblue")
barplot(
  height = values,
  names.arg = features,
  las = 2,
  col = colors,
  main = "Random Forest Feature Importance by IncNodePurity",
  xlab = "",
  ylab = "IncNodePurity",
  cex.names = 0.8
)
legend(50, 500,
  legend = c("Included in Subset (IncNodePurity > 100)", "Not Included in Subsset (IncNodePurity <= 100)"),
  fill = c("indianred2", "steelblue")
)


#################################### FINDING OPTIMAL RIDGE MODEL ON LASSO SUBSET ####################################

# Creating a dataset to be used for Ridge regression using only the subset variables specified by the Lasso model
vars = rownames(lasso_coef1)[lasso_coef1 != 0]
data_ridge = data[, c(vars[2:length(vars)], "next.war")]

# Defining list of lambda values that we want to assess the Ridge model for
lambda_vals = seq(0.0001, 1, 0.0001)

# Creating a dataframe to store results
results = data.frame(lambda = lambda_vals, RMSE = NA)

# Ensuring data is ordered by Year
data_ridge = data_ridge[order(data_ridge$Year.std), ]

# Iterating over each lambda value
for (l in 1:length(lambda_vals)){
  
  # Setting lambda for this iteration
  lambda = lambda_vals[l]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this lambda
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_ridge$Year.std))){
    
    # Defining training and test sets based on the current year
    train_data_ridge = data_ridge[data_ridge$Year.std < unique(data_ridge$Year.std)[year], ]
    test_data_ridge = data_ridge[data_ridge$Year.std == unique(data_ridge$Year.std)[year], ]
    
    # Extracting predictors and response for training and test sets
    X_train = as.matrix(train_data_ridge[, -length(names(train_data_ridge))])
    y_train = train_data_ridge$next.war
    X_test = as.matrix(test_data_ridge[, -length(names(train_data_ridge))])
    y_test = test_data_ridge$next.war
    
    # Fitting Ridge model with current lambda
    ridge_model = glmnet(X_train, y_train, alpha = 0, lambda = lambda)
    
    # Predicting on the test set
    predictions = predict(ridge_model, X_test)
    
    # Calculating RMSE for the test set
    rmse_vals = c(rmse_vals, sqrt(mean((predictions - y_test)^2)))
  }
  
  # Storing the average RMSE for this lambda
  results$RMSE[l] = mean(rmse_vals)
}

# Finding the optimal lambda (lambda = 0.1529)
results[results$RMSE == min(results$RMSE), ]


#################################### FINDING OPTIMAL RIDGE MODEL ON XGBOOST SUBSET ####################################

# Creating a dataset to be used for Ridge regression using only the subset variables specified by final XGBoost model
data_ridge = data[, c("Year.std", xgboost_terms, "next.war")]

# Defining list of lambda values that we want to assess the Ridge model for
lambda_vals = seq(0.0001, 1, 0.0001)

# Creating a dataframe to store results
results = data.frame(lambda = lambda_vals, RMSE = NA)

# Ensuring data is ordered by Year
data_ridge = data_ridge[order(data_ridge$Year.std), ]

# Iterating over each lambda value
for (l in 1:length(lambda_vals)){
  
  # Setting lambda for this iteration
  lambda = lambda_vals[l]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this lambda
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_ridge$Year.std))){
    
    # Defining training and test sets based on the current year
    train_data_ridge = data_ridge[data_ridge$Year.std < unique(data_ridge$Year.std)[year], -1]
    test_data_ridge = data_ridge[data_ridge$Year.std == unique(data_ridge$Year.std)[year], -1]
    
    # Extracting predictors and response for training and test sets
    X_train = as.matrix(train_data_ridge[, -length(names(train_data_ridge))])
    y_train = train_data_ridge$next.war
    X_test = as.matrix(test_data_ridge[, -length(names(train_data_ridge))])
    y_test = test_data_ridge$next.war
    
    # Fitting Ridge model with current lambda
    ridge_model = glmnet(X_train, y_train, alpha = 0, lambda = lambda)
    
    # Predicting on the test set
    predictions = predict(ridge_model, X_test)
    
    # Calculating RMSE for the test set
    rmse_vals = c(rmse_vals, sqrt(mean((predictions - y_test)^2)))
  }
  
  # Storing the average RMSE for this lambda
  results$RMSE[l] = mean(rmse_vals)
}

# Finding the optimal lambda (lambda = 0.1789)
results[results$RMSE == min(results$RMSE), ]


#################################### FINDING OPTIMAL RIDGE MODEL ON RANDOM FOREST SUBSET ####################################

# Creating a dataset to be used for Ridge regression using only the subset variables specified by final Random Forest model
data_ridge = data[, c("Year.std", rf_terms, "next.war")]

# Defining list of lambda values that we want to assess the Ridge model for
lambda_vals = seq(0.0001, 1, 0.0001)

# Creating a dataframe to store results
results = data.frame(lambda = lambda_vals, RMSE = NA)

# Ensuring data is ordered by Year
data_ridge = data_ridge[order(data_ridge$Year.std), ]

# Iterating over each lambda value
for (l in 1:length(lambda_vals)){
  
  # Setting lambda for this iteration
  lambda = lambda_vals[l]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this lambda
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_ridge$Year.std))){
    
    # Defining training and test sets based on the current year
    train_data_ridge = data_ridge[data_ridge$Year.std < unique(data_ridge$Year.std)[year], -1]
    test_data_ridge = data_ridge[data_ridge$Year.std == unique(data_ridge$Year.std)[year], -1]
    
    # Extracting predictors and response for training and test sets
    X_train = as.matrix(train_data_ridge[, -length(names(train_data_ridge))])
    y_train = train_data_ridge$next.war
    X_test = as.matrix(test_data_ridge[, -length(names(train_data_ridge))])
    y_test = test_data_ridge$next.war
    
    # Fitting Ridge model with current lambda
    ridge_model = glmnet(X_train, y_train, alpha = 0, lambda = lambda)
    
    # Predicting on the test set
    predictions = predict(ridge_model, X_test)
    
    # Calculating RMSE for the test set
    rmse_vals = c(rmse_vals, sqrt(mean((predictions - y_test)^2)))
  }
  
  # Storing the average RMSE for this lambda
  results$RMSE[l] = mean(rmse_vals)
}

# Finding the optimal lambda (lambda = 0.0305)
results[results$RMSE == min(results$RMSE), ]


#################################### FINDING OPTIMAL SVM MODEL ON RFE FEATURE-SELECTED SUBSET ####################################

# Defining control parameters for RFE
control = rfeControl(functions = caretFuncs, method = "cv", number = 10)

# Performing RFE with radial SVM
set.seed(123)
svm_profile = rfe(
  x = data[, -85], 
  y = data$next.war,
  sizes = c(1:5),  # Number of features to select
  rfeControl = control,
  method = "svmRadial"
)

# Extracting 36 most important features
ranked_features = svm_profile$optVariables[1:36]

# Creating subset data consisting only of these 36 features, Year.std, and next.war
data_sub = data[, c("Year.std", ranked_features, "next.war")]

# Removing aliased variables
data_sub = data_sub[, -c(22, 23, 30)]

# Correlation heat map of covariates
cor_matrix = cor(data_sub, use = "complete.obs")
heatmap(cor_matrix, main = "Heatmap of Pairwise Correlations", symm = TRUE,
        Rowv = NA, Colv = NA, margins = c(7, 4))

# Paring down to most predictive, low-multicollinearity variables
data_sub2 = data_sub[, c("Year.std", "WAR", "avg.war", "r_run", "b_ball", "double", "b_foul", "b_game",
                        "woba", "b_lob", "b_intent_walk", "xiso", "avg_hyper_speed", "player_age",
                        "strikeout", "r_total_stolen_base")]

# Adding pairwise interactions and next.war back into the dataframe
data_sub2 = as.data.frame(model.matrix(~ .^2 - 1, data = data_sub2))
data_sub2$next.war = data_sub$next.war

# Removing any interaction terms involving Year.std since we do not plan to include it in the SVM model
data_sub2 = data_sub2[, -c(17:31)]

# Reordering columns so that chosen interactions are the first interaction terms
data_sub2 = data_sub2[, c(1:16, 30, 104, 88, 40, 87, 17:29, 31:39, 41:86, 89:103, 105:122)]

## ITERATION PROCESS TO DETERMINE WHICH INTERACTION TERM SHOULD BE INCLUDED IN SVM MODEL NEXT ##
# Creating empty dataframe to store results
results = data.frame(Variable = rep(NA, 101), Cost = rep(NA, 101), Gamma = rep(NA, 101), RMSE = rep(NA, 101))

# Iterating over the remaining interaction terms to determine which term lowers the mean RMSE of the SVM model the most
for (v in 21:121){
  
  # Defining svm data, ordering by year
  data_svm = data_sub2[, c(1:20, v, 122)]
  data_svm = data_svm[order(data_svm$Year.std), ]
  
  # Creating a dataframe with the mean RMSE for each set of tuning parameters
  param_grid = expand.grid(
    kernel = c("radial"),
    cost = seq(1, 10, 1),
    gamma = seq(0.001, 0.01, 0.001)
  )
  param_grid$RMSE = NA
  
  # Iterating over all rows in the parameters dataset
  for (r in 1:length(param_grid$kernel)){
    
    # Setting the parameters for this iteration
    kernel_r = param_grid$kernel[r]
    cost_r = param_grid$cost[r]
    gamma_r = param_grid$gamma[r]
    
    # Creating empty vector to store RMSE values for each Train/Test iteration for this set of parameters
    rmse_vals = c()
    
    # Training/Testing over each year in the dataset
    for (year in 2:length(unique(data_svm$Year.std))){
      
      # Defining the training and testing datasets
      train_data = data_svm[data_svm$Year.std < unique(data_svm$Year.std)[year], -1]
      test_data = data_svm[data_svm$Year.std == unique(data_svm$Year.std)[year], -1]
      
      # Training svm model
      svm_model = svm(next.war ~ ., data = train_data, type = "eps-regression",
                      kernel = kernel_r, cost = cost_r, gamma = gamma_r)
      
      # Making predictions
      predictions = predict(svm_model, newdata = test_data)
      
      # Calculating RMSE, appending to vector
      rmse = sqrt(mean((predictions - test_data$next.war)^2))
      rmse_vals = c(rmse_vals, rmse)
    }
    
    # Storing the average RMSE for this set of parameters
    param_grid$RMSE[r] = mean(rmse_vals)
  }
  
  # Storing the additional variable name, the paramter values, and the mean RMSE
  results$Variable[v - 20] = names(data_sub2)[v]
  results$Cost[v - 20] = param_grid$cost[param_grid$RMSE == min(param_grid$RMSE)]
  results$Gamma[v - 20] = param_grid$gamma[param_grid$RMSE == min(param_grid$RMSE)]
  results$RMSE[v - 20] = param_grid$RMSE[param_grid$RMSE == min(param_grid$RMSE)]
}

# Defining the final svm subset as the RFE feature selected subset, ordering by year
data_svm = data_sub2[, c(1:21, 122)]
data_svm = data_svm[order(data_svm$Year.std), ]

# Creating a dataframe with the mean RMSE for each set of tuning parameters
param_grid = expand.grid(
  kernel = c("radial", "linear", "polynomial", "sigmoid"),
  cost = seq(0, 12, 0.01),
  gamma = seq(0.001, 0.01, 0.0001)
)
param_grid$RMSE = NA

# Iterating over all rows in the parameters dataset
for (r in 1:length(param_grid$kernel)){
  
  # Setting the parameters for this iteration
  kernel_r = param_grid$kernel[r]
  cost_r = param_grid$cost[r]
  gamma_r = param_grid$gamma[r]
  
  # Creating empty vector to store RMSE values for each Train/Test iteration for this set of parameters
  rmse_vals = c()
  
  # Training/Testing over each year in the dataset
  for (year in 2:length(unique(data_svm$Year.std))){
    
    # Defining the training and testing datasets
    train_data = data_svm[data_svm$Year.std < unique(data_svm$Year.std)[year], -1]
    test_data = data_svm[data_svm$Year.std == unique(data_svm$Year.std)[year], -1]
    
    # Training svm model
    svm_model = svm(next.war ~ ., data = train_data, type = "eps-regression",
                    kernel = kernel_r, cost = cost_r, gamma = gamma_r)
    
    # Making predictions
    predictions = predict(svm_model, newdata = test_data)
    
    # Calculating RMSE, appending to vector
    rmse = sqrt(mean((predictions - test_data$next.war)^2))
    rmse_vals = c(rmse_vals, rmse)
  }
  
  # Storing the average RMSE for this set of parameters
  param_grid$RMSE[r] = mean(rmse_vals)
}

# Printing the combination of tuning parameters that correspond to the best RMSE model
param_grid[param_grid$RMSE == min(param_grid$RMSE), ]

# Creating a linear model of the same covariates for comparison and to assess vif values
test_final = lm(next.war ~ WAR + avg.war + r_run + b_ball + double + b_foul +
                  b_game + woba + b_lob + b_intent_walk + xiso + avg_hyper_speed + player_age +
                  strikeout + r_total_stolen_base + WAR:r_total_stolen_base +
                  b_lob:player_age + b_game:b_intent_walk + avg.war:avg_hyper_speed +
                  b_game:b_lob, data = data)
vif(test_final, type = 'predictor')


#################################### FINAL MODEL PERFORMANCE COMPARISON ####################################

# Setting the final optimal parameter values
kernel_final = as.character(param_grid$kernel[param_grid$RMSE == min(param_grid$RMSE)])
cost_final = param_grid$cost[param_grid$RMSE == min(param_grid$RMSE)]
gamma_final = param_grid$gamma[param_grid$RMSE == min(param_grid$RMSE)]

# Adding interaction terms to the final svm dataset
data_svm_final = main[, c(names(data_svm)[1:16], "next.war")]
data_svm_final$WAR.r_total_stolen_base = data_svm_final$WAR * data_svm_final$r_total_stolen_base
data_svm_final$b_lob.player_age = data_svm_final$b_lob * data_svm_final$player_age
data_svm_final$b_game.b_intent_walk = data_svm_final$b_game * data_svm_final$b_intent_walk
data_svm_final$avg.war.avg_hyper_speed = data_svm_final$avg.war * data_svm_final$avg_hyper_speed
data_svm_final$b_game.b_lob = data_svm_final$b_game * data_svm_final$b_lob
data_svm_final = data_svm_final[, c(1:16, 18:22, 17)]

# Fitting the final SVM model to the full dataset
svm_mod_final = svm(next.war ~ ., data = data_svm_final, type = "eps-regression",
                      kernel = kernel_final, cost = cost_final, gamma = gamma_final)

# Using the final SVM model to predict WAR values
svm_predictions_final = svm_mod_final$fitted
residuals_svm = svm_mod_final$residuals
rmse_svm = sqrt(mean((residuals_svm)^2))

# Durbin-Watson test for Lag-1 Autocorrelation in SVR residuals
durbinWatsonTest(residuals_svm)

# calculating the rmse for the OLS model
ols_final = lm(next.war ~ WAR + player_age + avg.war + b_ball + strikeout + xiso + woba + sprint_speed +
                 f_strike_percent + edge_percent + Year.std + avg_hyper_speed + b_game + b_foul +
                 double + b_intent_walk + WAR:f_strike_percent + avg.war:woba, data = data)
rmse_ols = sqrt(mean((ols_final$residuals)^2))

# Comparing OLS vs. SVM residuals
par(mfrow = c(2,2), mar = c(2.5, 2.5, 3, 1))

# Plotting model residual distributions (Intercept)
hist(intercept$residuals, 
     col="lightblue3",
     xlim=c(min(c(war_mod$residuals, residuals_svm)), 6),
     main="Intercept OLS vs. SVR", 
     xlab="", 
     ylab="", 
     border="white")
hist(residuals_svm, 
     col= rgb(0.804, 0.361, 0.361, alpha=0.5),
     add=TRUE, 
     border="white")

# Plotting model residual distributions (WAR)
hist(war_mod$residuals, 
     col="lightblue3",
     xlim=c(min(c(war_mod$residuals, residuals_svm)), 6),
     main="WAR OLS vs. SVR", 
     xlab="", 
     ylab="",
     yaxt = "n",
     border="white")
hist(residuals_svm, 
     col= rgb(0.804, 0.361, 0.361, alpha=0.5),
     add=TRUE, 
     border="white")

# Adding a legend to distinguish the histograms
 legend(3.5, 600, inset = c(-10, 0),
        legend=c("OLS Residuals", "SVR Residuals"), 
        fill=c(rgb(0.678, 0.847, 0.902), rgb(0.804, 0.361, 0.361, alpha = 0.5)))

# Plotting model residual distributions (WAR + avg.war + player_age)
hist(war_mod2$residuals, 
     col="lightblue3",
     xlim=c(min(c(war_mod$residuals, residuals_svm)), 6),
     main="WAR + avg.war + player_age OLS vs. SVR", 
     xlab="", 
     ylab="", 
     border="white")
hist(residuals_svm, 
     col= rgb(0.804, 0.361, 0.361, alpha=0.5),
     add=TRUE, 
     border="white")

# Plotting model residual distributions (Final OLS Model)
hist(final_mod$residuals, 
     col="lightblue3",
     xlim=c(min(c(war_mod$residuals, residuals_svm)), 6),
     main="Optimal OLS vs. SVR", 
     xlab="", 
     ylab="", 
     yaxt = "n",
     border="white")
hist(residuals_svm, 
     col= rgb(0.804, 0.361, 0.361, alpha=0.5),
     add=TRUE, 
     border="white")
