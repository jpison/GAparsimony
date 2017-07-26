GAparsimony
===========

GAparsimony R package is a GA-based optimization method for searching
accurate parsimonious models by combining feature selection (FS), model
hyperparameter optimization (HO), and parsimonious model selection
(PMS).

PMS is based on separate cost and complexity evaluations. The best
individuals are initially sorted by an error fitness function, and
afterwards, models with similar costs are rearranged according to model
complexity measurement so as to foster models of lesser complexity. The
algorithm can be run sequentially or in parallel using an explicit
master-slave parallelization.

Installation
------------

Get the released version from CRAN:

``` {.r}
#install.packages("GAparsimony")
```

Or the development version from GitHub:

``` {.r}
# install.packages("devtools")
devtools::install_github("jpison/GAparsimony")
```

    ## Skipping install of 'GAparsimony' from a github remote, the SHA1 (211d3933) has not changed since last install.
    ##   Use `force = TRUE` to force installation

How to use this package
-----------------------

### Example 1: Classification

This example shows how to search, for the *Sonar* database, a parsimony
classification SVM model with **GAparsimony** and **caret** packages.

First, we create a 80% of database for searching the model and the
remaining 20% for the test database. The test database will be only used
for checking the models’ generalization capability.

``` {.r}
# Training and test Datasets
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Loading required package: withr

``` {.r}
library(GAparsimony)
```

    ## Loading required package: foreach

    ## Loading required package: iterators

``` {.r}
library(mlbench)
```

    ## Warning: package 'mlbench' was built under R version 3.3.3

``` {.r}
data(Sonar)

set.seed(1234)
inTraining <- createDataPartition(Sonar$Class, p=.80, list=FALSE)
data_train <- Sonar[ inTraining,]
data_test  <- Sonar[-inTraining,]
```

With small databases, it is highly recommended to execute
**GAparsimony** with a different set of test databases in order to find
the most important input features and model parameters. In this example,
one iteration is showed with a training database composed of 60 input
features and 167 instances, and a test database with only 41 instances.
Therefore, a robust validation metric will be necessary.

``` {.r}
print(dim(data_train))
```

    ## [1] 167  61

``` {.r}
print(dim(data_test))
```

    ## [1] 41 61

In the next step, a fitness function is created, *fitness\_SVM()*.

This function extracts **C** and **sigma** SVM parameters from the first
two elements of *chromosome* vector. Next 60 elements of chromosome
correspond with the selected input features, *selec\_feat*. They are
binarized to one when they are one greater than \> 0.50.

A SVM model is trained with these parameters and selected input
features. Finally, *fitness\_SVM()* returns a vector with three values:
the kappa statistic obtained with a 10 repeats of a 10-fold
cross-validation process, the kappa measured with the test database to
check the model generalization capability, and the model complexity.

In this example, the model complexity combines the number of features
multiplied by 1E6 plus the number of support vectors in the selected
model. Therefore, PMS considers the most parsimonious model with the
lower number of features. Between two models with the same number of
features, the lower number of support vectors will determine the most
parsimonious model.

However, other parsimonious metrics could be considered in future
applications.

``` {.r}
# Function to evaluate each SVM individual
# ----------------------------------------
fitness_SVM <- function(chromosome, ...)
{
  # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
  tuneGrid <- data.frame(C=chromosome[1],sigma=chromosome[2])
  
  # Next values of chromosome are the selected features (TRUE if > 0.50)
  selec_feat <- chromosome[3:length(chromosome)]>0.50
  
  # Return -Inf if there is not selected features
  if (sum(selec_feat)<1) return(c(kappa_val=-Inf,kappa_test=-Inf,complexity=Inf))
  
  # Extract features from the original DB plus response (last column)
  data_train_model <- data_train[,c(selec_feat,TRUE)]
  data_test_model <- data_test[,c(selec_feat,TRUE)]
  
  # How to validate each individual
  # 'repeats' could be increased to obtain a more robust validation metric. Also,
  # 'number' of folds could be adjusted to improve the measure.
  train_control <- trainControl(method = "repeatedcv",number = 10,repeats = 10)

  # train the model
  set.seed(1234)
  model <- train(Class ~ ., data=data_train_model, trControl=train_control, 
                 method="svmRadial", tuneGrid=tuneGrid, verbose=F)

  # Extract kappa statistics (repeated k-fold CV and testing kappa)
  kappa_val <- model$results$Kappa
  kappa_test <- postResample(pred=predict(model, data_test_model),
                                obs=data_test_model[,ncol(data_test_model)])[2]
  # Obtain Complexity = Num_Features*1E6+Number of support vectors
  complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 
  
  # Return(-validation error, -testing error, model_complexity)
  vect_errors <- c(kappa_val=kappa_val,kappa_test=kappa_test,complexity=complexity)
  return(vect_errors)
}
```

The GA-PARSIMONY process begins defining the range of the SVM parameters
and their names. Also, *rerank\_error* can be tuned with different
*ga\_parsimony* runs to improve the **model generalization capability**.
In this example, *rerank\_error* has been fixed to 0.001 but other
values could improve the trade-off between model complexity and model
accuracy.

``` {.r}
# ---------------------------------------------------------------------------------
# Search the best parsimonious model with GA-PARSIMONY by using Feature Selection,
# Parameter Tuning and Parsimonious Model Selection
# ---------------------------------------------------------------------------------
library(GAparsimony)

# Ranges of size and decay
min_param <- c(00.0001, 0.00001)
max_param <- c(99.9999, 0.99999)
names_param <- c("C","sigma")

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error <- 0.001
```

Starts the GA optimizaton process with 40 individuals per generation and
a maximum number of 100 iterations with an early stopping when
validation measure does not increase significantly in 10 generations.
Parallel is activated. In addition, history of each iteration is saved
in order to use *plot* and *parsimony\_importance* methods.

``` {.r}
# GA optimization process with 40 individuals per population, 100 max generations with an early stopping of 10 generations
# (8 minutes with 8 cores)!!!!! Reduce maxiter to understand the process if it is too computational expensive...
GAparsimony_model <- ga_parsimony(fitness=fitness_SVM,
                                  min_param=min_param,
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=ncol(data_train)-1,
                                  names_features=colnames(data_train)[-ncol(data_train)],
                                  keep_history = TRUE,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 100, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  parallel = TRUE, seed_ini = 1234)
```

Show the results of the best parsimonious model. We can see similar
validation and testing kappas.

``` {.r}
print(paste0("Best Parsimonious SVM with C=",GAparsimony_model@bestsolution['C'],
             " sigma=", GAparsimony_model@bestsolution['sigma'], " -> ",
             " KappaVal=",round(GAparsimony_model@bestsolution['fitnessVal'],6),
             " KappaTst=",round(GAparsimony_model@bestsolution['fitnessTst'],6),
             " Num Features=",round(GAparsimony_model@bestsolution['complexity']/1E6,0),
             " Complexity=",round(GAparsimony_model@bestsolution['complexity'],2)))
```

    ## [1] "Best Parsimonious SVM with C=44.1161803299857 sigma=0.043852464390368 ->  KappaVal=0.855479 KappaTst=0.852341 Num Features=24 Complexity=24000113"

``` {.r}
print(summary(GAparsimony_model))
```

    ## +------------------------------------+
    ## |             GA-PARSIMONY           |
    ## +------------------------------------+
    ## 
    ## GA-PARSIMONY settings: 
    ##  Number of Parameters      =  2 
    ##  Number of Features        =  60 
    ##  Population size           =  40 
    ##  Maximum of generations    =  100 
    ##  Number of early-stop gen. =  10 
    ##  Elitism                   =  8 
    ##  Crossover probability     =  0.8 
    ##  Mutation probability      =  0.1 
    ##  Max diff(error) to ReRank =  0.001 
    ##  Perc. of 1s in first popu.=  0.9 
    ##  Prob. to be 1 in mutation =  0.1 
    ##  Search domain = 
    ##                 C   sigma V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14
    ## Min_param  0.0001 0.00001  0  0  0  0  0  0  0  0  0   0   0   0   0   0
    ## Max_param 99.9999 0.99999  1  1  1  1  1  1  1  1  1   1   1   1   1   1
    ##           V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 V29 V30
    ## Min_param   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ## Max_param   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
    ##           V31 V32 V33 V34 V35 V36 V37 V38 V39 V40 V41 V42 V43 V44 V45 V46
    ## Min_param   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ## Max_param   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
    ##           V47 V48 V49 V50 V51 V52 V53 V54 V55 V56 V57 V58 V59 V60
    ## Min_param   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ## Max_param   1   1   1   1   1   1   1   1   1   1   1   1   1   1
    ## 
    ## 
    ## GA-PARSIMONY results: 
    ##  Iterations                = 31 
    ##  Best indiv's validat.cost = 0.8554789 
    ##  Best indiv's testing cost = 0.8523409 
    ##  Best indiv's complexity   = 24000113 
    ##  Elapsed time in minutes   = 7.543422 
    ## 
    ## 
    ## BEST SOLUTION = 
    ##                    [,1]
    ## fitnessVal 8.554789e-01
    ## fitnessTst 8.523409e-01
    ## complexity 2.400011e+07
    ## C          4.411618e+01
    ## sigma      4.385246e-02
    ## V1         1.000000e+00
    ## V2         0.000000e+00
    ## V3         0.000000e+00
    ## V4         0.000000e+00
    ## V5         1.000000e+00
    ## V6         0.000000e+00
    ## V7         0.000000e+00
    ## V8         1.000000e+00
    ## V9         1.000000e+00
    ## V10        1.000000e+00
    ## V11        1.000000e+00
    ## V12        1.000000e+00
    ## V13        0.000000e+00
    ## V14        0.000000e+00
    ## V15        0.000000e+00
    ## V16        1.000000e+00
    ## V17        1.000000e+00
    ## V18        0.000000e+00
    ## V19        0.000000e+00
    ## V20        0.000000e+00
    ## V21        0.000000e+00
    ## V22        0.000000e+00
    ## V23        1.000000e+00
    ## V24        0.000000e+00
    ## V25        0.000000e+00
    ## V26        1.000000e+00
    ## V27        0.000000e+00
    ## V28        1.000000e+00
    ## V29        0.000000e+00
    ## V30        0.000000e+00
    ## V31        0.000000e+00
    ## V32        1.000000e+00
    ## V33        1.000000e+00
    ## V34        0.000000e+00
    ## V35        0.000000e+00
    ## V36        1.000000e+00
    ## V37        0.000000e+00
    ## V38        0.000000e+00
    ## V39        0.000000e+00
    ## V40        1.000000e+00
    ## V41        0.000000e+00
    ## V42        1.000000e+00
    ## V43        0.000000e+00
    ## V44        0.000000e+00
    ## V45        1.000000e+00
    ## V46        0.000000e+00
    ## V47        0.000000e+00
    ## V48        0.000000e+00
    ## V49        0.000000e+00
    ## V50        0.000000e+00
    ## V51        0.000000e+00
    ## V52        1.000000e+00
    ## V53        1.000000e+00
    ## V54        1.000000e+00
    ## V55        1.000000e+00
    ## V56        1.000000e+00
    ## V57        0.000000e+00
    ## V58        0.000000e+00
    ## V59        0.000000e+00
    ## V60        1.000000e+00

Plot GA evolution.

![GA-PARSIMONY Evolution](https://github.com/jpison/GAparsimony/blob/master/images/classification.png)

GA-PARSIMONY evolution

Show percentage of appearance of each feature in elitists

``` {.r}
# Percentage of appearance of each feature in elitists
print(parsimony_importance(GAparsimony_model))
```

    ##       V54       V40       V56       V60        V9       V45       V36 
    ## 99.596774 99.193548 98.387097 98.387097 97.983871 97.983871 97.580645 
    ##       V52       V32       V16       V23       V26       V55       V11 
    ## 97.580645 96.774194 96.370968 94.758065 93.548387 93.548387 93.145161 
    ##       V33       V10       V42       V17        V1       V53        V5 
    ## 92.338710 91.935484 89.516129 87.903226 86.693548 85.080645 80.241935 
    ##       V12       V28        V8       V57       V30       V37        V3 
    ## 79.435484 78.225806 63.306452 54.435484 52.822581 51.209677 35.887097 
    ##       V20       V41       V51       V31       V22       V47       V18 
    ## 35.887097 33.870968 32.661290 27.419355 26.612903 25.403226 22.580645 
    ##       V38       V25       V35        V4       V14       V43       V21 
    ## 22.580645 22.177419 21.774194 20.967742 20.967742 20.564516 18.951613 
    ##       V49       V44       V48       V29       V15       V50       V13 
    ## 17.741935 16.532258 16.532258 15.322581 14.516129 12.903226 12.500000 
    ##       V34        V7        V6        V2       V27       V39       V46 
    ## 11.693548  9.677419  8.870968  8.467742  7.661290  7.661290  6.451613 
    ##       V58       V19       V24       V59 
    ##  6.048387  5.241935  4.838710  4.435484

### Example 2: Regression

This example shows how to search, for the *Boston* database, a parsimony
regressor ANN model with **GAparsimony** and **caret** packages.

First, we create a 80% of database for searching the model and the
remaining 20% for the test database. The test database will be only used
for checking the models’ generalization capability.

``` {.r}
# Load Boston database and scale it
library(MASS)
data(Boston)
Boston_scaled <- data.frame(scale(Boston))

# Define an 80%/20% train/test split of the dataset
set.seed(1234)
trainIndex <- createDataPartition(Boston[,"medv"], p=0.80, list=FALSE)
data_train <- Boston_scaled[trainIndex,]
data_test <- Boston_scaled[-trainIndex,]
# Restore 'Response' to original values
data_train[,ncol(data_train)] <- Boston$medv[trainIndex]
data_test[,ncol(data_test)] <- Boston$medv[-trainIndex]
print(dim(data_train))
```

    ## [1] 407  14

``` {.r}
print(dim(data_test))
```

    ## [1] 99 14

Similar to the previous example a fitness function is created,
*fitness\_NNET()*.

This function extracts **size** and **decay** NNET parameters from the
first two elements of *chromosome* vector. Next 13 elements of
chromosome correspond with the selected input features, *selec\_feat*.
They are binarized to one when they are one greater than \> 0.50.

A NNET model is trained with these parameters and selected input
features. Finally, *fitness\_NNET()* returns a vector with three values:
the negative RMSE obtained with a 5 repeats of a 10-fold
cross-validation process, the negative RMSE measured with the test
database to check the model generalization capability, and the model
complexity. Negative values of RMSE are returned because *ga\_parsimony*
**maximizes** the validation cost,

In this example, the model complexity combines the number of features
multiplied by 1E6 plus the sum of the squared network weights.
Therefore, PMS considers the most parsimonious model with the lower
number of features. Between two models with the same number of features,
the lower sum of the squared network weights will determine the most
parsimonious model.

However, other parsimonious metrics could be considered in future
applications.

``` {.r}
# Function to evaluate each ANN individual
# ----------------------------------------
fitness_NNET <- function(chromosome, ...)
{
  # First two values in chromosome are 'size' & 'decay' of 'nnet' method
  tuneGrid <- data.frame(size=round(chromosome[1]),decay=chromosome[2])
  
  # Next values of chromosome are the selected features (TRUE if > 0.50)
  selec_feat <- chromosome[3:length(chromosome)]>0.50
  if (sum(selec_feat)<1) return(c(rmse_val=-Inf,rmse_test=-Inf,complexity=Inf))
  
  # Extract features from the original DB plus response (last column)
  data_train_model <- data_train[,c(selec_feat,TRUE)]
  data_test_model <- data_test[,c(selec_feat,TRUE)]
  
  # How to validate each individual
  # 'repeats' could be increased to obtain a more robust validation metric. Also,
  # 'number' of folds could be adjusted to improve the measure.
  train_control <- trainControl(method = "repeatedcv",number = 10,repeats = 5)
  
  # train the model
  set.seed(1234)
  model <- train(medv ~ ., data=data_train_model, trControl=train_control, 
                 method="nnet", tuneGrid=tuneGrid, trace=F, linout = 1)
  
  # Extract errors
  rmse_val <- model$results$RMSE
  rmse_test <- sqrt(mean((unlist(predict(model, newdata = data_test_model)) - data_test_model$medv)^2))
  # Obtain Complexity = Num_Features*1E6+sum(neural_weights^2)
  complexity <- sum(selec_feat)*1E6+sum(model$finalModel$wts*model$finalModel$wts)  
  
  # Return(-validation error, -testing error, model_complexity)
  # errors are negative because GA-PARSIMONY tries to maximize values
  vect_errors <- c(rmse_val=-rmse_val,rmse_test=-rmse_test,complexity=complexity)
  return(vect_errors)
}
```

Initial settings.

``` {.r}
# ---------------------------------------------------------------------------------
# Search the best parsimonious model with GA-PARSIMONY by using Feature Selection,
# Parameter Tuning and Parsimonious Model Selection
# ---------------------------------------------------------------------------------
library(GAparsimony)

# Ranges of size and decay
min_param <- c(1, 0.0001)
max_param <- c(25 , 0.9999)
names_param <- c("size","decay")

# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error <- 0.01  
```

Search the best parsimonious model.

``` {.r}
# GA optimization process with 40 individuals per population, 100 max generations with an early stopping of 10 generations
# (34 minutes with 8 cores)!!!!! Reduce maxiter to understand the process if it is too computational expensive...
GAparsimony_model <- ga_parsimony(fitness=fitness_NNET,
                                  min_param=min_param,
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=ncol(data_train)-1,
                                  names_features=colnames(data_train)[-ncol(data_train)],
                                  keep_history = TRUE,
                                  rerank_error = rerank_error,
                                  popSize = 40,
                                  maxiter = 100, early_stop=10,
                                  feat_thres=0.90, # Perc selected features in first generation
                                  feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                  not_muted=2,
                                  parallel = TRUE, seed_ini = 1234)

print(paste0("Best Parsimonious ANN with ",round(GAparsimony_model@bestsolution['size']),
             " hidden neurons and decay=", GAparsimony_model@bestsolution['decay'], " -> ",
             " RMSEVal=",round(-GAparsimony_model@bestsolution['fitnessVal'],6),
             " RMSETst=",round(-GAparsimony_model@bestsolution['fitnessTst'],6)))
```

    ## [1] "Best Parsimonious ANN with 18 hidden neurons and decay=0.982727317560044 ->  RMSEVal=3.084126 RMSETst=3.025289"

``` {.r}
print(summary(GAparsimony_model))
```

    ## +------------------------------------+
    ## |             GA-PARSIMONY           |
    ## +------------------------------------+
    ## 
    ## GA-PARSIMONY settings: 
    ##  Number of Parameters      =  2 
    ##  Number of Features        =  13 
    ##  Population size           =  40 
    ##  Maximum of generations    =  100 
    ##  Number of early-stop gen. =  10 
    ##  Elitism                   =  8 
    ##  Crossover probability     =  0.8 
    ##  Mutation probability      =  0.1 
    ##  Max diff(error) to ReRank =  0.01 
    ##  Perc. of 1s in first popu.=  0.9 
    ##  Prob. to be 1 in mutation =  0.1 
    ##  Search domain = 
    ##           size  decay crim zn indus chas nox rm age dis rad tax ptratio
    ## Min_param    1 0.0001    0  0     0    0   0  0   0   0   0   0       0
    ## Max_param   25 0.9999    1  1     1    1   1  1   1   1   1   1       1
    ##           black lstat
    ## Min_param     0     0
    ## Max_param     1     1
    ## 
    ## 
    ## GA-PARSIMONY results: 
    ##  Iterations                = 30 
    ##  Best indiv's validat.cost = -3.084126 
    ##  Best indiv's testing cost = -3.025289 
    ##  Best indiv's complexity   = 11001231 
    ##  Elapsed time in minutes   = 34.04644 
    ## 
    ## 
    ## BEST SOLUTION = 
    ##                     [,1]
    ## fitnessVal -3.084126e+00
    ## fitnessTst -3.025289e+00
    ## complexity  1.100123e+07
    ## size        1.752418e+01
    ## decay       9.827273e-01
    ## crim        1.000000e+00
    ## zn          1.000000e+00
    ## indus       1.000000e+00
    ## chas        0.000000e+00
    ## nox         1.000000e+00
    ## rm          1.000000e+00
    ## age         1.000000e+00
    ## dis         1.000000e+00
    ## rad         1.000000e+00
    ## tax         1.000000e+00
    ## ptratio     1.000000e+00
    ## black       0.000000e+00
    ## lstat       1.000000e+00

Plot GA evolution.

![GA-PARSIMONY
evolution](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABUAAAAPACAMAAADDuCPrAAAAsVBMVEUAAAAAADEAADoAAFUAAGYAMHcAOjoAOmYAOpAAVZYAZmYAZpAAZrYXAAA6AAA6OgA6OpA6ZmY6kJA6kLY6kNtmAABmAGZmOgBmZjpmZmZmkJBmtrZmtv+QOgCQZgCQZpCQkDqQkGaQkNuQtpCQvJCQ29uQ2/+WVQCW09O109O2ZgC225C2/9u2///MzMzT09PbkDrbtmbb/7bb/9vb///l5eX/tmb/25D//7b//9v///+YOKqNAAAACXBIWXMAAB2HAAAdhwGP5fFlAAAgAElEQVR4nO2dDZvkuHpQzXAJSxiakNsDF+gBQsgCFW62Qpbqmfr/P4zyh2zJkqwPv7Jl+5zn2Z3qKpUkv7ZPyZIsN08AAMii2bsCAABHBYECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACATBAoAkAkCBQDIBIECAGSCQAEAMkGgAACZIFAAgEwQKABAJggUACAT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)

GA-PARSIMONY evolution

Show percentage of appearance of each feature in elitists

``` {.r}
# Percentage of appearance of each feature in elitists
print(parsimony_importance(GAparsimony_model))
```

    ##         rm        age        dis        rad        tax      lstat 
    ## 100.000000 100.000000 100.000000 100.000000 100.000000 100.000000 
    ##        nox    ptratio      indus         zn       crim      black 
    ##  99.583333  99.583333  99.166667  98.333333  97.083333  25.416667 
    ##       chas 
    ##   5.416667

References
----------

Sanz-Garcia A., Fernandez-Ceniceros J., Antonanzas-Torres F.,
Pernia-Espinoza A.V., Martinez-de-Pison F.J. (2015). GA-PARSIMONY: A
GA-SVR approach with feature selection and parameter optimization to
obtain parsimonious solutions for predicting temperature settings in a
continuous annealing furnace. Applied Soft Computing 35, 23-38.

Urraca R., Sodupe-Ortega E., Antonanzas E., Antonanzas-Torres F.,
Martinez-de-Pison, F.J. (2017). Evaluation of a novel GA-based
methodology for model structure selection: The GA-PARSIMONY.
Neurocomputing, Online July 2017.
[https://doi.org/10.1016/j.neucom.2016.08.154](https://doi.org/10.1016/j.neucom.2016.08.154)

Fernandez-Ceniceros J., Sanz-Garcia A., Antonanzas-Torres F.,
Martinez-de-Pison F.J. (2015). A numerical-informational approach for
characterising the ductile behaviour of the T-stub component. Part 2:
Parsimonious soft-computing-based metamodel. Engineering Structures 82,
249-260.

Antonanzas-Torres F., Urraca R., Antonanzas J., Fernandez-Ceniceros J.,
Martinez-de-Pison F.J. (2015). Generation of daily global solar
irradiation with support vector machines for regression. Energy
Conversion and Management 96, 277-286.
