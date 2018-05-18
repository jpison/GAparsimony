GAparsimony
===========

GAparsimony for R is a package for searching with genetic algorithms (GA) 
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
install.packages("GAparsimony")
```

Or the development version from GitHub:

``` {.r}
# install.packages("devtools")
devtools::install_github("jpison/GAparsimony")
```

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


How to use this package
-----------------------

### Example 1: Classification

This example shows how to search, for the *Sonar* database, a parsimony
SVM classificator with **GAparsimony** and **caret** packages.

First, we create a 80% of database for searching the model and the
remaining 20% for testing. The test database will be only used
for checking the models’ generalization capability.

``` {.r}
# Training and test Datasets
library(caret)
library(GAparsimony)
library(mlbench)
data(Sonar)

set.seed(1234)
inTraining <- createDataPartition(Sonar$Class, p=.80, list=FALSE)
data_train <- Sonar[ inTraining,]
data_test  <- Sonar[-inTraining,]
```
With small databases, it is highly recommended to execute
**GAparsimony** with different seeds in order to find
the most important input features and model parameters.

In this example, one GA optimization is presented with a training database 
composed of 60 input features and 167 instances, and a test database with only 41 instances.
Hence, a robust validation metric is necessary. Thus, a repeated cross-validation is performed.

``` {.r}
print(dim(data_train))
print(dim(data_test))
```
    ## [1] 167  61
    ## [1] 41 61

In the next step, a fitness function is created: *fitness\_SVM()*.

This function extracts **C** and **sigma** SVM parameters from the first
two elements of *chromosome* vector. Next 60 elements of chromosome
correspond with the selected input features, *selec\_feat*. They are
binarized to one when they are one greater than \> 0.50.

A SVM model is trained with these parameters and the selected input
features. Finally, *fitness\_SVM()* returns a vector with three values:
the *kappa* statistic obtained with the mean of 10 runs of a 10-fold
cross-validation process, the *kappa* measured with the test database to
check the model generalization capability, and the model complexity.

In this example, the model complexity combines the number of selected features
multiplied by 1E6 plus the number of support vectors of each model. 
Therefore, PMS considers the most parsimonious model with the
lower number of features. Between two models with the same number of
features, the lower number of support vectors will determine the most
parsimonious model. However, other parsimonious metrics could be considered in future
applications (AIC, BIC, GDF, ...)..

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

  # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
  kappa_val <- model$results$Kappa
  kappa_test <- postResample(pred=predict(model, data_test_model),
                                obs=data_test_model[,ncol(data_test_model)])[2]
  # Obtain Complexity = Num_Features*1E6+Number of support vectors
  complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 
  
  # Return(validation score, testing score, model_complexity)
  vect_errors <- c(kappa_val=kappa_val,kappa_test=kappa_test,complexity=complexity)
  return(vect_errors)
}
```

The GA-PARSIMONY process begins defining the range of the SVM parameters
and their names. Also, *rerank\_error* can be tuned with different
*ga\_parsimony* runs to improve the **model generalization capability**.
In this example, *rerank\_error* has been fixed to 0.001 but other
values could improve the trade-off between model complexity and model
accuracy. For example, with *rerank\_error=0.01*, we can be interested 
in obtaining models with a smaller number of inputs with a *kappa* rounded
to two decimals.

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
# (CAUTION! 7.34 minutes with 8 cores)!!!!! Reduce maxiter to understand the process if it is too computational expensive...
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
validation and testing *kappas*.

``` {.r}
print(paste0("Best Parsimonious SVM with C=",GAparsimony_model@bestsolution['C'],
             " sigma=", GAparsimony_model@bestsolution['sigma'], " -> ",
             " KappaVal=",round(GAparsimony_model@bestsolution['fitnessVal'],6),
             " KappaTst=",round(GAparsimony_model@bestsolution['fitnessTst'],6),
             " Num Features=",round(GAparsimony_model@bestsolution['complexity']/1E6,0),
             " Complexity=",round(GAparsimony_model@bestsolution['complexity'],2)))
```
``` {.r}
    ## [1] "Best Parsimonious SVM with C=44.1161803299857 sigma=0.043852464390368 ->  KappaVal=0.855479 KappaTst=0.852341 Num Features=24 Complexity=24000113"
```
Summary() function shows the GA initial settings and two solutions: the solution with the best validation score in the whole GA optimization process, and finally, the best parsimonious individual at the last generation. 

``` {.r}
print(summary(GAparsimony_model))

# +------------------------------------+
##   |             GA-PARSIMONY           |
##   +------------------------------------+
## 
##   GA-PARSIMONY settings:
#   Number of Parameters      =  2
#   Number of Features        =  60
#   Population size           =  40
#   Maximum of generations    =  100
#   Number of early-stop gen. =  10
#   Elitism                   =  8
#   Crossover probability     =  0.8
#   Mutation probability      =  0.1
#   Max diff(error) to ReRank =  0.001
#   Perc. of 1s in first popu.=  0.9
#   Prob. to be 1 in mutation =  0.1
#   Search domain =
#     C   sigma V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28
#   Min_param  0.0001 0.00001  0  0  0  0  0  0  0  0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#   Max_param 99.9999 0.99999  1  1  1  1  1  1  1  1  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
#   V29 V30 V31 V32 V33 V34 V35 V36 V37 V38 V39 V40 V41 V42 V43 V44 V45 V46 V47 V48 V49 V50 V51 V52 V53 V54 V55 V56 V57 V58
#   Min_param   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#   Max_param   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
#   V59 V60
#   Min_param   0   0
#   Max_param   1   1
# 
# 
#   GA-PARSIMONY results:
#     Iterations                = 31
#   Best validation score = 0.8564253
# 
# 
#   Solution with the best validation score in the whole GA process =
#     [,1]
#   fitnessVal 8.564253e-01
#   fitnessTst 8.523409e-01
#   complexity 2.500011e+07
#   C          4.411618e+01
#   sigma      4.385246e-02
#   V1         1.000000e+00
#   V2         0.000000e+00
#   V3         0.000000e+00
#   V4         0.000000e+00
#   V5         1.000000e+00
#   V6         0.000000e+00
#   V7         0.000000e+00
#   V8         1.000000e+00
#   V9         1.000000e+00
#   V10        1.000000e+00
#   V11        1.000000e+00
#   V12        1.000000e+00
#   V13        0.000000e+00
#   V14        0.000000e+00
#   V15        0.000000e+00
#   V16        1.000000e+00
#   V17        1.000000e+00
#   V18        0.000000e+00
#   V19        0.000000e+00
#   V20        0.000000e+00
#   V21        0.000000e+00
#   V22        0.000000e+00
#   V23        1.000000e+00
#   V24        0.000000e+00
#   V25        0.000000e+00
#   V26        1.000000e+00
#   V27        0.000000e+00
#   V28        1.000000e+00
#   V29        1.000000e+00
#   V30        0.000000e+00
#   V31        0.000000e+00
#   V32        1.000000e+00
#   V33        1.000000e+00
#   V34        0.000000e+00
#   V35        0.000000e+00
#   V36        1.000000e+00
#   V37        0.000000e+00
#   V38        0.000000e+00
#   V39        0.000000e+00
#   V40        1.000000e+00
#   V41        0.000000e+00
#   V42        1.000000e+00
#   V43        0.000000e+00
#   V44        0.000000e+00
#   V45        1.000000e+00
#   V46        0.000000e+00
#   V47        0.000000e+00
#   V48        0.000000e+00
#   V49        0.000000e+00
#   V50        0.000000e+00
#   V51        0.000000e+00
#   V52        1.000000e+00
#   V53        1.000000e+00
#   V54        1.000000e+00
#   V55        1.000000e+00
#   V56        1.000000e+00
#   V57        0.000000e+00
#   V58        0.000000e+00
#   V59        0.000000e+00
#   V60        1.000000e+00
# 
# 
#   Results of the best individual at the last generation =
#     Best indiv's validat.cost = 0.8554789
#   Best indiv's testing cost = 0.8523409
#   Best indiv's complexity   = 24000113
#   Elapsed time in minutes   = 7.650901
# 
# 
#   BEST SOLUTION =
#   [,1]
#   fitnessVal 8.554789e-01
#   fitnessTst 8.523409e-01
#   complexity 2.400011e+07
#   C          4.411618e+01
#   sigma      4.385246e-02
#   V1         1.000000e+00
#   V2         0.000000e+00
#   V3         0.000000e+00
#   V4         0.000000e+00
#   V5         1.000000e+00
#   V6         0.000000e+00
#   V7         0.000000e+00
#   V8         1.000000e+00
#   V9         1.000000e+00
#   V10        1.000000e+00
#   V11        1.000000e+00
#   V12        1.000000e+00
#   V13        0.000000e+00
#   V14        0.000000e+00
#   V15        0.000000e+00
#   V16        1.000000e+00
#   V17        1.000000e+00
#   V18        0.000000e+00
#   V19        0.000000e+00
#   V20        0.000000e+00
#   V21        0.000000e+00
#   V22        0.000000e+00
#   V23        1.000000e+00
#   V24        0.000000e+00
#   V25        0.000000e+00
#   V26        1.000000e+00
#   V27        0.000000e+00
#   V28        1.000000e+00
#   V29        0.000000e+00
#   V30        0.000000e+00
#   V31        0.000000e+00
#   V32        1.000000e+00
#   V33        1.000000e+00
#   V34        0.000000e+00
#   V35        0.000000e+00
#   V36        1.000000e+00
#   V37        0.000000e+00
#   V38        0.000000e+00
#   V39        0.000000e+00
#   V40        1.000000e+00
#   V41        0.000000e+00
#   V42        1.000000e+00
#   V43        0.000000e+00
#   V44        0.000000e+00
#   V45        1.000000e+00
#   V46        0.000000e+00
#   V47        0.000000e+00
#   V48        0.000000e+00
#   V49        0.000000e+00
#   V50        0.000000e+00
#   V51        0.000000e+00
#   V52        1.000000e+00
#   V53        1.000000e+00
#   V54        1.000000e+00
#   V55        1.000000e+00
#   V56        1.000000e+00
#   V57        0.000000e+00
#   V58        0.000000e+00
#   V59        0.000000e+00
#   V60        1.000000e+00
```

Plot GA evolution.

```{r fig.cap = "GA-PARSIMONY evolution", echo=FALSE}
# Plot GA evolution ('keep_history' must be TRUE)
elitists <- plot(GAparsimony_model, window=FALSE, general_cex=0.6, pos_cost_num=-1, pos_feat_num=-1, digits_plot=3)
```


![GA-PARSIMONY Evolution](https://github.com/jpison/GAparsimony/blob/master/images/classification.png)

GA-PARSIMONY evolution

Show percentage of appearance for each feature in elitists

``` {.r}
# Percentage of appearance for each feature in elitists
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

This example shows how to search, for the *Boston* database, a parsimonious
ANN model for regression and with **GAparsimony** and **caret** packages.

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
print(dim(data_test))
```

    ## [1] 407  14
    ## [1] 99 14

Similar to the previous example, a fitness function is created:
*fitness\_NNET()*.

This function extracts **size** and **decay** NNET parameters from the
first two elements of *chromosome* vector. Next 13 elements of
chromosome correspond with the selected input features, *selec\_feat*.
They are binarized to one when are greater than \> 0.50.

A NNET model is trained with these parameters and selected input
features. Finally, *fitness\_NNET()* returns a vector with three values:
the negative RMSE obtained with a 5 repeats of a 10-fold
cross-validation process, the negative RMSE measured with the test
database to check the model generalization capability, and the model
complexity. Negative values of RMSE are returned because *ga\_parsimony*
tries to **maximize** the validation cost,

In this example, the model complexity combines the number of features
multiplied by 1E6 plus the sum of the squared network weights which measures 
the internal complexity of the ANN.

Therefore, PMS considers the most parsimonious model with the lower
number of features. Between two models with the same number of features,
the lower sum of the squared network weights will determine the most
parsimonious model (smaller weights reduce the propagation of disturbances).

However, other parsimonious metrics could be considered in future
applications (AIC, BIC, GDF, ...).

``` {.r}
# Function to evaluate each ANN individual
# ----------------------------------------
fitness_NNET <- function(chromosome, ...)
{
  # First two values in chromosome are 'size' & 'decay' of the 'nnet' method
  tuneGrid <- data.frame(size=round(chromosome[1]),decay=chromosome[2])
  
  # Next values of chromosome are the selected features that are > 0.50
  selec_feat <- chromosome[3:length(chromosome)]>0.50
  if (sum(selec_feat)<1) return(c(rmse_val=-Inf,rmse_test=-Inf,complexity=Inf))
  
  # Extract features from the original DB plus response (last column)
  data_train_model <- data_train[,c(selec_feat,TRUE)]
  data_test_model <- data_test[,c(selec_feat,TRUE)]
  
  # How to validate each individual
  # 'repeats' could be increased to obtain a more robust validation metric. Also,
  # 'number' of folds could be adjusted to improve the measure.
  train_control <- trainControl(method = "repeatedcv",number = 10,repeats = 5)
  
  # Train the model
  set.seed(1234)
  model <- train(medv ~ ., data=data_train_model, trControl=train_control, 
                 method="nnet", tuneGrid=tuneGrid, trace=F, linout = 1)
  
  # Extract errors
  rmse_val <- model$results$RMSE
  rmse_test <- sqrt(mean((unlist(predict(model, newdata = data_test_model)) - data_test_model$medv)^2))
  # Obtain Complexity = Num_Features*1E6+sum(neural_weights^2)
  complexity <- sum(selec_feat)*1E6+sum(model$finalModel$wts*model$finalModel$wts)  
  
  # Return(-validation error, -testing error, model_complexity)
  # Errors are negative because GA-PARSIMONY tries to maximize values
  vect_errors <- c(rmse_val=-rmse_val,rmse_test=-rmse_test,complexity=complexity)
  return(vect_errors)
}
```

Initial GA settings.

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
# (CAUTION!! 34 minutes with 8 cores)!!!!! Reduce maxiter to understand the process if it is too computational expensive...
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
Summary() function shows the GA initial settings and two solutions: the solution with the best validation score in the whole GA optimization process, and finally, the best parsimonious individual at the last generation.

``` {.r}
+------------------------------------+
|             GA-PARSIMONY           |
+------------------------------------+

GA-PARSIMONY settings: 
 Number of Parameters      =  2 
 Number of Features        =  13 
 Population size           =  40 
 Maximum of generations    =  100 
 Number of early-stop gen. =  10 
 Elitism                   =  8 
 Crossover probability     =  0.8 
 Mutation probability      =  0.1 
 Max diff(error) to ReRank =  0.01 
 Perc. of 1s in first popu.=  0.9 
 Prob. to be 1 in mutation =  0.1 
 Search domain = 
          size  decay crim zn indus chas nox rm age dis rad tax ptratio black lstat
Min_param    1 0.0001    0  0     0    0   0  0   0   0   0   0       0     0     0
Max_param   25 0.9999    1  1     1    1   1  1   1   1   1   1       1     1     1


GA-PARSIMONY results: 
 Iterations                = 30 
 Best validation score = -3.076862 


Solution with the best validation score in the whole GA process = 
                    [,1]
fitnessVal -3.076862e+00
fitnessTst -2.937227e+00
complexity  1.100125e+07
size        1.789428e+01
decay       9.819532e-01
crim        1.000000e+00
zn          1.000000e+00
indus       1.000000e+00
chas        0.000000e+00
nox         1.000000e+00
rm          1.000000e+00
age         1.000000e+00
dis         1.000000e+00
rad         1.000000e+00
tax         1.000000e+00
ptratio     1.000000e+00
black       0.000000e+00
lstat       1.000000e+00


Results of the best individual at the last generation = 
 Best indiv's validat.cost = -3.084126 
 Best indiv's testing cost = -3.025289 
 Best indiv's complexity   = 11001231 
 Elapsed time in minutes   = 33.92762 


BEST SOLUTION = 
                    [,1]
fitnessVal -3.084126e+00
fitnessTst -3.025289e+00
complexity  1.100123e+07
size        1.789428e+01
decay       9.827273e-01
crim        1.000000e+00
zn          1.000000e+00
indus       1.000000e+00
chas        0.000000e+00
nox         1.000000e+00
rm          1.000000e+00
age         1.000000e+00
dis         1.000000e+00
rad         1.000000e+00
tax         1.000000e+00
ptratio     1.000000e+00
black       0.000000e+00
lstat       1.000000e+00
```

Plot GA evolution.


```{r fig.cap = "GA-PARSIMONY evolution", echo=FALSE}
# Plot GA evolution ('keep_history' must be TRUE)
elitists <- plot(GAparsimony_model, window=FALSE, general_cex=0.6, pos_cost_num=-1, pos_feat_num=-1, digits_plot=3)
```


![GA-PARSIMONY Evolution](https://github.com/jpison/GAparsimony/blob/master/images/regression.png)


GA-PARSIMONY evolution

Show the percentage of appearance of each feature in the elitists.

``` {.r}
# Percentage of appearance of each feature in the elitists
print(parsimony_importance(GAparsimony_model))
```

    ##         rm        age        dis        rad        tax      lstat 
    ## 100.000000 100.000000 100.000000 100.000000 100.000000 100.000000 
    ##        nox    ptratio      indus         zn       crim      black 
    ##  99.583333  99.583333  99.166667  98.333333  97.083333  25.416667 
    ##       chas 
    ##   5.416667

