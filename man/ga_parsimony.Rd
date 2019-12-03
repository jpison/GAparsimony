\name{ga_parsimony}
\alias{ga_parsimony}
\alias{show,ga_parsimony-method}
\alias{print,ga_parsimony-method}
\alias{print,ga_parsimony-method}
\alias{numericOrNA}

\title{GA-PARSIMONY}

\description{A GA-based optimization method for searching accurate parsimonious models by combining feature selection, model tuning, and parsimonious model selection (PMS).
 PMS procedure is based on separate cost and complexity evaluations. The best individuals are initially sorted by an error fitness function, and afterwards, models with similar costs are rearranged according to their model complexity so as to foster models of lesser complexity. The algorithm can be run sequentially or in parallel using an explicit master-slave parallelisation.}

\usage{
ga_parsimony(fitness, \dots, 
	min_param, max_param, nFeatures,  
	names_param=NULL, names_features=NULL,
	object=NULL, iter_ini=NULL,
	type_ini_pop="improvedLHS", 
	popSize = 50, pcrossover = 0.8,  maxiter = 40, 
	feat_thres=0.90, rerank_error = 0.0, iter_start_rerank = 0,
	pmutation = 0.10, feat_mut_thres=0.10, not_muted=3,
	elitism = base::max(1, round(popSize * 0.20)),
	population = parsimony_population,
	selection = parsimony_nlrSelection, 
	crossover = parsimony_crossover, 
	mutation = parsimony_mutation, 
	keep_history = FALSE,
	path_name_to_save_iter = NULL,	
	early_stop = maxiter, maxFitness = Inf, suggestions = NULL, 
	parallel = FALSE,
	monitor = if (interactive()) parsimony_monitor else FALSE, 
	seed_ini = NULL, verbose=FALSE)
}
	

\arguments{
\item{fitness}{the fitness function, any allowable R function which takes as input an individual \code{chromosome} which combines the model parameters to tune and the features to be selected. Fitness function returns a numerical vector with three values: 
\code{"validation_cost"}: a robust validation cost measure, J (RMSE, AUC, Logloss). For example, \code{n}-repeated cross validation (CV) RMSE for regression or \code{n}-repeated CV Logloss or AUC for classification;
\code{"testing_cost"}: testing cost obtained with a testing dataset not included in the validation process. This value is only for checking the generalization capability of the model. NA value can be supplied if there is not a testing data set;
\code{"model_complexity"}: model complexity. Can be the number of features, number of support vectors in SVM, sum of power of weights in ANNs, depth in Trees, generalised degrees of freedom (GDF), or another complexity metrics o combination of them.

Note: the chromosome is a concatenated real vector with the model parameters (parameters-chromosome) and the binary selection of the input features (features-chromosome). For example, a chromosome defined as c(10, 0.01, 0,1,1,0,1,0,0) could corresponds to a SVR model parameters C=10 & gamma=0.01, and a selection of three input features (second, third and fifth) from a dataset of 7 features (0110100).
}

\item{\dots}{additional arguments to be passed to the fitness function. This allows to write fitness functions that keep some variables fixed during the search.}
\item{min_param}{a vector of length equal to the model parameters providing the minimum of the search space.}
\item{max_param}{a vector of length equal to the model parameters providing the maximum of the search space.}
\item{nFeatures}{a value specifying the number of maximum input features.}
\item{names_param}{a vector with the name of the model parameters.}
\item{names_features}{a vector with the name of the input features.}
\item{object}{object of 'ga_parsimony' class to continue GA process. 'ga_parsimony@history' must be provided. Note: all GA settings are obtained from 'object' in order to continue the GA process.}
\item{iter_ini}{Iteration/generation of 'object@history' to be used when 'object' is provided. If 'iter_ini==NULL' uses the last iteration of 'object'.}
\item{type_ini_pop}{method to create the first population with 'parsimony_population' function. This function is called when iter_ini==0 and 'suggestions' are not provided. Methods='randomLHS','geneticLHS','improvedLHS','maximinLHS','optimumLHS','random'. First 5 methods correspond with several latine hypercube sampling.}
\item{popSize}{the population size.}
\item{pcrossover}{the probability of crossover between pairs of chromosomes. Typically this is a large value and by default is set to 0.8.}
\item{maxiter}{the maximum number of iterations to run before the GA process is halted.}
\item{feat_thres}{proportion of selected features in the initial population. It is recommended a high percentage of selected features for the first generations. By default is set to 0.90.}
\item{rerank_error}{when a value is provided, a second reranking process according to the model complexities is called by \link{parsimony_rerank} function. Its primary objective is to select individuals with high validation cost while maintaining the robustness of a parsimonious model. This function switches the position of two models if the first one is more complex than the latter and no significant difference is found between their fitness values in terms of cost.  Therefore, if the absolute difference between the validation costs are lower than 'rerank_error' they are considered similar. Default value=0.01}
\item{iter_start_rerank}{iteration when ReRanking process is actived. Default=0. Sometimes is useful not to use ReRanking process in the first generations.}
\item{pmutation}{the probability of mutation in a parent chromosome. Usually mutation occurs with a small probability. By default is set to 0.10.}
\item{feat_mut_thres}{probability of the muted 'features-chromosome' to be one. Default value is set to 0.10.}
\item{not_muted}{number of the best elitists that are not muted in each generation. Default value is set to 3.}
\item{elitism}{the number of best individuals to survive at each generation. By default the top 20\% individuals will survive at each iteration.}
\item{population}{an R function for randomly generating an initial population. See \code{\link{parsimony_population}} for available functions.}
\item{selection}{an R function performing selection, i.e. a function which generates a new population of individuals from the current population probabilistically according to individual fitness. See \code{\link{parsimony_nlrSelection}} for available functions.}
\item{crossover}{an R function performing crossover, i.e. a function which forms offsprings by combining part of the genetic information from their parents. See \code{\link{parsimony_crossover}} for available functions.}
\item{mutation}{an R function performing mutation, i.e. a function which randomly alters the values of some genes in a parent chromosome. See \code{\link{parsimony_mutation}} for available functions.}
\item{keep_history}{If it is TRUE keeps in the list \code{object@history} each generation. This parameter must set TRUE in order to use 'plot' method or 'parsimony_importance' function.}
\item{path_name_to_save_iter}{If it is not NULL save the 'ga_parsimony' object to the 'path_name_to_save_iter' file at the end of each iteration. Note: use extension '.RData', example 'object.RData'}
\item{early_stop}{the number of consecutive generations without any improvement in the best fitness value before the GA is stopped.}
\item{maxFitness}{the upper bound on the fitness function after that the GA search is interrupted. Default value is set to +Inf}
\item{suggestions}{a matrix of solutions strings to be included in the initial population. If provided the number of columns must match (object@nParams+object@nFeatures). Can be used a previous population, for example: 'ga_parsimony@history[[2]]$population'. }
\item{parallel}{a logical argument specifying if parallel computing should be used (\code{TRUE}) or not (\code{FALSE}, default) for evaluating the fitness function. This argument could also be used to specify the number of cores to employ; by default, this is taken from \code{\link[parallel]{detectCores}}. Finally, the functionality of parallelization depends on system OS: on Windows only 'snow' type functionality is available, while on Unix/Linux/Mac OSX both 'snow' and 'multicore' (default) functionalities are available.}
\item{monitor}{a logical or an R function which takes as input the current state of the \code{ga_parsimony-class} object and show the evolution of the search. By default, for interactive sessions, the function \code{\link{parsimony_monitor}} depending on whether or not is an RStudio session, prints the average and best fitness values at each iteration. If set to \code{plot} these information are plotted on a graphical device. Other functions can be written by the user and supplied as argument. In non interactive sessions, by default \code{monitor = FALSE} so any output is suppressed.}
\item{seed_ini}{an integer value containing the random number generator state. This argument can be used to replicate the results of a GA search. Note that if parallel computing is required, the \pkg{doRNG} package must be installed.}
\item{verbose}{if it is TRUE shows additional information for debugging.}
}


\details{
\pkg{GAparsimony} package is a new GA wrapper automatic procedure that efficiently generated prediction models with reduced complexity and adequate generalization capacity. 
\code{ga_parsimony} function is primarily based on combining feature selection and model parameter tuning with a second novel GA selection process (\code{parsimony_rerank} function), in order to achieve better overall parsimonious models. 
Unlike other GA methodologies that use a penalty parameter for combining loss and complexity measures into a unique fitness function, the main contribution of this package is that \code{ga_parsimony} selects the best models by considering cost and complexity separately. For this purpose, the ReRank algorithm rearranges individuals by their complexity when there is not a significant difference between their costs. Thus, less complex models with similar accuracy are promoted. Furthermore, because the penalty parameter is unnecessary, there is no consequent uncertainty associated with assigning a correct value beforehand. As a result, with GA-PARSIMONY, an automatic method for obtaining parsimonious models is finally made possible.
}
  
\value{Returns an object of class \link{ga_parsimony-class}. See \code{\link{ga_parsimony-class}} for a description of available slots information.}

\references{

Urraca R., Sodupe-Ortega E., Antonanzas E., Antonanzas-Torres F., Martinez-de-Pison, F.J. (2017). Evaluation of a novel GA-based methodology for model structure selection: The GA-PARSIMONY. Neurocomputing, Online July 2017. https://doi.org/10.1016/j.neucom.2016.08.154

Sanz-Garcia A., Fernandez-Ceniceros J., Antonanzas-Torres F., Pernia-Espinoza A.V., Martinez-de-Pison F.J. (2015).  GA-PARSIMONY: A GA-SVR approach with feature selection and parameter optimization to obtain parsimonious solutions for predicting temperature settings in a continuous annealing furnace. Applied Soft Computing 35, 23-38.

Fernandez-Ceniceros J., Sanz-Garcia A., Antonanzas-Torres F., Martinez-de-Pison F.J. (2015). A numerical-informational approach for characterising the ductile behaviour of the T-stub component. Part 2: Parsimonious soft-computing-based metamodel. Engineering Structures 82, 249-260.

Antonanzas-Torres F., Urraca R., Antonanzas J., Fernandez-Ceniceros J., Martinez-de-Pison F.J. (2015). Generation of daily global solar irradiation with support vector machines for regression. Energy Conversion and Management 96, 277-286.


}
  
\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{http://www.mineriadatos.com}}

%\note{}

\seealso{
	\code{\link{ga_parsimony-class}},
	\code{\link{summary.ga_parsimony}}, 
	\code{\link{plot.ga_parsimony}}, 
  
	\code{\link{parsimony_Population}},
	\code{\link{parsimony_Selection}},
	\code{\link{parsimony_Crossover}},
	\code{\link{parsimony_Mutation}},
	
	\code{\link{parsimony_importance}},
	\code{\link{parsimony_rerank}}.
}

\examples{

#################################
### Example 1: Classification ###
#################################

# This a toy example that shows how to search, for the *iris* database, 
# a parsimony classification NNET model with 'GAparsimony' 
# and 'caret' packages. Validation errors and iterations have been
# reduced to speedup the process

library(GAparsimony)
# Training and testing Datasets
library(caret)

data(iris)
# Z-score of input features
iris_esc <- data.frame(scale(iris[,1:4]),Species=iris[,5])

# Define an 70%/30% train_val/test split of the dataset
set.seed(1234)
inTraining <- createDataPartition(iris_esc$Species, p=.70, list=FALSE)
data_train <- iris_esc[ inTraining,]
data_test  <- iris_esc[-inTraining,]

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

	# Validate each individual with only a 2-CV
	# Yo obtain a robust validation measure
	# use 'repeatedcv' with more folds and times
	# (see 2nd and 3rd examples...)
	train_control <- trainControl(method = "cv",number = 5)

	# train the model
	set.seed(1234)
	model <- train(Species ~ ., data=data_train_model, 
					trControl=train_control, 
					method="svmRadial", metric="Kappa", 
					tuneGrid=tuneGrid, verbose=FALSE)
				   
	# Extract validation and test accuracy
	accuracy_val <- model$results$Accuracy
	accuracy_test <- postResample(pred=predict(model, data_test_model),
						obs=data_test_model[,ncol(data_test_model)])[2]
	
	# Obtain Complexity = Num_Features*1E6+Number of support vectors
	complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 

	# Return(validation accuracy, testing accuracy, model_complexity)
	vect_errors <- c(accuracy_val=accuracy_val,
					accuracy_test=accuracy_test,complexity=complexity)
	return(vect_errors)
}


# ---------------------------------------------------------------------------------
# Search the best parsimonious model with GA-PARSIMONY by using Feature Selection,
# Parameter Tuning and Parsimonious Model Selection
# ---------------------------------------------------------------------------------
library(GAparsimony)

# Ranges of size and decay
min_param <- c(0.0001, 0.00001)
max_param <- c(0.9999, 0.99999)
names_param <- c("C","sigma")
  
# ga_parsimony can be executed with a different set of 'rerank_error' values
rerank_error <- 0.001

\donttest{
	
GAparsimony_model <- ga_parsimony(fitness=fitness_SVM,
                                  min_param=min_param, 
                                  max_param=max_param,
                                  names_param=names_param,
                                  nFeatures=ncol(data_train)-1,
                                  names_features=colnames(data_train)[-ncol(data_train)],
                                  keep_history = TRUE, 
                                  rerank_error = rerank_error,
                                  popSize = 20,
                                  maxiter = 20,
                                  early_stop=7,
                                  feat_thres=0.90,# Perc selec features in first iter
                                  feat_mut_thres=0.10,# Prob. feature to be 1 in mutation
                                  not_muted=1,
                                  parallel = FALSE, # speedup with 'n' cores or all with TRUE
                                  seed_ini = 1234)


print(paste0("Best Parsimonious SVM with C=",
		   GAparsimony_model@bestsolution['C'],
		   " sigma=",
		   GAparsimony_model@bestsolution['sigma'],
		   " -> ",
		   " AccuracyVal=",
		   round(GAparsimony_model@bestsolution['fitnessVal'],6),
		   " AccuracyTest=",
		   round(GAparsimony_model@bestsolution['fitnessTst'],6),
		   " Num Features=",
		   round(GAparsimony_model@bestsolution['complexity']/1E6,0),
		   " Complexity=",
		   round(GAparsimony_model@bestsolution['complexity'],2)))
  
print(summary(GAparsimony_model))
  
print(parsimony_importance(GAparsimony_model))


  
  #################################
  ### Example 2: Classification ###
  #################################
  
  #This example shows how to search, for the *Sonar* database, 
  #a parsimony classification SVM model with 'GAparsimony' and 'caret' packages.
  
  # Training and testing Datasets
  library(caret)
  library(GAparsimony)
  library(mlbench)
  data(Sonar)
  
  set.seed(1234)
  inTraining <- createDataPartition(Sonar$Class, p=.80, list=FALSE)
  data_train <- Sonar[ inTraining,]
  data_test  <- Sonar[-inTraining,]
  
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
                   method="svmRadial", metric="Kappa", 
				   tuneGrid=tuneGrid, verbose=FALSE)
    
    # Extract kappa statistics (repeated k-fold CV and testing kappa)
    kappa_val <- model$results$Kappa
    kappa_test <- postResample(pred=predict(model, data_test_model),
                    obs=data_test_model[,ncol(data_test_model)])[2]
    # Obtain Complexity = Num_Features*1E6+Number of support vectors
    complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 
    
    # Return(validation error, testing error, model_complexity)
    vect_errors <- c(kappa_val=kappa_val,kappa_test=kappa_test,complexity=complexity)
    return(vect_errors)
  }
  
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
  
  
  # 40 individuals per population, 100 max generations with an early stopping 
  # of 10 generations (CAUTION! 7.34 minutes with 8 cores)!!!!!
  GAparsimony_model <- ga_parsimony(fitness=fitness_SVM,
                                    min_param=min_param, 
                                    max_param=max_param,
                                    names_param=names_param,
                                    nFeatures=ncol(data_train)-1,
                                    names_features=colnames(data_train)[-ncol(data_train)],
                                    keep_history = TRUE, 
                                    rerank_error = rerank_error,
                                    popSize = 40, 
                                    maxiter = 100, 
                                    early_stop=10,
                                    feat_thres=0.90,# Perc selec features in first iter
                                    feat_mut_thres=0.10,# Prob. feature to be 1 in mutation
                                    parallel = TRUE, seed_ini = 1234)
  
  print(paste0("Best Parsimonious SVM with C=",
               GAparsimony_model@bestsolution['C'],
               " sigma=",
               GAparsimony_model@bestsolution['sigma'],
               " -> ",
               " KappaVal=",
               round(GAparsimony_model@bestsolution['fitnessVal'],6),
               " KappaTst=",
               round(GAparsimony_model@bestsolution['fitnessTst'],6),
               " Num Features=",
               round(GAparsimony_model@bestsolution['complexity']/1E6,0),
               " Complexity=",
               round(GAparsimony_model@bestsolution['complexity'],2)))
  
  print(summary(GAparsimony_model))
  
  # Plot GA evolution ('keep_history' must be TRUE)
  elitists <- plot(GAparsimony_model)
  
  # Percentage of appearance of each feature in elitists
  print(parsimony_importance(GAparsimony_model))
  
  
  
  
  #############################
  ### Example 3: Regression ###
  #############################
  
  # This example shows how to search, for the *Boston* database, a parsimony regressor ANN 
  # model with 'GAparsimony' and 'caret' packages.
  
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
                   method="nnet", tuneGrid=tuneGrid, trace=FALSE, linout = 1)
    
    # Extract errors
    rmse_val <- model$results$RMSE
    rmse_test <- sqrt(mean((unlist(predict(model, newdata = data_test_model)) - 
                              data_test_model$medv)^2))
    # Obtain Complexity = Num_Features*1E6+sum(neural_weights^2)
    complexity <- sum(selec_feat)*1E6+sum(model$finalModel$wts*model$finalModel$wts)  
    
    # Return(-validation error, -testing error, model_complexity)
    # errors are negative because GA-PARSIMONY tries to maximize values
    vect_errors <- c(rmse_val=-rmse_val,rmse_test=-rmse_test,complexity=complexity)
    return(vect_errors)
  }
  
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
  
  # 40 individuals per population, 100 max generations with an early stopping 
  # of 10 generations (CAUTION! 33.89 minutes with 8 cores)!!!!!
  GAparsimony_model <- ga_parsimony(fitness=fitness_NNET,
                                    min_param=min_param, 
                                    max_param=max_param,
                                    names_param=names_param,
                                    nFeatures=ncol(data_train)-1,
                                    names_features=colnames(data_train)[-ncol(data_train)],
                                    keep_history = TRUE, 
                                    rerank_error = rerank_error,
                                    popSize = 40,
                                    maxiter = 100, # Change to 100
                                    early_stop=10,
                                    feat_thres=0.90,# Perc selec features in first iter
                                    feat_mut_thres=0.10,# Prob. feature to be 1 in mutation
                                    not_muted=2,
                                    parallel = TRUE, seed_ini = 1234)
  
  print(paste0("Best Parsimonious ANN with ",
               round(GAparsimony_model@bestsolution['size']),
               " hidden neurons and decay=",
               GAparsimony_model@bestsolution['decay'],
               " -> ",
               " RMSEVal=",
               round(-GAparsimony_model@bestsolution['fitnessVal'],6),
               " RMSETst=",
               round(-GAparsimony_model@bestsolution['fitnessTst'],6)))
  print(summary(GAparsimony_model))
  
  # Plot GA evolution ('keep_history' must be TRUE)
  elitists <- plot(GAparsimony_model)
  
  # Percentage of appearance of each feature in elitists
  print(parsimony_importance(GAparsimony_model))
}
		 
}

\keyword{optimize}
