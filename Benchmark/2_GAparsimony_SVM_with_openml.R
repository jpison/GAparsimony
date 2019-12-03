library(MASS)
library(Metrics)

library(caret)
library(GAparsimony)
library(data.table)


# Function to evaluate each SVM individual
# ----------------------------------------
fitness_SVM <- function(chromosome, ...)
{
  # First two values in chromosome are 'C' & 'sigma' of 'svmRadial' method
  tuneGrid <- data.frame(C=chromosome[1],sigma=chromosome[2])
  
  # Next values of chromosome are the selected features (TRUE if > 0.50)
  selec_feat <- chromosome[3:length(chromosome)]>0.50
  
  # Return -Inf if there is not selected features
  if (sum(selec_feat)<1) return(c(logloss_val=-Inf,logloss_tst=-Inf,complexity=Inf))
  
  # Extract features from the original DB plus response (last column)
  data_train_model <- data_train[,c(selec_feat,TRUE)]
  data_test_model <- data_test[,c(selec_feat,TRUE)]
  colnames(data_train_model)[ncol(data_train_model)] = 'class'
  colnames(data_test_model)[ncol(data_test_model)] = 'class'
  
  # How to validate each individual
  # 'repeats' could be increased to obtain a more robust validation metric. Also,
  # 'number' of folds could be adjusted to improve the measure.
  train_control <- caret::trainControl(method = "repeatedcv",number = 5,repeats = 6, classProbs=TRUE, summaryFunction=mnLogLoss, allowParallel = FALSE)
  
  # train the model
  set.seed(1234)
  model <- train(class ~ ., data=data_train_model, trControl=train_control, metric="logLoss", method="svmRadial", tuneGrid=tuneGrid, verbose=F)
  
  # Extract kappa statistics (the repeated k-fold CV and the kappa with the test DB)
  logloss_val <- model$results$logLoss
  first_level = levels(data_test_model[,ncol(data_test_model)])[1]
  logloss_tst <- logLoss(as.numeric(data_test_model[,ncol(data_test_model)]==first_level),
                         pred=predict(model, data_test_model,type="prob")[,first_level])
  # Obtain Complexity = Num_Features*1E6+Number of support vectors
  complexity <- sum(selec_feat)*1E6+model$finalModel@nSV 
  
  # Return(validation score, testing score, model_complexity)
  vect_errors <- c(logloss_val=-logloss_val,logloss_tst=-logloss_tst,complexity=complexity)
  return(vect_errors)
}


# Start search of the best parsimonious model for each dataset
df = read.csv("bases_datos.csv")
dir.create('results')
for (numrow in 1:nrow(df))
{
  db_name = df[numrow,'name_df']
  print(db_name)
  database = read.csv(paste0('data/',db_name,'.csv'))
 
  # Imputation of NAs
  if (sum(is.na(database))>0)
    {
      colsremove = NULL
      print(sum(is.na(database)))
      # Fill with mode
      for (numcol in 1:ncol(database))
      {
        if (sum(is.na(database[,numcol]))==nrow(database)) colsremove = c(colsremove, numcol) else
          if (length(unique(database[,numcol]))<2) colsremove = c(colsremove, numcol) else
            if (sum(is.na(database[,numcol]))>0 && length(unique(database[,numcol]))>1) database[is.na(database[,numcol]),numcol] = names(sort(table(database[,numcol]),decreasing=TRUE))[1]
      }
      # Remove cols with all NA 
      if (length(colsremove)>0) database = database[,-colsremove]
      print(sum(is.na(database)))
    }

  set.seed(1234)
  train_index = createDataPartition(database[,ncol(database)], p=0.70, list=FALSE, times=10)
  # Ten runs with GA parsimony
  for (n_iter in 1:10)
  {
    data_train <- database[train_index[,n_iter],]
    data_test <- database[-train_index[,n_iter],]
    print(dim(data_train))
    print(dim(data_test))
    print("###########################################")
    print(n_iter)
    print("###########################################")
    print(df[numrow,])
    print("###########################################")
    print("###########################################")
    GAparsimony_model <- ga_parsimony(fitness=fitness_SVM,
                                      min_param=c(0.0001, 0.0001),
                                      max_param=c(9.9999, 0.9999),
                                      names_param=c("C","sigma"),
                                      nFeatures=ncol(data_train)-1,
                                      names_features=colnames(data_train)[-ncol(data_train)],
                                      keep_history = TRUE,
                                      rerank_error = 0.0001,
                                      popSize = 40,
                                      maxiter = 60, early_stop=10,
                                      feat_thres=0.90, # Perc selected features in first generation
                                      feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                      parallel = TRUE, seed_ini = 1234)
    save_name = paste0("results/GAparsimony_",db_name,"_iter_",n_iter,".RData")
    print(paste0("Saving ", save_name))
    save(GAparsimony_model,file=save_name) 
    gc()
  }
}