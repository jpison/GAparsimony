##############################################################################
#                                                                            #
#                         Parsimony GA operators                             #
#                                                                            #
##############################################################################


#########################################################
# parsimonyReRank: Function for reranking by complexity #
#########################################################
parsimony_rerank <- function(object, verbose=FALSE, ...)
{ 
  
  cost1 <- object@fitnessval
  cost1[is.na(cost1)] <- -Inf
  ord <- order(cost1, decreasing = TRUE)
  cost1 <- cost1[ord]
  complexity <- object@complexity
  complexity[is.na(complexity)] <- +Inf
  complexity <- complexity[ord]
  position <- seq_len(length(cost1))
  position <- position[ord]
  
  # start
  pos1 <- 1
  pos2 <- 2
  cambio <- FALSE
  #error_posic <- cost1[pos1]
  error_posic <- object@best_score
  
  while(pos1!=object@popSize)
  {
    # Obtaining errors
    if (pos2>object@popSize) {if (cambio) {pos2 <- pos1+1;cambio <- FALSE} else break}
    error_indiv2 <- cost1[pos2]
    
    # Compare error of first individual with error_posic. Is greater than threshold go to next point
    #      if ((Error.Indiv1-error_posic) > object@rerank_error) error_posic=Error.Indiv1
    
    error_dif <- abs(error_indiv2-error_posic)
    if (!is.finite(error_dif)) error_dif <- +Inf
    if (error_dif < object@rerank_error)
    {
      # If there is not difference between errors swap if Size2nd < SizeFirst
      size_indiv1 <- complexity[pos1]
      size_indiv2 <- complexity[pos2]
      if (size_indiv2<size_indiv1)
      {
        cambio <- TRUE
        
        swap_indiv <- cost1[pos1]
        cost1[pos1] <- cost1[pos2]
        cost1[pos2] <- swap_indiv
        
        swap_indiv <- complexity[pos1]
        complexity[pos1] <- complexity[pos2]
        complexity[pos2] <- swap_indiv
        
        swap_indiv <- position[pos1]
        position[pos1] <- position[pos2]
        position[pos2] <- swap_indiv
        
        if (verbose)
        {
          print(paste0("SWAP!!: pos1=",pos1,"(",size_indiv1,"),",
                       "pos2=",pos2,"(",size_indiv2,"),",
                       "error_dif=",error_dif))
          print("-----------------------------------------------------")
        }
      }
      pos2 <- pos2+1
    } else if(cambio) {cambio <- FALSE;pos2 <- pos1+1;} else {
      pos1 <- pos1+1;pos2 <- pos1+1;
      error_dif2 <- abs(cost1[pos1]-error_posic)
      if (!is.finite(error_dif2)) error_dif2 <- +Inf
      if (error_dif2>=object@rerank_error) {error_posic <- cost1[pos1]}}
  }
  return(position)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------



##########################################################################
# parsimony_importance: Feature Importance of elitists in the GA process #
##########################################################################
parsimony_importance <- function(object, verbose=FALSE, ...)
{
  if (length(object@history[[1]])<1) message("'object@history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
  min_iter <- 1
  max_iter <- object@iter
  
  nelitistm <- object@elitism
  features_hist <- NULL
  for (iter in min_iter:max_iter)
  {
    features_hist <- rbind(features_hist, object@history[[iter]]$population[1:nelitistm,-c(1:object@nParams)])
  }
  importance <- apply(features_hist,2,mean)
  names(importance) <- object@names_features
  imp_features <- 100*importance[order(importance,decreasing = T)]
  if (verbose)
  {
    names(importance) <- object@names_features
    cat("+--------------------------------------------+\n")
    cat("|                  GA-PARSIMONY              |\n")
    cat("+--------------------------------------------+\n\n")
    cat("Percentage of appearance of each feature in elitists: \n")
    print(imp_features)
  }
  return(imp_features)
}



################################################################
# parsimony_population: Function for creating first generation #
################################################################
parsimony_population <- function(object, type_ini_pop="randomLHS", ...)
{
  nvars <- object@nParams+object@nFeatures
  if (type_ini_pop=="randomLHS") population <- lhs::randomLHS(object@popSize,nvars)
  if (type_ini_pop=="geneticLHS") population <- lhs::geneticLHS(object@popSize,nvars)
  if (type_ini_pop=="improvedLHS") population <- lhs::improvedLHS(object@popSize,nvars)
  if (type_ini_pop=="maximinLHS") population <- lhs::maximinLHS(object@popSize,nvars)
  if (type_ini_pop=="optimumLHS") population <- lhs::optimumLHS(object@popSize,nvars)
  if (type_ini_pop=="random") population <- matrix(runif(object@popSize*nvars,object@popSize,nvars))
  
  # Scale matrix with the parameters range
  population <- sweep(population,2,(object@max_param-object@min_param),"*")
  population <- sweep(population,2,object@min_param,"+")
  # Convert features to binary 
  population[,(1+object@nParams):nvars] <- population[,(1+object@nParams):nvars]<=object@feat_thres
  return(population)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





#########################################
# Function for selecting in GAparsimony #
# Note: population has been sorted      #
#       with ReRank algorithm           #
#########################################

parsimony_lrSelection <- function(object, 
                            r = 2/(object@popSize*(object@popSize-1)), 
                            q = 2/object@popSize, ...)
{
# Linear-rank selection
# Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
  rank <- 1:object@popSize # population are sorted in GAparsimony
  prob <- q - (rank-1)*r
  sel <- sample(1:object@popSize, size = object@popSize, 
                prob = pmin(pmax(0, prob), 1, na.rm = TRUE),
                replace = TRUE)
  out <- list(population = object@population[sel,,drop=FALSE],
              fitnessval = object@fitnessval[sel],
              fitnesstst = object@fitnesstst[sel],
              complexity = object@complexity[sel])
  return(out)
}

parsimony_nlrSelection <- function(object, q = 0.25, ...)
{
# Nonlinear-rank selection
# Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
  rank <- 1:object@popSize # population are sorted
  prob <- q*(1-q)^(rank-1)
  sel <- sample(1:object@popSize, size = object@popSize, 
                prob = pmin(pmax(0, prob), 1, na.rm = TRUE),
                replace = TRUE)
  out <- list(population = object@population[sel,,drop=FALSE],
              fitnessval = object@fitnessval[sel],
              fitnesstst = object@fitnesstst[sel],
              complexity = object@complexity[sel])
  return(out)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





###########################
# Functions for crossover #
###########################

parsimony_crossover <- function(object, parents, alpha=0.1, perc_to_swap=0.5, ...)
{
  parents <- object@population[parents,,drop = FALSE]
  n <- ncol(parents)
  children <- parents
  pos_param <- 1:object@nParams
  pos_features <- (1+object@nParams):(object@nParams+object@nFeatures)
  
  # Heuristic Blending for parameters
  alpha <- 0.1
  Betas <- runif(object@nParams)*(1+2*alpha)-alpha
  children[1,pos_param] <- parents[1,pos_param]-Betas*parents[1,pos_param]+Betas*parents[2,pos_param]
  children[2,pos_param] <- parents[2,pos_param]-Betas*parents[2,pos_param]+Betas*parents[1,pos_param]
  
  # Random swapping for features
  swap_param <- runif(object@nFeatures)>=perc_to_swap
  if (sum(swap_param)>0)
  {
    features_parent1 <- as.vector(parents[1,pos_features])
    features_parent2 <- as.vector(parents[2,pos_features])
    pos_features <- pos_features[swap_param]
    children[1,pos_features] <- features_parent2[swap_param]
    children[2,pos_features] <- features_parent1[swap_param]
  }
  
  # correct params that are outside (min and max)
  thereis_min <- (children[1,] < object@min_param)
  children[1,thereis_min] <- object@min_param[thereis_min]
  thereis_min <- (children[2,] < object@min_param)
  children[2,thereis_min] <- object@min_param[thereis_min]
  
  thereis_max <- (children[1,] > object@max_param)
  children[1,thereis_max] <- object@max_param[thereis_max]
  thereis_max <- (children[2,] > object@max_param)
  children[2,thereis_max] <- object@max_param[thereis_max]
  
  
  out <- list(children = children, fitnessval = rep(NA,2), 
              fitnesstst = rep(NA,2), complexity = rep(NA,2))
  return(out)
}
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





##########################
# Functions for mutation #
##########################
parsimony_mutation <- function(object, ...)
{
  # Uniform random mutation (except first individual)
  nparam_to_mute <- round(object@pmutation*(object@nParams+object@nFeatures)*object@popSize)
  if (nparam_to_mute<1) nparam_to_mute=1
  
  for (item in seq(nparam_to_mute))
  {
    i <- sample((1+object@not_muted):object@popSize, size=1)
    j <- sample(1:(object@nParams+object@nFeatures), size = 1)
    object@population[i,j] <- runif(1, object@min_param[j], object@max_param[j])
    # If is a binary feature selection convert to binary
    if (j>=(1+object@nParams))  object@population[i,j] <- (object@population[i,j]<=object@feat_mut_thres)
    
    object@fitnessval[i] <- NA
    object@fitnesstst[i] <- NA
    object@complexity[i] <- NA
  }
  return(object)
}
# 
# parsimony_nraMutation <- function(object, parent, ...)
# {
# # Non uniform random mutation
#   mutate <- parent <- as.vector(object@population[parent,])
#   n <- length(parent)
#   g <- 1 - object@iter/object@maxiter # dempening factor
#   sa <- function(x) x*(1-runif(1)^g)
#   j <- sample(1:n, 1)
#   u <- runif(1)
#   if(u < 0.5)
#     { mutate[j] <- parent[j] - sa(parent[j] - object@max_param[j]) }
#   else
#   { mutate[j] <- parent[j] + sa(object@max_param[j] - parent[j]) }
#   # Convert features to binary 
#   mutate[(1+object@nParams):(object@nParams+object@nFeatures)] <- mutate[(1+object@nParams):(object@nParams+object@nFeatures)]>=object@feat_mut_thres
#   return(mutate)
# }
# 
# parsimony_rsMutation <- function(object, parent, ...)
# {
# # Random mutation around the solution
#   mutate <- parent <- as.vector(object@population[parent,])
#   dempeningFactor <- 1 - object@iter/object@maxiter
#   direction <- sample(c(-1,1),1)
#   value <- (object@max_param - object@min_param)*0.67
#   mutate <- parent + direction*value*dempeningFactor
#   outside <- (mutate < object@min_param | mutate > object@max_param)
#   for(j in which(outside))
#   { mutate[j] <- runif(1, object@min_param[j], object@max_param[j]) }
#   # Convert features to binary 
#   mutate[(1+object@nParams):(object@nParams+object@nFeatures)] <- mutate[(1+object@nParams):(object@nParams+object@nFeatures)]>=object@feat_mut_thres
#   return(mutate)
# }
# 
# # Power mutation(pow)
# #
# # a is the location parameter and b > 0 is the scaling parameter of a Laplace
# # distribution, which is generated as described in 
# # Krishnamoorthy K. (2006) Handbook of Statistical Distributions with 
# #   Applications, Chapman & Hall/CRC.
# #
# # For smaller values of b offsprings are likely to be produced nearer to 
# # parents, and for larger values of b offsprings are expected to be produced
# # far from parents.
# 
# # Deep et al. (2009) suggests to use pow = 10 for real-valued variables, and
# # pow = 4 for integer variables.
# #
# # References
# #
# # Deep K., Singh K.P., Kansal M.L., Mohan C. (2009) A real coded genetic
# #   algorithm for solving integer and mixed integer optimization problems.
# #   Applied Mathematics and Computation, 212(2), pp. 505-518.
# 
# parsimony_powMutation <- function(object, parent, pow = 4, ...)
# {
#   mutate <- parent <- as.vector(object@population[parent,])
#   n <- length(parent)
#   s <- runif(1)^pow
#   t <- (parent - object@min_param)/(object@max_param - parent)
#   r <- runif(n)
#   mutate <- parent + ifelse(r > t, 
#                             +s*(object@max_param - parent), 
#                             -s*(parent - object@min_param))
#   # Convert features to binary 
#   mutate[(1+object@nParams):(object@nParams+object@nFeatures)] <- mutate[(1+object@nParams):(object@nParams+object@nFeatures)]>=object@feat_mut_thres
#   return(mutate)
# }

