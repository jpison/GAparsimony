##############################################################################
#                                                                            #
#                        GA-PARSIMONY in R                                   #
# Author: Francisco Javier Martinez de Pison. fjmartin@unirioja.es           #
#         EDMANS Group http://www.mineriadatos.com                           #
#                                                                            #
# Note: This package is an adaptation of the excellent GA package in R (by   #
# @Luca Scrucca) for searching parsimonious solution by optimizing feature   #
# selection, parameter tuning & parsimonious model selection.                #
#                                                                            #
##############################################################################


ga_parsimony <- function (fitness, ..., 
                          min_param, max_param, nFeatures,  
                          names_param=NULL, names_features=NULL,
                          iter_ini=0, object=NULL,
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
                          early_stop = maxiter, maxFitness = Inf, suggestions = NULL, 
                          parallel = FALSE,
                          monitor = if (interactive()) parsimony_monitor else FALSE, 
                          seed_ini = NULL, verbose=FALSE) 
{
  call <- match.call()
  
  # Check parameters
  # ----------------
  if (!is.function(population)) population <- get(population)
  if (!is.function(selection)) selection <- get(selection)
  if (!is.function(crossover)) crossover <- get(crossover)
  if (!is.function(mutation)) mutation <- get(mutation)
  if (missing(fitness)) stop("A fitness function must be provided!!!")
  if (!is.function(fitness)) stop("A fitness function must be provided!!!")
  if (popSize < 10) warning("The population size is less than 10!!!")
  if (maxiter < 1)  stop("The maximum number of iterations must be at least 1!!!")
  if (elitism > popSize) stop("The elitism cannot be larger that population size.")
  if (pcrossover < 0 | pcrossover > 1) stop("Probability of crossover must be between 0 and 1!!!")
  if (pmutation < 0 | pmutation > 1) stop("Probability of mutation must be between 0 and 1!!!")
  if (missing(min_param) & missing(max_param)) stop("A min and max range of values must be provided!!!")
  if (length(min_param)!=length(max_param)) stop("min_param and max_param must have the same length!!!")
  if (missing(nFeatures)) stop("Number of features (nFeatures) must be provided!!!")
  
  # nvars=chromosome length
  # -----------------------
  nParams <- length(as.vector(min_param))
  min_param <- c(as.vector(min_param),rep(0,nFeatures))
  max_param <- c(as.vector(max_param),rep(1,nFeatures))
  nvars <- nParams+nFeatures
  
  # Set monitor function
  # --------------------
  if (is.logical(monitor)) {if (monitor) monitor <- parsimony_monitor}
  if (is.null(monitor)) monitor <- FALSE
  
  # Initialize parallel computing
  # ----------------------
  parallel <- if (is.logical(parallel)) {if (parallel) startParallel(parallel) else FALSE} else {startParallel(parallel)}
  on.exit(if (parallel) parallel::stopCluster(attr(parallel,"cluster")))
  # define operator to use depending on parallel being TRUE or FALSE
  `%DO%` <- if(parallel) `%dopar%` else `%do%`
  
 
    # Get suggestions
    # ---------------
    if (is.null(suggestions))
    {
      suggestions <- matrix(nrow = 0, ncol = nvars)
    } else
    {
      if (is.vector(suggestions))
      {
        if (nvars > 1) suggestions <- matrix(suggestions, nrow = 1) else suggestions <- matrix(suggestions, ncol = 1)
      } else 
      {
        suggestions <- as.matrix(suggestions)
      }
      if (nvars != ncol(suggestions)) stop("Provided suggestions (ncol) matrix do not match number of variables of the problem!")
    }
  
  
  
  # Initial settings
  # ----------------
  i. <- NULL
  if (!is.null(seed_ini)) set.seed(seed_ini) else set.seed(1234)
  fitnessSummary <- matrix(as.double(NA), nrow = maxiter, ncol = 6*3)
  colnames(fitnessSummary) <- paste0(rep(c("max","mean","q3","median","q1","min"),3),rep(c("val","tst","complex"),each=6))
  bestSolList <- vector(mode = "list", length = maxiter)
  FitnessVal_vect <- rep(NA, popSize)
  FitnessTst_vect <- rep(NA, popSize)
  Complexity_vect <- rep(NA, popSize)
 
  
  # Initialize 'object'
  # -------------------
  object <- new("ga_parsimony", call = call, 
                min_param = min_param, max_param = max_param,
                nParams = nParams, feat_thres=feat_thres, 
                feat_mut_thres=feat_mut_thres, not_muted=not_muted, 
                rerank_error=rerank_error, iter_start_rerank=iter_start_rerank,
                nFeatures=nFeatures, 
                names_param = if (is.null(names_param)) character() else names_param,
                names_features = if (is.null(names_features)) character() else names_features, 
                popSize = popSize, iter = 0, early_stop = early_stop, maxiter = maxiter, 
                suggestions = suggestions, population = matrix(), elitism = elitism, 
                pcrossover = pcrossover, minutes_total=0,
                history = vector(mode = "list",length = maxiter),
                pmutation = if (is.numeric(pmutation)) pmutation else NA, 
                fitnessval = FitnessVal_vect, fitnesstst=FitnessTst_vect, complexity=Complexity_vect,
                summary = fitnessSummary, bestSolList = bestSolList)

  
  # First population
  # ----------------
  Pop <- matrix(as.double(NA), nrow = popSize, ncol = nvars)
  ng <- min(nrow(suggestions), popSize)
  if (ng > 0) Pop[1:ng, ] <- suggestions
  
  if (popSize > ng) Pop[(ng + 1):popSize, ] <- population(object,type_ini_pop=type_ini_pop)[1:(popSize-ng),]
  object@population <- Pop
  if (verbose)
  {
    print("Step 0. Initial population")
    print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
    readline(prompt="Press [enter] to continue")
  }
  
  
  # Main Loop
  # ---------
  for (iter in seq_len(maxiter))
    {
    tic <- Sys.time()
    
    object@iter <- iter
    if (!parallel) 
      {
      for (i in seq_len(popSize))
        {
        #if (i%%10==0) cat(".")
        # If There is nor fitnessval and sum of features>0
        if (is.na(FitnessVal_vect[i]) && sum(Pop[i,(1+object@nParams):nvars])>0)
          {
            fit <- fitness(Pop[i, ])
            FitnessVal_vect[i] <- fit[1]
            FitnessTst_vect[i] <- fit[2]
            Complexity_vect[i] <- fit[3]
            #print(fit)
          }
        }
      } else 
        {
          # varlist <- ls(envir = parent.frame(), all.names = TRUE)
          # varlist <- varlist[varlist != "..."]
          # pkgs <- .packages() #.packages = pkgs, .export=varlist,.verbose=TRUE
          Results_parallel <- foreach(i. = seq_len(popSize)) %DO% 
            {if (is.na(FitnessVal_vect[i.]) && sum(Pop[i.,(1+object@nParams):nvars])>0) fitness(Pop[i., ]) else c(FitnessVal_vect[i.],FitnessTst_vect[i.], Complexity_vect[i.])}
          # Extract results
          Results_parallel <- matrix(unlist(Results_parallel),object@popSize,ncol=3,byrow = TRUE)
          FitnessVal_vect <- Results_parallel[,1]
          FitnessTst_vect <- Results_parallel[,2]
          Complexity_vect <- Results_parallel[,3]
        }
    
    if (!is.null(seed_ini)) set.seed(seed_ini*iter) else set.seed(1234*iter)
    
    # Sort by the Fitness Value
    # ----------------------------
    ord <- order(FitnessVal_vect, decreasing = TRUE, na.last = TRUE)
    PopSorted <- Pop[ord, , drop = FALSE]
    FitnessValSorted <- FitnessVal_vect[ord]
    FitnessTstSorted <- FitnessTst_vect[ord]
    ComplexitySorted <- Complexity_vect[ord]
    
    object@population <- PopSorted
    object@fitnessval <- FitnessValSorted
    object@fitnesstst <- FitnessTstSorted
    object@complexity <- ComplexitySorted
    
    Pop <- PopSorted
    FitnessVal_vect <- FitnessValSorted
    FitnessTst_vect <- FitnessTstSorted
    Complexity_vect <- ComplexitySorted

    
    if (verbose)
      {
      print("Step 1. Fitness sorted")
      print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
      readline(prompt="Press [enter] to continue")
      }
    
    
    # Reorder models with ReRank function
    # -----------------------------------
    if (object@rerank_error!=0.0 && object@iter>=iter_start_rerank)
    {
      ord_rerank <- parsimony_rerank(object, verbose=verbose)
      PopSorted <- Pop[ord_rerank, ,drop=FALSE]
      FitnessValSorted <- FitnessVal_vect[ord_rerank]
      FitnessTstSorted <- FitnessTst_vect[ord_rerank]
      ComplexitySorted <- Complexity_vect[ord_rerank]
      
      object@population <- PopSorted
      object@fitnessval <- FitnessValSorted
      object@fitnesstst <- FitnessTstSorted
      object@complexity <- ComplexitySorted
      
      Pop <- PopSorted
      FitnessVal_vect <- FitnessValSorted
      FitnessTst_vect <- FitnessTstSorted
      Complexity_vect <- ComplexitySorted
      
      if (verbose)
        {
        print("Step 2. Fitness reranked")
        print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
        readline(prompt="Press [enter] to continue")
       }
      
    }

    # Keep results
    # ---------------
    fitnessSummary[iter, ] <- parsimony_summary(object)
    object@summary <- fitnessSummary
    
    # Keep Best Solution
    # ------------------
    object@bestfitnessVal <- object@fitnessval[1]
    object@bestfitnessTst <- object@fitnesstst[1]
    object@bestcomplexity <- object@complexity[1]
    object@bestsolution <- c(object@bestfitnessVal, object@bestfitnessTst, object@bestcomplexity,
                             as.vector(object@population[1, , drop = FALSE]))
    names(object@bestsolution) <- c("fitnessVal","fitnessTst","complexity",object@names_param,object@names_features)
    object@bestSolList[[iter]] <- object@bestsolution 
    
    # Keep elapsed time in minutes
    # ----------------------------
    tac <- Sys.time()
    object@minutes_gen <- as.double(difftime(tac,tic,units="mins"))
    object@minutes_total <- object@minutes_total+object@minutes_gen
    
    # Keep this generation into the History list
    # ------------------------------------------
    if (keep_history) object@history[[iter]] <- list(population=object@population, fitnessval=object@fitnessval, 
                                                     fitnesstst=object@fitnesstst, complexity=object@complexity)
    
    # Call to 'monitor' function
    # --------------------------
    if (is.function(monitor) && !verbose) monitor(object)  
    
    if (verbose)
    {
      print("Step 3. Fitness results")
      print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
      readline(prompt="Press [enter] to continue")
    }
    
    
    # Exit?
    # -----
    best_val_cost <- as.vector(na.omit(object@summary[,1]))
    if (object@bestfitnessVal >= maxFitness) break
    if (object@iter == maxiter) break
    if ((1+length(best_val_cost)-which.max(best_val_cost))>=early_stop) break
    
    
    # Selection Function
    # ------------------
    if (is.function(selection))
      {
      sel <- selection(object)
      Pop <- sel$population
      FitnessVal_vect <- sel$fitnessval
      FitnessTst_vect <- sel$fitnesstst
      Complexity_vect <- sel$complexity
      } else 
        {
          sel <- sample(1:popSize, size = popSize, replace = TRUE)
          Pop <- object@population[sel, ]
          FitnessVal_vect <- object@fitnessval[sel]
          FitnessTst_vect <- object@fitnesstst[sel]
          Complexity_vect <- object@complexity[sel]
        }
    object@population <- Pop
    object@fitnessval <- FitnessVal_vect
    object@fitnesstst <- FitnessTst_vect
    object@complexity <- Complexity_vect
    
    
    if (verbose)
      {
      print("Step 4. Selection")
      print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
      readline(prompt="Press [enter] to continue")
      }
    
    
    # CrossOver Function
    # ------------------
    if (is.function(crossover) & pcrossover > 0)
      {
      nmating <- floor(popSize/2)
      mating <- matrix(sample(1:(2 * nmating), size = (2 * nmating)), ncol = 2)
      for (i in seq_len(nmating))
        {
        if (pcrossover > runif(1))
          {
          parents <- mating[i, ]
          Crossover <- crossover(object, parents)
          Pop[parents, ] <- Crossover$children
          FitnessVal_vect[parents] <- Crossover$fitnessval
          FitnessTst_vect[parents] <- Crossover$fitnesstst
          Complexity_vect[parents] <- Crossover$complexity
          }
        }
      object@population <- Pop
      object@fitnessval <- FitnessVal_vect
      object@fitnesstst <- FitnessTst_vect
      object@complexity <- Complexity_vect
      
      if (verbose)
        {
        print("Step 5. CrossOver")
        print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
        readline(prompt="Press [enter] to continue")
        }
      
      }
    
    # New generation with elitists
    # ----------------------------
    if (elitism > 0)
      {
      Pop[1:elitism, ] <- PopSorted[1:elitism,]
      FitnessVal_vect[1:elitism] <- FitnessValSorted[1:elitism]
      FitnessTst_vect[1:elitism] <- FitnessTstSorted[1:elitism]
      Complexity_vect[1:elitism] <- ComplexitySorted[1:elitism]
      
      object@population <- Pop
      object@fitnessval <- FitnessVal_vect
      object@fitnesstst <- FitnessTst_vect
      object@complexity <- Complexity_vect
      
      if (verbose)
        {
        print("Step 6. With Elitists")
        print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
        readline(prompt="Press [enter] to continue")
        }
     }
    
    
    # Mutation function
    # -----------------
    if (is.function(mutation) & pmutation > 0)
    {
      object <- mutation(object)
      Pop <- object@population
      FitnessVal_vect <- object@fitnessval 
      FitnessTst_vect <- object@fitnesstst 
      Complexity_vect <- object@complexity
      
      if (verbose)
      {
        print("Step 7. Mutation")
        print(head(cbind(FitnessVal_vect, FitnessTst_vect, Complexity_vect, object@population),10))
        readline(prompt="Press [enter] to continue")
      }
    }
  } # End of loop
  
 return(object)
}

  
  
  
setClassUnion("numericOrNA", members = c("numeric", "logical"))

setClass(Class = "ga_parsimony", 
         representation(call = "language",
                        bestfitnessVal = "numeric",
                        bestfitnessTst = "numeric",
                        bestcomplexity = "numeric",
                        bestsolution = "numeric",
                        min_param = "numericOrNA", 
                        max_param = "numericOrNA", 
                        nParams = "numeric",
                        feat_thres = "numeric",
                        feat_mut_thres = "numeric",
                        not_muted = "numeric",
                        rerank_error = "numeric",
                        iter_start_rerank = "numeric",
                        nFeatures = "numeric",
                        names_param = "character",
                        names_features = "character",
                        popSize = "numeric",
                        iter = "numeric", 
                        early_stop = "numeric",
                        maxiter = "numeric",
                        minutes_gen = "numeric",
                        minutes_total = "numeric",
                        suggestions = "matrix",
                        population = "matrix",
                        elitism = "numeric", 
                        pcrossover = "numeric", 
                        pmutation = "numericOrNA",
                        fitnessval = "numericOrNA",
                        fitnesstst = "numericOrNA",
                        complexity = "numericOrNA",
                        summary = "matrix",
                        bestSolList = "list",
                        history = "list"
         ),
         package = "GAparsimony" 
) 

setMethod("print", "ga_parsimony", function(x, ...) str(x))

setMethod("show", "ga_parsimony",
          function(object)
          { cat("An object of class \"ga_parsimony\"\n")
            cat("\nCall:\n", deparse(object@call), "\n\n",sep="")
            cat("Available slots:\n")
            print(slotNames(object))
          }) 


summary.ga_parsimony <- function(object, ...)
{
  varnames <- c(object@names_param,object@names_features)
  domain <- rbind(object@min_param, object@max_param)
  rownames(domain) <- c("Min_param", "Max_param")
  colnames(domain) <- varnames

  suggestions <- NULL
  if(nrow(object@suggestions) > 0)
  {
    suggestions <- object@suggestions
    dimnames(suggestions) <- list(1:nrow(suggestions), varnames)
  }

  out <- list(popSize = object@popSize,
              maxiter = object@maxiter,
              early_stop = object@early_stop,
              rerank_error = object@rerank_error,
              elitism = object@elitism,
              nParams = object@nParams,
              nFeatures = object@nFeatures,
              pcrossover = object@pcrossover,
              pmutation = object@pmutation,
              feat_thres = object@feat_thres,
              feat_mut_thres = object@feat_mut_thres,
              not_muted = object@not_muted,
              domain = domain,
              suggestions = object@suggestions,
              iter = object@iter,
              bestfitnessVal = object@bestfitnessVal,
              bestfitnessTst = object@bestfitnessTst,
              bestcomplexity = object@bestcomplexity,
              minutes_total = object@minutes_total,
              bestsolution = object@bestsolution)
  class(out) <- "summary.ga_parsimony"
  return(out)
}

setMethod("summary", "ga_parsimony", summary.ga_parsimony)

print.summary.ga_parsimony <- function(x, digits = getOption("digits"), ...)
{
  dotargs <- list(...)
  if(is.null(dotargs$head)) dotargs$head <- 10
  if(is.null(dotargs$tail)) dotargs$tail <- 1
  if(is.null(dotargs$chead)) dotargs$chead <- 20
  if(is.null(dotargs$ctail)) dotargs$ctail <- 1

  cat("+------------------------------------+\n")
  cat("|             GA-PARSIMONY           |\n")
  cat("+------------------------------------+\n\n")
  cat("GA-PARSIMONY settings: \n")
  cat(paste(" Number of Parameters      = ", x$nParams, "\n"))
  cat(paste(" Number of Features        = ", x$nFeatures, "\n"))
  cat(paste(" Population size           = ", x$popSize, "\n"))
  cat(paste(" Maximum of generations    = ", x$maxiter, "\n"))
  cat(paste(" Number of early-stop gen. = ", x$early_stop, "\n"))
  cat(paste(" Elitism                   = ", x$elitism, "\n"))
  cat(paste(" Crossover probability     = ", format(x$pcrossover, digits = digits), "\n"))
  cat(paste(" Mutation probability      = ", format(x$pmutation, digits = digits), "\n"))
  cat(paste(" Max diff(error) to ReRank = ", format(x$rerank_error, digits = digits), "\n"))
  cat(paste(" Perc. of 1s in first popu.= ", format(x$feat_thres, digits = digits), "\n"))
  cat(paste(" Prob. to be 1 in mutation = ", format(x$feat_mut_thres, digits = digits), "\n"))
  
  cat(paste(" Search domain = \n"))
  print(x$domain, digits = digits)

  if(!is.null(x$suggestions) && nrow(x$suggestions)>0)
  { cat(paste("Suggestions =", "\n"))
    do.call(".printShortMatrix",
            c(list(x$suggestions, digits = digits),
              dotargs[c("head", "tail", "chead", "ctail")]))
    # print(x$suggestions, digits = digits, ...)
  }

  cat("\n\nGA-PARSIMONY results: \n")
  cat(paste(" Iterations                =", format(x$iter, digits = digits), "\n"))
  cat(paste(" Best indiv's validat.cost =", format(x$bestfitnessVal, digits = digits), "\n"))
  cat(paste(" Best indiv's testing cost =", format(x$bestfitnessTst, digits = digits), "\n"))
  cat(paste(" Best indiv's complexity   =", format(x$bestcomplexity, digits = digits), "\n"))
  cat(paste(" Elapsed time in minutes   =", format(x$minutes_total, digits = digits), "\n"))
  cat(paste("\n\nBEST SOLUTION = \n"))
  do.call(".printShortMatrix",c(list(x$bestsolution, digits = digits),head=length(x$bestsolution)))
  #print(as.vector(x$bestsolution)) #, digits = digits, ...)
  invisible()
}


# Plot a boxplot evolution of val cost, tst cost and complexity for the elitists
# ------------------------------------------------------------------------------
plot.ga_parsimony <- function(x, general_cex = 0.7, min_ylim=NULL, max_ylim=NULL, 
                              min_iter=NULL, max_iter=NULL, main_label="Boxplot cost evolution", 
                              iter_auto_ylim=3, steps=5, pos_cost_num=-3.1,  pos_feat_num=-1.7,
                              digits_plot=4, width_plot=12, height_plot=6, window=TRUE, ...)
{
  object <- x
  if (window) dev.new(1,width = width_plot, height = height_plot)
  if (length(object@history[[1]])<1) message("'object@history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
  if (is.null(min_iter)) min_iter <- 1
  if (is.null(max_iter)) max_iter <- object@iter
  
  nelitistm <- object@elitism
  mat_val <- NULL
  mat_tst <- NULL
  mat_complex <- NULL
  for (iter in min_iter:max_iter)
  {
    mat_val <- cbind(mat_val, object@history[[iter]]$fitnessval[1:nelitistm])
    mat_tst <- cbind(mat_tst, object@history[[iter]]$fitnesstst[1:nelitistm])
    mat_complex <- cbind(mat_complex, apply(object@history[[iter]]$population[1:nelitistm,(1+object@nParams):(object@nParams+object@nFeatures)],1,sum))
                                         
  }


  # Plot the range of num features and the nfeatures of the best individual
  # -----------------------------------------------------------------------
  plot((min_iter-1):max_iter, c(NA,mat_complex[1,]), lty="dashed", type="l", lwd=1.2,xaxt="n",yaxt="n",xlab="",ylab="", bty="n", axes=FALSE, 
       xlim=c(min_iter-1,max_iter),ylim=c(1,object@nFeatures))
  x_pol <- c(min_iter:max_iter,max_iter:min_iter, min_iter)
  max_pol <- apply(mat_complex,2,max)
  min_pol <- apply(mat_complex,2,min)
  y_pol <- c(max_pol, min_pol[length(min_pol):1],max_pol[1])
  polygon(x_pol,y_pol,col="gray90",border="gray80")
  lines(min_iter:max_iter, mat_complex[1,], lty="dashed")
  mtext("Number of features of best indiv.",side=4, line=-0.5, cex=general_cex*1.65)
  
  # Axis of side 4 (vertical right)
  # -----------------------------------------------------------------------
  axis_side4 <- seq(from=1,to=object@nFeatures,by=round(object@nFeatures/8));
  if (axis_side4[length(axis_side4)]!=object@nFeatures) axis_side4 <- c(axis_side4,object@nFeatures);
  if ((axis_side4[length(axis_side4)]-axis_side4[length(axis_side4)-1]) <= 2 && object@nFeatures>=20) axis_side4 <- axis_side4[-(length(axis_side4)-1)];
  axis(side=4, at=axis_side4, labels=F, tick=T,lwd.ticks=0.7,tcl=-0.25, xpd=TRUE, pos=max_iter,bty="n", cex=general_cex*2)
  mtext(axis_side4,side=4,line=pos_feat_num,at=axis_side4, cex=general_cex*1.5)
  
  
  
  
  # Boxplot evolution
  # ------------------
  par(new=TRUE)
  
  if (is.null(min_ylim)) if (!is.null(iter_auto_ylim) && iter_auto_ylim>=min_iter) min_ylim <- min(c(mat_val[,iter_auto_ylim],mat_tst[,iter_auto_ylim]),na.rm=TRUE) else min_ylim <- min(c(mat_val,mat_tst),na.rm=TRUE)
  if (is.null(max_ylim)) max_ylim <- max(c(mat_val,mat_tst),na.rm=TRUE)
  
  
  boxplot(mat_val,
          col="white", xlim=c(min_iter-1,max_iter), ylim=c(min_ylim,max_ylim), 
          xaxt = "n", xlab = "", ylab = "", border=T, axes=F,outline=F,
          medlwd=0.75, pars=list(yaxt="n",xaxt="n", xlab = "", ylab = "", 
                                 boxwex = 0.7, staplewex = 0.6, outwex = 0.5,lwd=0.75))
  boxplot(mat_tst, col="lightgray", 
          xlim=c(min_iter,(max_iter+1)),ylim=c(min_ylim,max_ylim), add=TRUE, border=T,outline=F,medlwd=0.75,
          pars=list(yaxt="n",xaxt="n", xlab = "", ylab = "",bty="n", axes=F,
                    boxwex = 0.7, staplewex = 0.6, outwex = 0.5,lwd=0.75))
  
  lines(mat_val[1,],col="black",lty=1,lwd=1.8)
  lines(mat_tst[1,],col="black",lty="dotdash",lwd=1.8)
  
  if (window) title(main=main_label)
  
  # Axis 
  # -----
  
  # Axis X
  pos_txt_gen <- seq(from=min_iter-1,to=max_iter,by=5)
  pos_txt_gen[1] <- 1
  axis(side=1,at=c(min_iter:max_iter), labels=F, tick=T, lwd.ticks=0.7,  tcl= -0.25, pos=min_ylim)
  axis(side=1,at=pos_txt_gen, labels=F, tick=T, lwd.ticks=0.7,   tcl= -0.5, pos=min_ylim)
  mtext("Number of generation", side=1, line=1, adj=0.5, cex=general_cex*1.65)
  mtext(paste("G.",pos_txt_gen,sep=""),side=1,line=-0.35,at=pos_txt_gen, cex=general_cex*1.5)
  
  # Axis Y
  as<-axis(side=2, at=round(seq(from=min_ylim,to=max_ylim,length.out=steps),3), labels=F, tick=T, 
           lwd.ticks=0.7, tcl= -0.20, xpd=TRUE, pos=1, bty="n", cex=general_cex*2)
  mtext("Cost", side=2, line=-2.0, adj=0.5,cex=general_cex*1.65)  
  mtext(round(as,3), side=2, line=pos_cost_num, at=as, cex=general_cex*1.5)

  # legend(x=pos_legend,max_ylim,c(paste0("Validation cost for best individual ('white' box plot of elitists)"),
  #                            paste0("Testing cost of best individual ('gray' box plot of elitists)"),
  #                            paste0("Number of features of best individual")),
  #        lty=c("solid","dotdash","dashed"), cex=general_cex*1.4,lwd=c(1.4,1.7,1.2),
  #        bty="n")
  mtext(paste0("Results for the best individual:  val.cost (white)=",round(mat_val[1,max_iter],digits_plot),
               ", tst.cost (gray)=",round(mat_tst[1,max_iter],digits_plot),
               ", complexity=",round(mat_complex[1,max_iter],digits_plot),side=3,line=0,cex=general_cex*1.2))
  return(list(mat_val=mat_val, mat_tst=mat_tst,  mat_complex=mat_complex))
}

setMethod("plot", "ga_parsimony", plot.ga_parsimony)





