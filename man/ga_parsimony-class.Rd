\name{ga_parsimony-class}
\docType{class}
\alias{ga_parsimony-class}

\title{Class "ga_parsimony"}

\description{An S4 class for searching parsimonious models by feature selection and parameter tuning with genetic algorithms.}

\section{Objects from the Class}{
Objects can be created by calls to the \code{\link{ga_parsimony}} function.
}

\section{Slots}{
  \describe{
    \item{\code{call}}{an object of class \code{"call"} representing the matched call;}
    \item{\code{min_param}}{a vector of length equal to the model parameters providing the minimum of the search space;}
    \item{\code{max_param}}{a vector of length equal to the model parameters providing the maximum of the search space;}
    \item{\code{nParams}}{a value specifying the number of model parameter to be tuned;}
	\item{\code{feat_thres}}{proportion of selected features in the initial population. It is recommended a high percentage of selected features for the first generations;}
	\item{\code{feat_mut_thres}}{threshold to consider a random number between 0 and 1 is considered one if a value of the parameters-chromosome is muted. Default value is set to 0.5;}
	\item{\code{not_muted}}{number of the best elitists that are not muted. Default value is set to 3;}
	\item{\code{rerank_error}}{when a value distinct to zero is provided a second reranking process according to the model complexities is called by 'parsimonyReRank' function. Its primary objective is to select individuals with high validation cost while maintaining the robustness of a parsimonious model. This function switches the position of two models if the first one is more complex than the latter and no significant difference is found between their fitness values in terms of cost.  Therefore, if the absolute difference between the validation costs are lower than 'rerank_error' they are considered similar. Default value=0.01;}
	\item{\code{nFeatures}}{a value specifying the number of maximum input features;}
	\item{\code{names_param}}{a vector with the name of the model parameters;}
	\item{\code{names_features}}{a vector with the name of the input features;}
	\item{\code{popSize}}{the population size;}
    \item{\code{iter}}{the actual (or final) iteration of GA search;}
	\item{\code{iter_start_rerank}}{iteration when ReRanking process is actived. Default=0. Sometimes is useful not to use ReRanking process in the first generations;}
    \item{\code{early_stop}}{the number of consecutive generations without any improvement in the best fitness value before the GA is stopped;}
    \item{\code{maxiter}}{the maximum number of iterations to run before the GA search is halted;}
	\item{\code{minutes_gen}}{elapsed time of this generation (in minutes);}
    \item{\code{minutes_total}}{total elapsed time (in minutes);}
    \item{\code{suggestions}}{a matrix of user provided solutions and included in the initial population;}
    \item{\code{population}}{the current (or final) population;}
    \item{\code{elitism}}{the number of best fitness individuals to survive at each generation;}
    \item{\code{pcrossover}}{the crossover probability;}
    \item{\code{pmutation}}{the mutation probability;}
    \item{\code{fitnessval}}{the values of validation cost for the current (or final) population;}
	\item{\code{fitnesstst}}{the values of testing cost for the current (or final) population;}
	\item{\code{complexity}}{the values of model complexities for the current (or final) population;}
    \item{\code{summary}}{a matrix of summary statistics for fitness values at each iteration (along the rows);}
    \item{\code{bestSolList}}{a list with the best solution of all iterations;}
	\item{\code{bestfitnessVal}}{the validation cost of the best solution at the last iteration;}
	\item{\code{bestfitnessTst}}{the testing cost of the best solution at the last iteration;}
	\item{\code{bestcomplexity}}{the model complexity of the best solution at the last iteration;}
	\item{\code{bestsolution}}{the best solution at the last iteration;}
	\item{\code{history}}{a list with the population of all iterations;}
  }
}

\author{Francisco Javier Martinez-de-Pison. \email{fjmartin@unirioja.es}. EDMANS Group. http://www.mineriadatos.com}

\keyword{classes}
\seealso{For examples of usage see \code{\link{ga_parsimony}}.}

%\examples{}

\keyword{classes}
\keyword{optimize}

