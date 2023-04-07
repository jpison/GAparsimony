\name{parsimony_rerank}
\alias{parsimony_rerank}

\title{Function for reranking by complexity in parsimonious model selection process}

\description{Promotes models with similar fitness but lower complexity to top positions.}

\usage{
parsimony_rerank(object, verbose=FALSE, ...)
}

\arguments{
  \item{object}{An object of class \code{"ga_parsimony"} resulting from a call to function \code{\link{ga_parsimony}} with \code{keep_history} parameter set to TRUE.}
  \item{verbose}{If it is TRUE shows additional information.}
  \item{\dots}{Further arguments passed to or from other methods.}
}
  

\details{
	This method corresponds with the second step of parsimonious model selection (PMS) procedure. PMS works in the following way: in each GA generation, best solutions are first sorted by their cost, J. 
	Then, in a second step, individuals with less complexity are moved to the top positions when the absolute difference of their J is lower than a \code{object@rerank_error} threshold value. 
	Therefore, the selection of less complex solutions among those with similar accuracy promotes the evolution of robust solutions with better generalization capabilities.
}

\value{Return a vector with the new position of the individuals.}

%\references{}

\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{https://edmans.webs.com/}}

%\note{}

\seealso{\code{\link{ga_parsimony}}}

\examples{
library(GAparsimony)
object <- new("ga_parsimony", 
              rerank_error=0.2,
              best_score = 2.0,
              popSize = 4, 
              fitnessval = c(2.0, 1.9, 1.1, 1.0), 
              complexity=c(2,1,2,1))

pop_ini <- data.frame(fitnessval=object@fitnessval, 
                      complexity=object@complexity)
print("INITIAL POPULATION:")
print(pop_ini)

print("POPULATION ORDERED BY COMPLEXITY")
print(paste0("WHEN abs(diff(fitnessval)) < ",
             object@rerank_error,":"))
pop_ini[parsimony_rerank(object),]
}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
