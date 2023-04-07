\name{parsimony_Population}
\alias{parsimony_Population}
\alias{parsimony_population}

\title{Population initialization in GA-PARSIMONY with a combined chromosome of model parameters and selected features}

\description{Functions for creating an initial population to be used in the GA-PARSIMONY process.}

\usage{
parsimony_population(object, type_ini_pop="randomLHS", ...)
}

\arguments{
  \item{object}{An object of class \code{"ga_parsimony"}, usually resulting from a call to function \code{\link{ga_parsimony}}.}
  \item{type_ini_pop}{How to create the initial population. 'random' optiom initialize a random population between the predefined ranges. Values 'randomLHS', 'geneticLHS', 'improvedLHS', 'maximinLHS' & 'optimumLHS' corresponds with several methods of the Latin Hypercube Sampling (see 'lhs' package for more details).}
  \item{\dots}{Further arguments passed to or from other methods.}
}
  

\details{
	\code{parsimony_population} generates a random population of \code{object@popSize} individuals. For each individual a random chromosome is generated with \code{object@nParams} real values in the range [\code{object@min_param}, \code{object@max_param}] plus \code{object@nFeatures} random binary values for feature selection.
'random' or Latin Hypercube Sampling can be used to create a efficient spread initial population.
}

\value{Return a matrix of dimension \code{object@popSize} rows and \code{object@nParams}+\code{object@nFeatures} columns.}

%\references{}

\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{https://edmans.webs.com/}}

%\note{}

\seealso{\code{\link{ga_parsimony}}}

%\examples{}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
