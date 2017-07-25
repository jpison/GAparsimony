\name{parsimony_Selection}
\alias{parsimony_Selection}
%
\alias{parsimony_lrSelection}
\alias{parsimony_nlrSelection}

\title{Selection operators in GA-PARSIMONY}

\description{Functions implementing selection genetic operator in GA-PARSIMONY after \link{parsimony_rerank} process. Linear-rank or Nonlinear-rank selection (Michalewicz (1996)).}

\usage{
parsimony_lrSelection(object, r = 2/(object@popSize*(object@popSize-1)), q = 2/object@popSize, ...)
parsimony_nlrSelection(object, q = 0.25, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{An object of class \code{"ga_parsimony"}, usually resulting from a call to function \code{\link{ga_parsimony}}.}
  \item{r}{A tuning parameter for the specific selection operator.}
  \item{q}{A tuning parameter for the specific selection operator.}
  \item{\dots}{Further arguments passed to or from other methods.}
}

%\details{}

\value{
Return a list with four elements:

\item{population}{a matrix of dimension \code{object@popSize} times the number of decision variables containing the selected individuals or strings;}
\item{fitnessval}{a vector of length \code{object@popSize} containing the fitness validation values for the selected individuals;}
\item{fitnesstst}{a vector of length \code{object@popSize} containing the fitness with the test database (if it was supplied), for the selected individuals;}
\item{complexity}{a vector of length \code{object@popSize} containing the model complexity for the selected individuals.}
}

%\references{}
\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{http://www.mineriadatos.com}}
%\note{}

\seealso{\code{\link{ga_parsimony}}}

%\examples{}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
