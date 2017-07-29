\name{parsimony_importance}
\alias{parsimony_importance}

\title{Percentage of appearance of each feature in elitist population}

\description{Shows the percentage of appearance of each feature in the whole GA-PARSIMONY process but only for the elitist-population.}

\usage{
parsimony_importance(object, verbose=FALSE, ...)
}

\arguments{
  \item{object}{An object of class \code{"ga_parsimony"} resulting from a call to function \code{\link{ga_parsimony}} with \code{keep_history} parameter set to TRUE.}
  \item{verbose}{If it is TRUE shows additional information.}
  \item{\dots}{Further arguments passed to or from other methods.}
}
  

\details{
	\code{parsimony_importance} extracts elistist population from all generations. Obtains the percentage of appearance of each feature in the all GA process. Return the features higher-ordered.
}

\value{Return a vector with the higher-ordered percentage of appearance of each feature in the elitist-population and in the whole GA process.}

%\references{}

\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{http://www.mineriadatos.com}}

%\note{}

\seealso{\code{\link{ga_parsimony}}}

%\examples{}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
