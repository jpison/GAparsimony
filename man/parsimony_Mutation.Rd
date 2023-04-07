\name{parsimony_Mutation}
\alias{parsimony_Mutation}
%
\alias{parsimony_mutation}

\title{Mutation operators in GA-PARSIMONY}

\description{Functions implementing mutation genetic operator for GA-PARSIMONY. Method mutes a \code{object@pmutation} percentage of them. If the value corresponds to a model parameter, algorithm uses uniform random mutation. For binary select features, method sets to one if the random value between [0,1] is lower or equal to \code{object@feat_mut_thres}.}

\usage{
parsimony_mutation(object, \dots)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
 \item{object}{An object of class \code{"ga_parsimony"}, usually resulting from a call to function \code{\link{ga_parsimony}}.}
 \item{\dots}{Further arguments passed to or from other methods.}
}

%\details{}
  
\value{Return \code{object} with the population muted.}

%\references{}

\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{https://edmans.webs.com/}}

\seealso{\code{\link{ga_parsimony}}}

%\note{}

%\seealso{\code{\link{ga_parsimony}}}

%\examples{}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
