\name{parsimony_crossover}
\alias{parsimony_Crossover}
%
\alias{parsimony_crossover}

\title{Crossover operators in GA-PARSIMONY}

\description{Functions implementing particular crossover genetic operator for GA-PARSIMONY. Method uses for model parameters Heuristic Blending and random swapping for binary selected features.}

\usage{
parsimony_crossover(object, parents, alpha=0.1, perc_to_swap=0.5, \dots)
}

\arguments{
  \item{object}{An object of class \code{"ga_parsimony"}, usually resulting from a call to function \code{\link{ga_parsimony}}.}
  \item{parents}{A two-rows matrix of values indexing the parents from the current population.}
  \item{alpha}{A tuning parameter for the Heuristic Blending outer bounds [Michalewicz, 1991]. Typical and default value is 0.1.}
  \item{perc_to_swap}{Percentage of features for swapping in the crossovering process.}
  \item{\dots}{Further arguments passed to or from other methods.}
}

%\details{}

\value{
Return a list with two elements:
\item{children}{Matrix of dimension 2 times the number of decision variables containing the generated offsprings;}
\item{fitnessval}{Vector of length 2 containing the fitness validation values for the offsprings. A value \code{NA} is returned if an offspring is different (which is usually the case) from the two parents.}
\item{fitnesstst}{Vector of length 2 containing the fitness with the test database (if it was supplied), for the offsprings. A value \code{NA} is returned if an offspring is different (which is usually the case) from the two parents.}
\item{complexity}{Vector of length 2 containing the model complexity for the offsprings. A value \code{NA} is returned if an offspring is different (which is usually the case) from the two parents.}
}

%\references{}
\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{https://edmans.webs.com/}}
%\note{}

\seealso{\code{\link{ga_parsimony}}}

%\examples{}

%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
