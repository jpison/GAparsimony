\name{summary.ga_parsimony-method}
\docType{methods}
\alias{summary,ga_parsimony-method}
\alias{summary.ga_parsimony}
\alias{print.summary.ga_parsimony}

\title{Summary for GA-PARSIMONY}

\description{Summary method for class \code{\link{ga_parsimony-class}}.}

\usage{
\S4method{summary}{ga_parsimony}(object, \dots)
\S3method{print}{summary.ga_parsimony}(x, digits = getOption("digits"), \dots)
}

\arguments{
\item{object}{an object of class \code{\link{ga_parsimony-class}}.}
\item{x}{an object of class \code{summary.ga_parsimony}.}
\item{digits}{number of significant digits.}
\item{\dots}{further arguments passed to or from other methods.}
}

%\details{}
  
\value{The \code{summary} function returns an object of class  \code{\link{ga_parsimony-class}} which can be printed by the corresponding \code{print} method. The function also returns invisibly a list with the information from the genetic algorithm search.}

%\references{}
  
\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{https://edmans.webs.com/}}

%\note{}

\seealso{\code{\link{ga_parsimony}}}

\keyword{optimize}
