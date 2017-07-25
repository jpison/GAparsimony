\name{plot.ga_parsimony-method}
\docType{methods}
\alias{plot.ga_parsimony-method}
\alias{plot.ga_parsimony}

\title{Plot of GA evolution of elitists}

\description{
  The \code{plot} method for \code{\link{ga_parsimony_class}} objects gives a evolution plot of the 
  validation and testing errors, and the number of model features selected of elitists.
}

\usage{
\S4method{plot}{ga_parsimony}(x, general_cex = 0.7, min_ylim=NULL, max_ylim=NULL, iter_auto_ylim=3, 
                              min_iter=NULL, max_iter=NULL, main_label="Boxplot cost evolution", 
                              steps=5, pos_cost_num=-3.1,  pos_feat_num=-1.7,
                              digits_plot=4, width_plot=12, height_plot=6, window=TRUE, ...)
}

\arguments{
  \item{x}{An object of class \code{"ga-parsimony"}.}
  \item{general_cex}{Main text scale.}
  \item{min_ylim}{Min limit on the y-axis.}
  \item{max_ylim}{Max limit on the y-axis.}
  \item{min_iter}{Min GA iteration to visualize.}
  \item{max_iter}{Max GA iteration to visualize.}
  \item{main_label}{Main plot title.}
  \item{iter_auto_ylim}{If it is not NULL, GA iteration to choose the min limit of y-axis.}
  \item{steps}{Number of divisions in y-axis.}

  \item{pos_cost_num}{Relative position of numbers in cost axis.}
  \item{pos_feat_num}{Relative position of numbers in feature axis.}
  \item{digits_plot}{Number of digits to visualize.}
  \item{width_plot}{Figure width in inches.}
  \item{height_plot}{Figure height in inches.}
  \item{window}{If TRUE shows a new window.}
  
  \item{\dots}{Further arguments, currently not used.}
}

\details{Plot method shows the evolution of validation and testing errors, and the number of model features selected of elitists.
White and grey box-plots represent validation and testing errors of elitists evolution, respectively. Continuous and dashed-dotted lines show the validation and testing error 
of the best individual for each generation, respectively. Finally, the shaded area delimits the maximum and minimum number of features, and the dashed line, the number fo features 
of the best individual.}
  
\value{
The method invisibly return a list with the elistists validation error, testing error and model complexity in the whole GA process.
}

\seealso{\code{\link{ga_parsimony}}, \code{\link{ga_parsimony_class}}.}

%\references{}

\author{Francisco Javier Martinez de Pison. \email{fjmartin@unirioja.es}. EDMANS Group. \url{http://www.mineriadatos.com}}

\examples{
}

\keyword{methods}
\keyword{hplot}
