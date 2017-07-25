\name{GAparsimony-package}
\alias{GAparsimony-package}
\alias{GAparsimony}
\docType{package}

\title{GAparsimony}

\description{
Combines feature selection, model tuning, and parsimonious model selection with GA optimization. GA selection procedure is based on separate cost and complexity evaluations. Therefore, the best individuals are initially sorted by an error fitness function, and afterwards, models with similar costs are rearranged according to modelcomplexity measurement so as to foster models of lesser complexity. The algorithm can be run sequentially or in parallel using an explicit master-slave parallelisation.
}

\details{
\pkg{GAparsimony} package is a new GA wrapper automatic procedure that efficiently generated prediction models with reduced complexity and adequate generalization capacity. 
\code{ga_parsimony} function is primarily based on combining feature selection and model parameter tuning with a second novel GA selection process (ReRank algorithm) in order to achieve better overall parsimonious models. 
Unlike other GA methodologies that use a penalty parameter for combining loss and complexity measures into a unique fitness function, the main contribution of this package is that \code{ga_parsimony} selects the best models by considering cost and complexity separately. For this purpose, the ReRank algorithm rearranges individuals by their complexity when there is not a significant difference between their costs. Thus, less complex models with similar accuracy are promoted. Furthermore, because the penalty parameter is unnecessary, there is no consequent uncertainty associated with assigning a correct value beforehand. As a result, with GA-PARSIMONY, an automatic method for obtaining parsimonious models is finally made possible.
}

\author{Francisco Javier Martinez de Pison Ascacibar \email{fjmartin@unirioja.es}}

\section{References}{
Sanz-Garcia A., Fernandez-Ceniceros J., Antonanzas-Torres F., Pernia-Espinoza A.V., Martinez-de-Pison F.J. (2015).  GA-PARSIMONY: A GA-SVR approach with feature selection and parameter optimization to obtain parsimonious solutions for predicting temperature settings in a continuous annealing furnace. Applied Soft Computing 35, 23-38.
Fernandez-Ceniceros J., Sanz-Garcia A., Antonanzas-Torres F., Martinez-de-Pison F.J. (2015). A numerical-informational approach for characterising the ductile behaviour of the T-stub component. Part 2: Parsimonious soft-computing-based metamodel. Engineering Structures 82, 249-260.
Antonanzas-Torres F., Urraca R., Antonanzas J., Fernandez-Ceniceros J., Martinez-de-Pison F.J. (2015). Generation of daily global solar irradiation with support vector machines for regression. Energy Conversion and Management 96, 277-286.
}

% \examples{}

\keyword{package}


