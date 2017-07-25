
#############################################################################

clearConsoleLine <- function()
{
  cat(paste0(rep("\b", getOption("width")), collapse = ""))
  flush.console()
}

#############################################################################

#----------------------------------------------------------------------------#

is.RStudio <- function () 
{
  Sys.getenv("RSTUDIO") == "1"
}

#----------------------------------------------------------------------------#
# print a short version of a matrix by allowing to select the number of 
# head/tail rows and columns to display

.printShortMatrix <- function(x, head = 2, tail = 1, chead = 5, ctail = 1, ...)
{ 
  x <- as.matrix(x)
  nr <- nrow(x)
  nc <- ncol(x)
  if(is.na(head <- as.numeric(head))) head <- 2
  if(is.na(tail <- as.numeric(tail))) tail <- 1
  if(is.na(chead <- as.numeric(chead))) chead <- 5
  if(is.na(ctail <- as.numeric(ctail))) ctail <- 1
  
  if(nr > (head + tail + 1))
    { rnames <- rownames(x)
      if(is.null(rnames)) 
        rnames <- paste("[", 1:nr, ",]", sep ="")
      x <- rbind(x[1:head,,drop=FALSE], 
                 rep(NA, nc), 
                 x[(nr-tail+1):nr,,drop=FALSE])
      rownames(x) <- c(rnames[1:head], "...", rnames[(nr-tail+1):nr])
  }
  if(nc > (chead + ctail + 1))
    { cnames <- colnames(x)
      if(is.null(cnames)) 
        cnames <- paste("[,", 1:nc, "]", sep ="")
      x <- cbind(x[,1:chead,drop=FALSE], 
                 rep(NA, nrow(x)), 
                 x[,(nc-ctail+1):nc,drop=FALSE])
      colnames(x) <- c(cnames[1:chead], "...", cnames[(nc-ctail+1):nc])
  }
          
  print(x, na.print = "", ...)
}
