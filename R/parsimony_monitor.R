
# new function for monitoring within RStudio
parsimony_monitor <- function(object, digits = getOption("digits"), ...)
{ 
  fitnessval <- na.exclude(object@fitnessval)
  fitnesstst <- na.exclude(object@fitnesstst)
  complexity <- na.exclude(object@complexity)
  time_min <- na.exclude(object@minutes_gen)

  sumryStat <- c(mean(fitnessval), max(fitnessval), 
                 mean(fitnesstst), fitnesstst[which.max(fitnessval)], 
                 mean(complexity), complexity[which.max(fitnessval)],
                 time_min) 
  sumryStat <- format(sumryStat, digits = digits)
  
  
#  if (Sys.getenv("RSTUDIO") == "1")
#  {
#    cat(paste0(rep("\b", getOption("width")), collapse = ""))
#    flush.console()
#  }
  
  cat(paste("\rGA-PARSIMONY | iter =", object@iter, "\n")) 
  cat(paste("MeanVal =", sumryStat[1], 
            # "| MeanTst =", sumryStat[3],
            # "| MeanComplexity =", sumryStat[5], 
            "| ValBest =", object@bestfitnessVal,
            "| TstBest =", object@bestfitnessTst,
            "| ComplexBest =", object@bestcomplexity,
            "| Time(min)=", object@minutes_gen,
            "\n"))
  flush.console()
}


parsimony_summary <- function(object, ...)
{
  # compute summary for each step
  x1 <- na.exclude(as.vector(object@fitnessval))
  q1 <- fivenum(x1)
  x2 <- na.exclude(as.vector(object@fitnesstst))
  q2 <- fivenum(x1)
  x3 <- na.exclude(as.vector(object@complexity))
  q3 <- fivenum(x1)
  c(maxval = q1[5], meanval = mean(x1), q3val = q1[4], medianval = q1[3], q1val = q1[2], minval = q1[1],
    maxtst = q2[5], meantst = mean(x2), q3tst = q2[4], mediantst = q2[3], q1tst = q2[2], mintst = q2[1],
    maxcomplex = q3[5], meancomplex = mean(x3), q3complex = q3[4], mediancomplex = q3[3], q1complex = q3[2], mincomplex = q3[1])
}


