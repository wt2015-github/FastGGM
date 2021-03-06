\name{FastGGM-package}
\alias{FastGGM-package}
\alias{FastGGM}
\docType{package}
\title{
Fast estimation of Gaussian graphical model
}
\description{
FastGGM is used to study conditional dependence among variables, which is well known as the Gaussian graphical model (GGM). It uses scaled Lasso regression to obtain asympotically efficient estimation of each entry of a precision matrix under a sparseness condition. It not only estimates the conditional dependence between each pair of variables but also supplies p-value and confidence interval. This is a fast package even for a high-dimensional data set.
}
\usage{
FastGGM(x, lambda)

FastGGM_Parallel(x, lambda)

FastGGM_edgs(x, pairs, lambda)
}
\arguments{
\tabular{ll}{
x: \tab a n*p matrix with n samples and p variables.\cr
pairs: \tab a 2-column matrix of variable index for edges (variable pairs), each row has two indexes of a variable pair, start from 1. Note if only one edge, it should be a 1*2 matrix.\cr
lambda: \tab penalty parameter in Lasso regression, larger lambda results in faster calculation and a sparser graph, if don't set then use default sqrt(2*log(p/sqrt(n))/n).\cr
}
}
\details{
FastGGM and FastGGM_Parallel are used for construct global Gaussian graphical model with one and multiple CPUs.

FastGGM_edges is used for analyzing specified edges (variable pairs) conditionally on all other variables.
}
\value{
\tabular{ll}{
precision: \tab a matrix of precision for the GGM or a vector of precision for the variable pairs.\cr
p_precision: \tab a matrix of p-value for the precision matrix or a vector of p-value for the precision of the variable pairs.\cr
partialCor: \tab a matrix of partial correlation for the GGM or a vector of partial correlation for the vairable pairs.\cr
p_partialCor: \tab a matrix of p-value for the partial correlation matrix or a vector of p-value for the partial correlation of the variable pairs.\cr
CI_low_parCor: \tab a matrix of lower value of 95\% confidence interval for partial correlation of the GGM or a vector of lower value of 95\% confidence interval for partial correlation of the variable pairs.\cr
CI_high_parCor: \tab a matrix of higher value of 95\% confidence interval for partial correlation or a vector of higher value of 95\% confidence interval for partial correlation for the variable pairs.\cr
}
}
\author{
Ting Wang <wang9ting@gmail.com>, Zhao Ren <zren@pitt.edu>, Ark Fang <fangz.ark@gmail.com>, Ying Ding <YINGDING@pitt.edu>, Wei Chen <wei.chen@chp.edu>.
}
\references{
Ren, Zhao, et al. "Asymptotic normality and optimalities in estimation of large Gaussian graphical models." The Annals of Statistics 43.3 (2015): 991-1026.

Wang, Ting et al. FastGGM: An efficient algorithm for the inference of Gaussian graphical model in biological networks. PLoS Computational Biology, February 12 (2016). DOI: 10.1371/journal.pcbi.1004755.
}
\seealso{
This package is based on libraries \code{\link[Rcpp]{Rcpp}} and \code{\link[RcppParallel]{RcppParallel}}
}
\examples{
## Simulate a sparse precision matrix Omega and sample a data matrix X based on it:
library(MASS) 
prop <- 0.02  # Sparseness
p <- 100  # Number of variables
n <- 50  # Number of samples
for(k in 1:100){
  Omega0.tmp <- matrix(sample(c(0.3, 0.6, 1), p*p, replace=T), p, p)*matrix(rbinom(p*p, 1, prop), p, p) 
  Omega0 <- Omega0.tmp 
  for(i in 1:(p-1)){
    for(j in (i+1):p){
      Omega0[j, i] <- Omega0.tmp[i, j]    
    }    
  } 
  p_prime <- p/2
  Omega <- Omega0
  Omega[(p_prime+1):(2*p_prime), (p_prime+1):(2*p_prime)] <- 2*Omega0[1:p_prime,1:p_prime]
  diag(Omega) <- c(rep(4, p_prime), rep(8, p_prime))
  eigenvalue <- eigen(Omega)$values
  ratio <- max(eigenvalue)/min(eigenvalue)
  cat(k, max(eigenvalue), min(eigenvalue), ratio, '\n')
  if(ratio > 0){
    break
  }
}
cov <- solve(Omega)
X <- mvrnorm(n, rep(0, p), cov)

## Run FastGGM
library(FastGGM)
outlist1 <- FastGGM(X)

library(RcppParallel)
setThreadOptions(numThreads = 4) # set 4 threads for parallel computing
# If not use the above two commands, it will automatically use all available CPUs
outlist2 <- FastGGM_Parallel(X)

# To calculate for edges/variable pairs (1,4), (2,5) and (3,6)
pairs <- matrix(1:6, ncol=2)
outlist3 <- FastGGM_edges(X, pairs)
}
