# FastGGM
An R package of fast estimating large Gaussian Graphical Model. 

[Homepage](http://www.pitt.edu/~wec47/fastGGM.html) | [Source Code](https://github.com/wt2015-github/FastGGM)

## Description
FastGGM is used to study conditional dependence among variables, which is well known as the Gaussian graphical model (GGM). It uses scaled Lasso regression to obtain asympotically efficient estimation of each entry of a precision matrix under a sparseness condition. It not only estimates the conditional dependence between each pair of variables but also supplies p-value and confidence interval. This is a fast package even for a high-dimensional data set.

## Installation
This package relies on libraries "Rcpp" and "RcppParallel". R commands for installation:
```
install.packages("Rcpp")
install.packages("RcppParallel")
install.packages(pkgs = "FastGGM.tar.gz", repos = NULL, type = "source")
```

## Usage
```
FastGGM(x, lambda)
FastGGM_Parallel(x, lambda)
FastGGM_edgs(x, pairs, lambda)
```

## Arguments
* *x*: a n*p matrix with n samples and p variables.
* *pairs*: a 2-column matrix of variable index for edges (variable pairs), each row has two indexes of a variable pair, start from 1. Note if only one edge, it should be a 1*2 matrix.
* *lambda*: penalty parameter in Lasso regression, larger lambda results in faster calculation and a sparser graph, if don't set then use default sqrt(2*log(p/sqrt(n))/n).

## Details
**FastGGM** and **FastGGM_Parallel** are used for construct global Gaussian graphical model with one and multiple CPUs. **FastGGM_edges** is used for analyzing specified edges (variable pairs) conditionally on all other variables.

## Values
* *precision*: a matrix of precision for the GGM or a vector of precision for the variable pairs.
* *p_precision*: a matrix of p-value for the precision matrix or a vector of p-value for the precision of the variable pairs.
* *partialCor*: a matrix of partial correlation for the GGM or a vector of partial correlation for the vairable pairs.
* *p_partialCor*: a matrix of p-value for the partial correlation matrix or a vector of p-value for the partial correlation of the variable pairs.
* *CI_low_parCor*: a matrix of lower value of 95% confidence interval for partial correlation of the GGM or a vector of lower value of 95% confidence interval for partial correlation of the variable pairs.
* *CI_high_parCor*:  a matrix of higher value of 95% confidence interval for partial correlation or a vector of higher value of 95% confidence interval for partial correlation for the variable pairs.

## Example:
```
# Simulate a sparse precision matrix Omega and sample a data matrix X based on it:
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

# Run FastGGM
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
```

## References
* Wang, Ting et al. "FastGGM: An efficient algorithm for the inference of Gaussian graphical model in biological networks." *PloS Computational Biology* 12.2 (2016): e1004755.
* Ren, Zhao, et al. "Asymptotic normality and optimalities in estimation of large Gaussian graphical models." *The Annals of Statistics* 43.3 (2015): 991-1026.

## Main Authors
[Ting Wang](http://wt2015-github.github.io/) ([email](wang9ting@gmail.com)), [Zhao Ren](http://www.pitt.edu/~zren/) ([email](zren@pitt.edu)), [Ying Ding](http://www.publichealth.pitt.edu/home/directory/ying-ding) ([email](YINGDING@pitt.edu)), [Wei Chen](http://www.pitt.edu/~wec47/index.html) ([email](wei.chen@chp.edu)).
