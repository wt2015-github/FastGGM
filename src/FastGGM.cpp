#include <Rcpp.h>
#include <RcppParallel.h>
#include <cmath>
#include <algorithm>
using namespace std;
using namespace Rcpp;
using namespace RcppParallel;

/*
*** By Ting & Ark, 2015.6.4
*** Reference: Ren, Zhao, et al. "Asymptotic normality and optimalities in estimation of large gaussian graphical model." arXiv preprsize_t arXiv:1309.6024 (2013). 
*** Function 1: Fast conditional GGM
Inputs:
  X: a matrix with n observation and p variables
  lambda: penalty parameter in Lasso regression, if don't set then use default sqrt(2*log(p/sqrt(n))/n)
Outputs:
  precision: a matrix of precision
  p_precision: a matrix of p-value of precision
  partialCor: a matrix of partial correlation
  p_partialCor: a matrix of p-value of partial correlation
  CI_low_parCor: a matrix of lower value of 95% confidence interval for partial correlation
  CI_high_parCor: a matrix of higher value of 95% confidence interval for partial correlation
Usage:
  out_list <- FastGGM(X, lambda)
  
*** Function 2: Conditional dependence for a subset of edges (element pairs)
Inputs:
  X: a matrix with n observation and p variables
  pairs: a matrix with two columns for edges (variable pairs), each row has two indexes of a variable pair, start from 1
  lambda: penalty parameter in Lasso regression, if don't set then use default sqrt(2*log(p/sqrt(n))/n)
Outputs:
  precision: a vector of precision
  p_precision: a vector of p-value of precision
  partialCor: a vector of partial correlation
  p_partialCor: a vector of p-value of partial correlation
  CI_low_parCor: a vector of lower value of 95% confidence interval for partial correlation
  CI_high_parCor: a vector of higher value of 95% confidence interval for partial correlation
Usage:
  out_list <- FastGGM_edges(X, pairs, lambda)

*** Function 3: Fast conditional GGM, parallel computing
Inputs:
  X: a n*p matrix with n observations and p variables
  lambda: penalty parameter in Lasso regression, if don't set then use default sqrt(2*log(p/sqrt(n))/n)
Outputs:
  precision: a matrix of precision
  p_precision: a matrix of p-value of precision
  partialCor: a matrix of partial correlation
  p_partialCor: a matrix of p-value of partial correlation
  CI_low_parCor: a matrix of lower value of 95% confidence interval for partial correlation
  CI_high_parCor: a matrix of higher value of 95% confidence interval for partial correlation
Usage:
  setThreadOptions(numThreads = 4) //set number of threads
  out_list <- FastGGM_Parallel(X, lambda)
*/

/* ****** Sub-function, fast Lasso regression ******* */
NumericVector FastLasso_Rcpp(NumericVector ipy, NumericMatrix ipx, double lambda, size_t N){
  double tol = 1e-3; //threshold for stopping
  size_t p = ipy.size();
  double dbm;  //maximum of beta difference for while loop
  NumericVector beta(p); //initialize output beta vector with length p and filled with 0.0
  NumericVector gc(p); //initialize grandient components
  
  //update beta vector with coordinate descent, covariance updates
  do{
    dbm = 0;
    for(size_t j = 0; j < p; j++){
      double z = (ipy[j] - gc[j])/N + beta[j];
      double beta_tmp = max(0.0, z - lambda) - max(0.0, -z - lambda);
      double diffBeta = beta_tmp - beta[j];
      double diffabs = abs(diffBeta);
      if (diffabs > 0){
        beta[j] = beta_tmp;
        for (size_t i = 0; i < p; i++){
          gc[i] = gc[i] + ipx(i,j) * diffBeta;
        }
        dbm = max(dbm, diffabs);
      }
    }
  }
  while(dbm >= tol);
  
  return beta;
}

/* ***** Function 1 ******** */
// [[Rcpp::export]]
List FastGGM(NumericMatrix x, double lambda=NA_REAL){
  // Stop threshold and parameter for edge decision
  double tol = 1e-3;
  
  size_t n = x.nrow();
  size_t p = x.ncol();
  
  //penalty parameter for L1 norm of betas
  if(NumericVector::is_na(lambda)){
    lambda = sqrt(2 * log(p/sqrt(n))/n);
    cout << "Use default lambda = sqrt(2*log(p/sqrt(n))/n)" << endl;
  }
  cout << "In this case, lambda = " << lambda << endl;
  
  NumericMatrix precision(p,p);
  NumericMatrix p_precision(p,p);
  NumericMatrix partialCor(p,p);
  NumericMatrix p_partialCor(p,p);
  NumericMatrix CI_low_parCor(p,p);
  NumericMatrix CI_high_parCor(p,p);
  
  // Center matrix
  cout << "Center each column." << endl;
  NumericMatrix x_cent(n,p);
  for(size_t i = 0; i < p; i++){
    x_cent(_,i) = x(_,i) - mean(x(_,i));
  }
  
  // Standardize matrix
  cout << "Standardize each column." << endl;
  NumericVector x_l2norm(p);
  for(size_t i = 0; i < p; i++){
    x_l2norm[i] = sqrt(sum(pow(x_cent(_,i),2)));
    if(x_l2norm[i] == 0){
      stop("some variables have 0 variance, please remove them at first.");
    }
  }
  NumericMatrix x_stan(n,p);
  for(size_t i = 0; i < p; i++){
    x_stan(_,i) = x_cent(_,i) / x_l2norm[i] * sqrt(n);
  }
  
  // Pre-calculate inner product matrixes
  cout << "Pre-calculate inner product matrixes." << endl;
  NumericMatrix IP_YX(p,p);
  for(size_t i = 0; i < p; i++){
    for(size_t j = 0; j < p; j++){
      IP_YX(i,j) = inner_product(x_cent(_,i).begin(), x_cent(_,i).end(), x_stan(_,j).begin(), 0.0);
    }
  }
  NumericMatrix IP_XX(p,p);
  for(size_t i = 0; i < p; i++){
    IP_XX(i,_) = IP_YX(i,_) / x_l2norm[i] * sqrt(n);
  }
  
  // Pre-calculate scaled Lasso for each variable
  cout << "Pre-calculate scaled Lasso for each variable." << endl;
  NumericMatrix beta_pre(p,p);
  NumericMatrix epsi_pre(n,p);
  
  for(size_t i = 0; i < p; i++){
    if(i % 100 == 0){
      cout <<"Pre-Lasso for variable " << (i + 1) << endl;
    }
    double sigma = 1.0;
    NumericVector beta(p-1);
    NumericVector epsi(n);
    double reg, diffBeta, sigma_tmp, diffSigma;
    
    // Extract IP_XX[-i,-i] and IP_YX[i,-i]
    NumericMatrix tmpxx(p-1,p-1);
    NumericVector tmpyx(p-1);
    size_t t = 0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        tmpyx[t] = IP_YX(i,k);
        size_t t1 = 0;
        for (size_t l = 0; l < p; l++){ // l is index of row, k is index of col
          if (l != i){
            tmpxx(t1,t) = IP_XX(l,k);
            t1++;
          }
        }
        t++;
      }
    }
    // Extract x_stan[,-i]
    NumericMatrix tmp_x(n, p-1);
    t=0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        tmp_x(_,t) = x_stan(_,k);
        t++;
      }
    }
    
    //Scaled_lasso
    size_t iter = 0;
    do {
      //Update beta when fix sigma
      reg = sigma * lambda;
      NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
      diffBeta = max(abs(beta_tmp - beta));
      beta = beta_tmp;
      //Update sigma when fix beta
      NumericVector tmp_y(n);
      for (size_t k = 0; k < n; k++ ){
        tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
      } //Multiplication of x.stan[,-i] and beta
      epsi = x_cent(_,i) - tmp_y;
      sigma_tmp = sqrt(sum(pow(epsi,2))/n);
      diffSigma = abs(sigma_tmp - sigma);
      sigma = sigma_tmp;
      iter++;
    }
    while((diffSigma >= tol || diffBeta >= tol) && iter < 10); //Stop iteration
    epsi_pre(_,i) = epsi;
    t = 0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        beta_pre(k,i) = beta[t];
        t++;
      }
    }
  }
  
  // Fast pairwise GGM based on the pre-calculated scaled Lasso
  cout << "Perform pairwise GGM." << endl;
  for(size_t i = 0; i < (p-1); i++){
    if(i % 100 == 0){
      cout <<"Pair-Lasso for variable " << (i + 1) << endl;
    }
    for(size_t j = (i+1); j < p; j++){
      NumericVector epsi_i(n);
      NumericVector epsi_j(n);
      
      // Solve scaled Lasso i without variable j
      if(beta_pre(j,i) == 0){
        epsi_i = epsi_pre(_,i);
      } else{
        double sigma = 1.0;
        NumericVector beta(p-2);
        NumericVector epsi(n);
        double reg, diffBeta, sigma_tmp, diffSigma;
        
        // Extract IP_XX[-c(i,j),-c(i,j)] and IP_YX[i,-c(i,j)]
        NumericMatrix tmpxx(p-2,p-2);
        NumericVector tmpyx(p-2);
        size_t t = 0;
        for (size_t k = 0; k < p; k++){
          if (k != i && k != j){
            tmpyx[t] = IP_YX(i,k);
            size_t t1 = 0;
            for (size_t l = 0; l < p; l++){
              if ( l != i && l != j){
                tmpxx(t1,t) = IP_XX(l,k);
                t1++;
              }
            }
            t++;
          }
        }
        // Extract x_stan[,-c(i,j)]
        NumericMatrix tmp_x(n, p-2);
        t=0;
        for (size_t k = 0; k < p; k++){
          if (k != i && k != j){
            tmp_x(_,t) = x_stan(_,k);
            t++;
          }
        }
        
        //Scaled_lasso
        size_t iter = 0;
        do {
          //Update beta when fix sigma
          reg = sigma * lambda;
          NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
          diffBeta = max(abs(beta_tmp - beta));
          beta = beta_tmp;
          //Update sigma when fix beta
          NumericVector tmp_y(n);
          for (size_t k = 0; k < n; k++){
            tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
          }
          epsi = x_cent(_,i) - tmp_y;
          sigma_tmp = sqrt(sum(pow(epsi,2))/n);
          diffSigma = abs(sigma_tmp - sigma);
          sigma = sigma_tmp;
          iter++;
        }
        while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
        epsi_i = epsi;
      }
      
      // Solve scaled Lasso j without variable i
      if(beta_pre(i,j) == 0){
        epsi_j = epsi_pre(_,j);
      } else{
        double sigma = 1.0;
        NumericVector beta(p-2);
        NumericVector epsi(n);
        double reg, diffBeta, sigma_tmp, diffSigma;
        
        NumericMatrix tmpxx(p-2,p-2);
        NumericVector tmpyx(p-2);
        size_t t = 0;
        for (size_t k = 0; k < p; k++){
          if ( k != i && k != j){
            tmpyx[t] = IP_YX(j,k);
            size_t t1 = 0;
            for (size_t l = 0; l < p; l++){
              if ( l != i && l != j){
                tmpxx(t1,t) = IP_XX(l,k);
                t1++;
              }
            }
            t++;
          }
        }
        NumericMatrix tmp_x(n, p-2);
        t = 0;
        for (size_t k = 0; k < p; k++){
          if (k != i && k != j){
            tmp_x(_,t) = x_stan(_,k);
            t++;
          }
        }
        
        //Scaled_lasso
        size_t iter = 0;
        do {
          //Update beta when fix sigma
          reg = sigma * lambda;
          NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
          diffBeta = max(abs(beta_tmp - beta));
          beta = beta_tmp;
          //Update sigma when fix beta
          NumericVector tmp_y(n);
          for (size_t k = 0; k < n; k++){
            tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
          }
          epsi = x_cent(_,j) - tmp_y;
          sigma_tmp = sqrt(sum(pow(epsi,2))/n);
          diffSigma = abs(sigma_tmp - sigma);
          sigma = sigma_tmp;
          iter++;
        }
        while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
        epsi_j = epsi;
      }
      
      // Precision, solve(t(epsi_ij)%*%epsi_ij/n)
      NumericMatrix omega_tmp(2,2);
      NumericMatrix omega(2,2);
      omega_tmp(0,0) = inner_product(epsi_i.begin(), epsi_i.end(), epsi_i.begin(), 0.0)/n;
      omega_tmp(1,1) = inner_product(epsi_j.begin(), epsi_j.end(), epsi_j.begin(), 0.0)/n;
      omega_tmp(0,1) = inner_product(epsi_i.begin(), epsi_i.end(), epsi_j.begin(), 0.0)/n;
      omega_tmp(1,0) = omega_tmp(0,1);
      // Inverse matrix of omega_tmp
      double tmp = omega_tmp(0,0) * omega_tmp(1,1) - omega_tmp(0,1) * omega_tmp(1,0);
      omega(0,0) = omega_tmp(1,1)/tmp;
      omega(0,1) = -omega_tmp(0,1)/tmp;
      omega(1,0) = -omega_tmp(1,0)/tmp;
      omega(1,1) = omega_tmp(0,0)/tmp;
      precision(i,i) = omega(0,0);
      precision(j,j) = omega(1,1);
      precision(i,j) = omega(0,1);
      precision(j,i) = omega(1,0);
      
      // P-value of precision
      double var_new = (pow(omega(0,1),2) + omega(0,0) * omega(1,1))/n;
      double zscore = abs(omega(0,1)/sqrt(var_new));
      p_precision(i,j) = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
      p_precision(j,i) = p_precision(i,j);
      
      // Partial correlation
      partialCor(i,j) = -omega(0,1)/sqrt(omega(0,0) * omega(1,1));
      partialCor(j,i) = partialCor(i,j);
      
      // P-value of partial correlation
      var_new = pow((1-pow(partialCor(i,j),2)),2)/n;
      zscore = abs(partialCor(i,j))/sqrt(var_new);
      p_partialCor(i,j) = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
      p_partialCor(j,i) = p_partialCor(i,j);
      
      // 95% confidence interval of partial correlation
      double z_95CI = 1.96; // z-score of 95% CI
      CI_low_parCor(i,j) = partialCor(i,j) - z_95CI * (1 - pow(partialCor(i,j),2))/sqrt(n);
      CI_low_parCor(j,i) = CI_low_parCor(i,j);
      CI_high_parCor(i,j) = partialCor(i,j) + z_95CI * (1 - pow(partialCor(i,j),2))/sqrt(n);
      CI_high_parCor(j,i) = CI_high_parCor(i,j);
    }
  }
  
  return List::create(_["precision"] = precision, _["partialCor"] = partialCor, _["p_precision"] = p_precision, _["p_partialCor"] = p_partialCor, _["CI_low_parCor"] = CI_low_parCor, _["CI_high_parCor"] = CI_high_parCor);
}

/* ***** Function 2 ******** */
// [[Rcpp::export]]
List FastGGM_edges(NumericMatrix x, NumericMatrix pairs, double lambda=NA_REAL){
  size_t n = x.nrow();
  size_t p = x.ncol();
  
  if(pairs.ncol() != 2){
    stop("matrix of variable pairs should have 2 columns.");
  }
  
  for(size_t i = 0; i < pairs.nrow(); i++){
    if(pairs(i,0) == pairs(i,1)){
      stop("indexes of a variable pair should be different.");
    }
  }
  
  if(max(pairs(_,0)) > p || max(pairs(_,1)) > p){
  	stop("variable index should not be larger than the number of variables.");
  }
  
  // Stop threshold and parameter for edge decision
  double tol = 1e-3;
  pairs(_,0) = pairs(_,0) - 1;
  pairs(_,1) = pairs(_,1) - 1;
  
  NumericVector precision(pairs.nrow());
  NumericVector p_precision(pairs.nrow());
  NumericVector partialCor(pairs.nrow());
  NumericVector p_partialCor(pairs.nrow());
  NumericVector CI_low_parCor(pairs.nrow());
  NumericVector CI_high_parCor(pairs.nrow());
  
  //penalty parameter for L1 norm of betas
  if(NumericVector::is_na(lambda)){
    lambda = sqrt(2 * log(p/sqrt(n))/n);
    cout << "Use default lambda = sqrt(2*log(p/sqrt(n))/n)" << endl;
  }
  cout << "In this case, lambda = " << lambda << endl;

  // Center matrix
  cout << "Center matrix columns." << endl;
  NumericMatrix x_cent(n,p);
  for(size_t k = 0; k < p; k++){
    x_cent(_,k) = x(_,k) - mean(x(_,k));
  }
  
  // Standardize matrix
  cout << "Standardize matrix columns." << endl;
  NumericVector x_l2norm(p);
  for(size_t k = 0; k < p; k++){
    x_l2norm[k] = sqrt(sum(pow(x_cent(_,k),2)));
    if(x_l2norm[k] == 0){
      stop("some variables have 0 variance, please remove them at first.");
    }
  }
  NumericMatrix x_stan(n,p);
  for(size_t k = 0; k < p; k++){
    x_stan(_,k) = x_cent(_,k) / x_l2norm[k] * sqrt(n);
  }
  
  // Pre-calculate inner product matrixes
  cout << "Pre-calculate inner product matrixes." << endl;
  NumericMatrix IP_YX(p,p);
  for(size_t i = 0; i < p; i++){
    for(size_t j = 0; j < p; j++){
      IP_YX(i,j) = inner_product(x_cent(_,i).begin(), x_cent(_,i).end(), x_stan(_,j).begin(), 0.0);
    }
  }
  NumericMatrix IP_XX(p,p);
  for(size_t i = 0; i < p; i++){
    IP_XX(i,_) = IP_YX(i,_) / x_l2norm[i] * sqrt(n);
  }
  
  // Pre-calculate scaled Lasso for each variable
  cout << "Pre-calculate scaled Lasso for each variable." << endl;
  NumericMatrix beta_pre(p,p);
  NumericMatrix epsi_pre(n,p);
  NumericVector variable(pairs.nrow() + pairs.nrow());
  size_t mm = 0;
  for( ; mm < pairs.nrow(); mm++){
    variable[mm] = pairs(mm,0);
  }
  for(size_t nn=0; nn < pairs.nrow(); mm++, nn++){
    variable[mm] = pairs(nn,1);
  }
  NumericVector variable_uniq = unique(variable);
  
  for(size_t ii = 0; ii < variable_uniq.size(); ii++){
    int i = variable_uniq[ii];
    double sigma = 1.0;
    NumericVector beta(p-1);
    NumericVector epsi(n);
    double reg, diffBeta, sigma_tmp, diffSigma;
    
    // Extract IP_XX[-i,-i] and IP_YX[i,-i]
    NumericMatrix tmpxx(p-1,p-1);
    NumericVector tmpyx(p-1);
    size_t t = 0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        tmpyx[t] = IP_YX(i,k);
        size_t t1 = 0;
        for (size_t l = 0; l < p; l++){ // l is index of row, k is index of col
          if (l != i){
            tmpxx(t1,t) = IP_XX(l,k);
            t1++;
          }
        }
        t++;
      }
    }
    // Extract x_stan[,-i]
    NumericMatrix tmp_x(n, p-1);
    t=0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        tmp_x(_,t) = x_stan(_,k);
        t++;
      }
    }
    
    //Scaled_lasso
    size_t iter = 0;
    do {
      //Update beta when fix sigma
      reg = sigma * lambda;
      NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
      diffBeta = max(abs(beta_tmp - beta));
      beta = beta_tmp;
      //Update sigma when fix beta
      NumericVector tmp_y(n);
      for (size_t k = 0; k < n; k++ ){
        tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
      } //Multiplication of x.stan[,-i] and beta
      epsi = x_cent(_,i) - tmp_y;
      sigma_tmp = sqrt(sum(pow(epsi,2))/n);
      diffSigma = abs(sigma_tmp - sigma);
      sigma = sigma_tmp;
      iter++;
    }
    while((diffSigma >= tol || diffBeta >= tol) && iter < 10); //Stop iteration
    epsi_pre(_,i) = epsi;
    t = 0;
    for (size_t k = 0; k < p; k++){
      if (k != i){
        beta_pre(k,i) = beta[t];
        t++;
      }
    }
  }
  
  // Fast pairwise GGM based on the pre-calculated scaled Lasso
  cout << "Perform GGM for each pair of variables." << endl;
  for(size_t jj = 0; jj < pairs.nrow(); jj++){
    cout <<"Pair-Lasso for variable pair " << (jj + 1) << endl;
    int i = pairs(jj,0);
    int j = pairs(jj,1);
    
    NumericVector epsi_i(n);
    NumericVector epsi_j(n);
    
    // Solve scaled Lasso i without variable j
    if(beta_pre(j,i) == 0){
      epsi_i = epsi_pre(_,i);
    } else{
      double sigma = 1.0;
      NumericVector beta(p-2);
      NumericVector epsi(n);
      double reg, diffBeta, sigma_tmp, diffSigma;
      
      // Extract IP_XX[-c(i,j),-c(i,j)] and IP_YX[i,-c(i,j)]
      NumericMatrix tmpxx(p-2,p-2);
      NumericVector tmpyx(p-2);
      size_t t = 0;
      for (size_t k = 0; k < p; k++){
        if (k != i && k != j){
          tmpyx[t] = IP_YX(i,k);
          size_t t1 = 0;
          for (size_t l = 0; l < p; l++){
            if ( l != i && l != j){
              tmpxx(t1,t) = IP_XX(l,k);
              t1++;
            }
          }
          t++;
        }
      }
      // Extract x_stan[,-c(i,j)]
      NumericMatrix tmp_x(n, p-2);
      t=0;
      for (size_t k = 0; k < p; k++){
        if (k != i && k != j){
          tmp_x(_,t) = x_stan(_,k);
          t++;
        }
      }
      
      //Scaled_lasso
      size_t iter = 0;
      do {
        //Update beta when fix sigma
        reg = sigma * lambda;
        NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
        diffBeta = max(abs(beta_tmp - beta));
        beta = beta_tmp;
        //Update sigma when fix beta
        NumericVector tmp_y(n);
        for (size_t k = 0; k < n; k++){
          tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
        }
        epsi = x_cent(_,i) - tmp_y;
        sigma_tmp = sqrt(sum(pow(epsi,2))/n);
        diffSigma = abs(sigma_tmp - sigma);
        sigma = sigma_tmp;
        iter++;
      }
      while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
      epsi_i = epsi;
    }
    
    // Solve scaled Lasso j without variable i
    if(beta_pre(i,j) == 0){
      epsi_j = epsi_pre(_,j);
    } else{
      double sigma = 1.0;
      NumericVector beta(p-2);
      NumericVector epsi(n);
      double reg, diffBeta, sigma_tmp, diffSigma;
      
      NumericMatrix tmpxx(p-2,p-2);
      NumericVector tmpyx(p-2);
      size_t t = 0;
      for (size_t k = 0; k < p; k++){
        if ( k != i && k != j){
          tmpyx[t] = IP_YX(j,k);
          size_t t1 = 0;
          for (size_t l = 0; l < p; l++){
            if ( l != i && l != j){
              tmpxx(t1,t) = IP_XX(l,k);
              t1++;
            }
          }
          t++;
        }
      }
      NumericMatrix tmp_x(n, p-2);
      t = 0;
      for (size_t k = 0; k < p; k++){
        if (k != i && k != j){
          tmp_x(_,t) = x_stan(_,k);
          t++;
        }
      }
      
      //Scaled_lasso
      size_t iter = 0;
      do {
        //Update beta when fix sigma
        reg = sigma * lambda;
        NumericVector beta_tmp = FastLasso_Rcpp(tmpyx, tmpxx, reg, n);
        diffBeta = max(abs(beta_tmp - beta));
        beta = beta_tmp;
        //Update sigma when fix beta
        NumericVector tmp_y(n);
        for (size_t k = 0; k < n; k++){
          tmp_y[k] = inner_product(tmp_x(k,_).begin(), tmp_x(k,_).end(), beta.begin(), 0.0);
        }
        epsi = x_cent(_,j) - tmp_y;
        sigma_tmp = sqrt(sum(pow(epsi,2))/n);
        diffSigma = abs(sigma_tmp - sigma);
        sigma = sigma_tmp;
        iter++;
      }
      while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
      epsi_j = epsi;
    }
    
    // Precision, solve(t(epsi_ij)%*%epsi_ij/n)
    NumericMatrix omega_tmp(2,2);
    NumericMatrix omega(2,2);
    omega_tmp(0,0) = inner_product(epsi_i.begin(), epsi_i.end(), epsi_i.begin(), 0.0)/n;
    omega_tmp(1,1) = inner_product(epsi_j.begin(), epsi_j.end(), epsi_j.begin(), 0.0)/n;
    omega_tmp(0,1) = inner_product(epsi_i.begin(), epsi_i.end(), epsi_j.begin(), 0.0)/n;
    omega_tmp(1,0) = omega_tmp(0,1);
    // Inverse matrix of omega_tmp
    double tmp = omega_tmp(0,0) * omega_tmp(1,1) - omega_tmp(0,1) * omega_tmp(1,0);
    omega(0,0) = omega_tmp(1,1)/tmp;
    omega(0,1) = -omega_tmp(0,1)/tmp;
    omega(1,0) = -omega_tmp(1,0)/tmp;
    omega(1,1) = omega_tmp(0,0)/tmp;
    
    precision[jj] = omega(0,1);
    
    // P-value of precision
    double var_new = (pow(omega(0,1),2) + omega(0,0) * omega(1,1))/n;
    double zscore = abs(omega(0,1)/sqrt(var_new));
    p_precision[jj] = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
    
    // Partial correlation
    partialCor[jj] = -omega(0,1)/sqrt(omega(0,0) * omega(1,1));
    
    // P-value of partial correlation
    var_new = pow((1 - pow(partialCor[jj],2)),2)/n;
    zscore = abs(partialCor[jj])/sqrt(var_new);
    p_partialCor[jj] = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
    
    // 95% confidence interval of partial correlation
    double z_95CI = 1.96; // z-score of 95% CI
    CI_low_parCor[jj] = partialCor[jj] - z_95CI * (1 - pow(partialCor[jj],2))/sqrt(n);
    CI_high_parCor[jj] = partialCor[jj] + z_95CI * (1 - pow(partialCor[jj],2))/sqrt(n);
  }
  
  return List::create(_["precision"] = precision, _["partialCor"] = partialCor, _["p_precision"] = p_precision, _["p_partialCor"] = p_partialCor, _["CI_low_parCor"] = CI_low_parCor, _["CI_high_parCor"] = CI_high_parCor);
}

/* ***** Function 3, subfunction ******** */
//[[Rcpp::depends(RcppParallel)]]
struct Center : public Worker {
  const RMatrix<double> mat;
  RMatrix<double> rmat;
  Center(const NumericMatrix mat, NumericMatrix rmat)
  : mat(mat), rmat(rmat) {}
  void operator()(size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++){
      RMatrix<double>::Column col_1 = mat.column(i);
      double colSum = accumulate(col_1.begin(), col_1.end(), 0.0);
      double colMean = colSum / mat.nrow();
      for(size_t j = 0; j < mat.nrow(); j++){
        rmat(j,i) = mat(j,i) - colMean;
      }
    }
  }
};

/* ***** Function 3, subfunction ******** */
//[[Rcpp::depends(RcppParallel)]]
struct Standardize : public Worker {
  const RMatrix<double> mat;
  RMatrix<double> rmat;
  Standardize(const NumericMatrix mat, NumericMatrix rmat)
  : mat(mat), rmat(rmat) {}
  void operator()(size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++){
      RMatrix<double>::Column col_1 = mat.column(i);
      vector<double> tmp_pow2(mat.nrow());
      for(size_t k = 0; k < mat.nrow(); k++){
        tmp_pow2[k] = pow(col_1[k], 2);
      }
      double l2norm = accumulate(tmp_pow2.begin(), tmp_pow2.end(), 0.0);
      l2norm = sqrt(l2norm);
      if(l2norm == 0){
        stop("some variables have 0 variance, please remove them at first.");
      }
      for(size_t j = 0; j < mat.nrow(); j++){
        rmat(j,i) = mat(j,i) / l2norm * sqrt(mat.nrow());
      }
    }
  }
};

/* ***** Function 3, subfunction ******** */
// [[Rcpp::depends(RcppParallel)]]
struct InnerProduct : public Worker {
  const RMatrix<double> mat1;
  const RMatrix<double> mat2;
  RMatrix<double> rmat;
  InnerProduct(const NumericMatrix mat1, const NumericMatrix mat2, NumericMatrix rmat)
  : mat1(mat1), mat2(mat2), rmat(rmat) {}
  void operator()(size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      for (size_t j = 0; j < mat2.ncol(); j++) {
        RMatrix<double>::Column col_1 = mat1.column(i);
        RMatrix<double>::Column col_2 = mat2.column(j);
        rmat(i,j) = inner_product(col_1.begin(), col_1.end(), col_2.begin(), 0.0);
      }
    }
  }
};

/* ***** Function 3, subfunction ******** */
// [[Rcpp::depends(RcppParallel)]]
struct PreSLasso : public Worker {
  const RMatrix<double> ipyx;
  const RMatrix<double> ipxx;
  const RMatrix<double> stan;
  const RMatrix<double> cent;
  const double lambda;
  const double tol;
  RMatrix<double> rmat1; // output beta_pre
  RMatrix<double> rmat2; // output epsi_pre
  PreSLasso(const NumericMatrix ipyx, const NumericMatrix ipxx, const NumericMatrix stan, const NumericMatrix cent, const double lambda, const double tol, NumericMatrix rmat1, NumericMatrix rmat2)
  : ipyx(ipyx), ipxx(ipxx), stan(stan), cent(cent), lambda(lambda), tol(tol), rmat1(rmat1), rmat2(rmat2) {}
  void operator()(size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      vector<double> beta((stan.ncol() - 1), 0.0);
      vector<double> epsi(stan.nrow(), 0.0);
      double sigma = 1.0;
      double reg = 0.0;
      double sigma_tmp = 0.0;
      double diffSigma = 0.0;
      double diffBeta;
      // Extract IP_XX[-i,-i] and IP_YX[i,-i]
      vector< vector<double> > tmpxx( (stan.ncol() - 1), vector<double>((stan.ncol() - 1), 0.0) );
      vector<double> tmpyx((stan.ncol() - 1), 0.0);
      size_t t = 0;
      for (size_t k = 0; k < ipyx.ncol(); k++){
        if (k != i){
          tmpyx[t] = ipyx(i,k);
          size_t t1 = 0;
          for (size_t l = 0; l < ipxx.nrow(); l++){ // l is index of row, k is index of col
            if (l != i){
              tmpxx[t1][t] = ipxx(l,k);
              t1++;
            }
          }
          t++;
        }
      }
      // Extract x_stan[,-i]
      vector< vector<double> > tmp_x( stan.nrow(), vector<double>((stan.ncol() - 1), 0.0) );
      t = 0;
      for(size_t k = 0; k < stan.ncol(); k++){
        if (k != i){
          for(size_t m = 0; m < stan.nrow(); m++){
            tmp_x[m][t] = stan(m,k);
          }
          t++;
        }
      }
      //Scaled_lasso
      size_t iter = 0;
      do {
        diffBeta = 0.0;
        //Update beta when fix sigma, Lasso regression
        reg = sigma * lambda;
        vector<double> beta_tmp(tmpyx.size(), 0.0); //initialize output beta vector
        vector<double> gc(tmpyx.size(), 0.0); //initialize grandient components
        double dbm; //maximum of beta difference for while loop in Lasso
        do{
          dbm = 0.0;
          for(size_t jj = 0; jj < tmpyx.size(); jj++){
            double z = (tmpyx[jj] - gc[jj])/stan.nrow() + beta_tmp[jj];
            double beta_inlasso = max(0.0, z - reg) - max(0.0, -z - reg);
            double diffBeta_inlasso = beta_inlasso - beta_tmp[jj];
            double diffabs = abs(diffBeta_inlasso);
            if (diffabs > 0){
              beta_tmp[jj] = beta_inlasso;
              for (size_t ii = 0; ii < tmpyx.size(); ii++){
                gc[ii] = gc[ii] + tmpxx[ii][jj] * diffBeta_inlasso;
              }
              dbm = max(dbm, diffabs);
            }
          }
        }
        while(dbm >= tol); //update beta vector with coordinate descent, covariance updates
        for(size_t kk = 0; kk < beta_tmp.size(); kk++){
          diffBeta = max(diffBeta, abs(beta_tmp[kk] - beta[kk]));
          beta[kk] = beta_tmp[kk];
        }
        //Update sigma when fix beta
        vector<double> tmp_y(stan.nrow(), 0.0);
        for (size_t k = 0; k < stan.nrow(); k++ ){
          tmp_y[k] = inner_product(tmp_x[k].begin(), tmp_x[k].end(), beta.begin(), 0.0);
          epsi[k] = cent(k,i) - tmp_y[k];
        }
        vector<double> epsi2(stan.nrow(), 0.0);
        for(size_t kk = 0; kk < epsi.size(); kk++){
          epsi2[kk] = pow(epsi[kk], 2);
        }
        sigma_tmp = accumulate(epsi2.begin(), epsi2.end(), 0.0);
        sigma_tmp = sqrt(sigma_tmp/stan.nrow());
        diffSigma = abs(sigma_tmp - sigma);
        sigma = sigma_tmp;
        iter++;
      }
      while((diffSigma >= tol || diffBeta >= tol) && iter < 10); //Stop iteration
      for(size_t k = 0; k < stan.nrow(); k++){
        rmat2(k,i) = epsi[k];
      }
      t = 0;
      for (size_t k = 0; k < stan.ncol(); k++){
        if (k != i){
          rmat1(k,i) = beta[t];
          t++;
        }
      }
    }
  }
};

/* ***** Function 3, subfunction ******** */
// [[Rcpp::depends(RcppParallel)]]
struct PairGGM : public Worker {
  const RMatrix<double> prebeta;
  const RMatrix<double> preepsi;
  const RMatrix<double> ipyx;
  const RMatrix<double> ipxx;
  const RMatrix<double> stan;
  const RMatrix<double> cent;
  const double lambda;
  const double tol;
  RMatrix<double> rmat1; // output precision
  RMatrix<double> rmat2; // output p_precision
  RMatrix<double> rmat3; // output partialCor
  RMatrix<double> rmat4; // output p_partialCor
  RMatrix<double> rmat5; // output CI_low_parCor
  RMatrix<double> rmat6; // output CI_high_parCor
  PairGGM(const NumericMatrix prebeta, const NumericMatrix preepsi, const NumericMatrix ipyx, const NumericMatrix ipxx, const NumericMatrix stan, const NumericMatrix cent, const double lambda, const double tol, NumericMatrix rmat1, NumericMatrix rmat2, NumericMatrix rmat3, NumericMatrix rmat4, NumericMatrix rmat5, NumericMatrix rmat6)
      : prebeta(prebeta), preepsi(preepsi), ipyx(ipyx), ipxx(ipxx), stan(stan), cent(cent), lambda(lambda), tol(tol), rmat1(rmat1), rmat2(rmat2), rmat3(rmat3), rmat4(rmat4), rmat5(rmat5), rmat6(rmat6) {}
  void operator()(size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      for(size_t j = (i + 1); j < stan.ncol(); j++){
        vector<double> epsi_i(stan.nrow(), 0.0);
        vector<double> epsi_j(stan.nrow(), 0.0);
        if(prebeta(j,i) == 0){
          for(size_t k = 0; k < stan.nrow(); k++){
            epsi_i[k] = preepsi.column(i)[k];
          }
        } else{
          double sigma = 1.0;
          double reg = 0.0;
          double diffBeta;
          double sigma_tmp = 0.0;
          double diffSigma = 0.0;
          vector<double> beta((stan.ncol() - 2), 0.0);
          vector<double> epsi(stan.nrow(), 0.0);
          // Extract IP_XX[-c(i,j),-c(i,j)] and IP_YX[i,-c(i,j)]
          vector< vector<double> > tmpxx( (stan.ncol() - 2), vector<double>((stan.ncol() - 2), 0.0) );
          vector<double> tmpyx((stan.ncol() - 2), 0.0);
          size_t t = 0;
          for (size_t k = 0; k < stan.ncol(); k++){
            if (k != i && k != j){
              tmpyx[t] = ipyx(i,k);
              size_t t1 = 0;
              for (size_t l = 0; l < stan.ncol(); l++){
                if ( l != i && l != j){
                  tmpxx[t1][t] = ipxx(l,k);
                  t1++;
                }
              }
              t++;
            }
          }
          // Extract x_stan[,-c(i,j)]
          vector< vector<double> > tmp_x( stan.nrow(), vector<double>((stan.ncol() - 2), 0.0) );
          t=0;
          for (size_t k = 0; k < stan.ncol(); k++){
            if (k != i && k != j){
              for(size_t m = 0; m < stan.nrow(); m++){
                tmp_x[m][t] = stan(m,k);
              }
              t++;
            }
          }
          size_t iter = 0;
          do {
            diffBeta = 0.0;
            //Update beta when fix sigma, Lasso regression
            reg = sigma * lambda;
            vector<double> beta_tmp(tmpyx.size(), 0.0);
            vector<double> gc(tmpyx.size(), 0.0);
            double dbm;
            do{
              dbm = 0;
              for(size_t jj = 0; jj < tmpyx.size(); jj++){
                double z = (tmpyx[jj] - gc[jj])/stan.nrow() + beta_tmp[jj];
                double beta_inlasso = max(0.0, z - reg) - max(0.0, -z - reg);
                double diffBeta_inlasso = beta_inlasso - beta_tmp[jj];
                double diffabs = abs(diffBeta_inlasso);
                if (diffabs > 0){
                  beta_tmp[jj] = beta_inlasso;
                  for (size_t ii = 0; ii < tmpyx.size(); ii++){
                    gc[ii] = gc[ii] + tmpxx[ii][jj] * diffBeta_inlasso;
                  }
                  dbm = max(dbm, diffabs);
                }
              }
            }
            while(dbm >= tol);
            for(size_t kk = 0; kk < beta_tmp.size(); kk++){
              diffBeta = max(diffBeta, abs(beta_tmp[kk] - beta[kk]));
              beta[kk] = beta_tmp[kk];
            }
            //Update sigma when fix beta
            vector<double> tmp_y(stan.nrow(), 0.0);
            for (size_t k = 0; k < stan.nrow(); k++){
              tmp_y[k] = inner_product(tmp_x[k].begin(), tmp_x[k].end(), beta.begin(), 0.0);
              epsi[k] = cent(k,i) - tmp_y[k];
            }
            vector<double> epsi2(stan.nrow(), 0.0);
            for(size_t kk = 0; kk < epsi.size(); kk++){
              epsi2[kk] = pow(epsi[kk],2);
            }
            sigma_tmp = accumulate(epsi2.begin(), epsi2.end(), 0.0);
            sigma_tmp = sqrt(sigma_tmp/stan.nrow());
            diffSigma = abs(sigma_tmp - sigma);
            sigma = sigma_tmp;
            iter++;
          }
          while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
          epsi_i = epsi;
        }        
        if(prebeta(i,j) == 0){
          for(size_t k = 0; k < stan.nrow(); k++){
            epsi_j[k] = preepsi.column(j)[k];
          }
        } else{
          double sigma = 1.0;
          double reg = 0.0;
          double diffBeta;
          double sigma_tmp = 0.0;
          double diffSigma = 0.0;
          vector<double> beta((stan.ncol() - 2), 0.0);
          vector<double> epsi(stan.nrow(), 0.0);
          // Extract IP_XX[-c(i,j),-c(i,j)] and IP_YX[j,-c(i,j)]
          vector< vector<double> > tmpxx( (stan.ncol() - 2), vector<double>((stan.ncol() - 2), 0.0) );
          vector<double> tmpyx((stan.ncol() - 2), 0.0);
          size_t t = 0;
          for (size_t k = 0; k < stan.ncol(); k++){
            if (k != i && k != j){
              tmpyx[t] = ipyx(j,k);
              size_t t1 = 0;
              for (size_t l = 0; l < stan.ncol(); l++){
                if ( l != i && l != j){
                  tmpxx[t1][t] = ipxx(l,k);
                  t1++;
                }
              }
              t++;
            }
          }
          // Extract x_stan[,-c(i,j)]
          vector< vector<double> > tmp_x( stan.nrow(), vector<double>((stan.ncol() - 2), 0.0) );
          t=0;
          for (size_t k = 0; k < stan.ncol(); k++){
            if (k != i && k != j){
              for(size_t m = 0; m < stan.nrow(); m++){
                tmp_x[m][t] = stan(m,k);
              }
              t++;
            }
          }
          size_t iter = 0;
          do {
            diffBeta = 0.0;
            //Update beta when fix sigma, Lasso regression
            reg = sigma * lambda;
            vector<double> beta_tmp(tmpyx.size(), 0.0);
            vector<double> gc(tmpyx.size(), 0.0);
            double dbm;
            do{
              dbm = 0;
              for(size_t jj = 0; jj < tmpyx.size(); jj++){
                double z = (tmpyx[jj] - gc[jj])/stan.nrow() + beta_tmp[jj];
                double beta_inlasso = max(0.0, z - reg) - max(0.0, -z - reg);
                double diffBeta_inlasso = beta_inlasso - beta_tmp[jj];
                double diffabs = abs(diffBeta_inlasso);
                if (diffabs > 0){
                  beta_tmp[jj] = beta_inlasso;
                  for (size_t ii = 0; ii < tmpyx.size(); ii++){
                    gc[ii] = gc[ii] + tmpxx[ii][jj] * diffBeta_inlasso;
                  }
                  dbm = max(dbm, diffabs);
                }
              }
            }
            while(dbm >= tol);
            for(size_t kk = 0; kk < beta_tmp.size(); kk++){
              diffBeta = max(diffBeta, abs(beta_tmp[kk] - beta[kk]));
              beta[kk] = beta_tmp[kk];
            }
            //Update sigma when fix beta
            vector<double> tmp_y(stan.nrow(), 0.0);
            for (size_t k = 0; k < stan.nrow(); k++){
              tmp_y[k] = inner_product(tmp_x[k].begin(), tmp_x[k].end(), beta.begin(), 0.0);
              epsi[k] = cent(k,j) - tmp_y[k];
            }
            vector<double> epsi2(stan.nrow(), 0.0);
            for(size_t kk = 0; kk < epsi.size(); kk++){
              epsi2[kk] = pow(epsi[kk],2);
            }
            sigma_tmp = accumulate(epsi2.begin(), epsi2.end(), 0.0);
            sigma_tmp = sqrt(sigma_tmp/stan.nrow());
            diffSigma = abs(sigma_tmp - sigma);
            sigma = sigma_tmp;
            iter++;
          }
          while((diffSigma >= tol || diffBeta >= tol) && iter < 10);
          epsi_j = epsi;
        }
        // Precision, solve(t(epsi_ij)%*%epsi_ij/n)
        vector< vector<double> > omega_tmp( 2, vector<double>(2, 0.0) );
        vector< vector<double> > omega( 2, vector<double>(2, 0.0) );
        omega_tmp[0][0] = inner_product(epsi_i.begin(), epsi_i.end(), epsi_i.begin(), 0.0)/stan.nrow();
        omega_tmp[1][1] = inner_product(epsi_j.begin(), epsi_j.end(), epsi_j.begin(), 0.0)/stan.nrow();
        omega_tmp[0][1] = inner_product(epsi_i.begin(), epsi_i.end(), epsi_j.begin(), 0.0)/stan.nrow();
        omega_tmp[1][0] = omega_tmp[0][1];
        // Inverse matrix of omega_tmp
        double tmp = omega_tmp[0][0] * omega_tmp[1][1] - omega_tmp[0][1] * omega_tmp[1][0];
        omega[0][0] = omega_tmp[1][1]/tmp;
        omega[0][1] = -omega_tmp[0][1]/tmp;
        omega[1][0] = -omega_tmp[1][0]/tmp;
        omega[1][1] = omega_tmp[0][0]/tmp;
        rmat1(i,i) = omega[0][0];
        rmat1(j,j) = omega[1][1];
        rmat1(i,j) = omega[0][1];
        rmat1(j,i) = omega[1][0];
        // P-value of precision
        double var_new = (pow(omega[0][1], 2) + omega[0][0] * omega[1][1]) / stan.nrow();
        double zscore = abs(omega[0][1]) / sqrt(var_new);
        rmat2(i,j) = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
        rmat2(j,i) = rmat2(i,j);
        // Partial correlation
        rmat3(i,j) = -omega[0][1] / sqrt(omega[0][0] * omega[1][1]);
        rmat3(j,i) = rmat3(i,j);
        // P-value of partial correlation
        var_new = pow( (1 - pow(rmat3(i,j), 2) ), 2) / stan.nrow();
        zscore = abs(rmat3(i,j)) / sqrt(var_new);
        rmat4(i,j) = 2 * Rf_pnorm5(-zscore, 0.0, 1.0, 1, 0);
        rmat4(j,i) = rmat4(i,j);
        // 95% confidence interval of partial correlation
        double z_95CI = 1.96; // z-score of 95% CI
        rmat5(i,j) = rmat3(i,j) - z_95CI * (1 - pow(rmat3(i,j),2))/sqrt(stan.nrow());
        rmat5(j,i) = rmat5(i,j);
        rmat6(i,j) = rmat3(i,j) + z_95CI * (1 - pow(rmat3(i,j),2))/sqrt(stan.nrow());
        rmat6(j,i) = rmat6(i,j);
      }
    }
  }
};

/* ***** Function 3 ******** */
// [[Rcpp::export]]
List FastGGM_Parallel(NumericMatrix x, double lambda=NA_REAL){
  cout << "Parallel computing FastGGM." << endl;
  // Stop threshold and parameter for edge decision
  double tol = 1e-3;
  //penalty parameter for L1 norm of betas
  if(NumericVector::is_na(lambda)){
    lambda = sqrt( 2 * log(x.ncol()/sqrt(x.nrow()))/x.nrow() );
    cout << "Use default lambda = sqrt(2*log(p/sqrt(n))/n)" << endl;
  }
  cout << "In this case, lambda = " << lambda << endl;
   
  // Center matrix
  cout << "Center each column." << endl;
  NumericMatrix x_cent(x.nrow(), x.ncol());
  Center center(x, x_cent); // declare an instance
  parallelFor(0, x.ncol(), center);  // call it with parallelFor
  
  // Standardize matrix
  cout << "Standardize each column." << endl;
  NumericMatrix x_stan(x.nrow(), x.ncol());
  Standardize standardize(x_cent, x_stan); 
  parallelFor(0, x.ncol(), standardize);
  
  // Pre-calculate inner product matrixes
  cout << "Pre-calculate inner product matrixes." << endl;
  NumericMatrix IP_YX(x.ncol(), x.ncol());
  InnerProduct innerProduct1(x_cent, x_stan, IP_YX); 
  parallelFor(0, x.ncol(), innerProduct1);
  NumericMatrix IP_XX(x.ncol(), x.ncol());
  InnerProduct innerProduct2(x_stan, x_stan, IP_XX); 
  parallelFor(0, x.ncol(), innerProduct2);
  
  // Pre-calculate scaled Lasso for each variable
  cout << "Pre-calculate scaled Lasso for each variable." << endl;
  NumericMatrix beta_pre(x.ncol(), x.ncol());
  NumericMatrix epsi_pre(x.nrow(), x.ncol());
  PreSLasso preSLasso(IP_YX, IP_XX, x_stan, x_cent, lambda, tol, beta_pre, epsi_pre);
  parallelFor(0, x.ncol(), preSLasso);
  
  // Fast pairwise GGM based on the pre-calculated scaled Lasso
  cout << "Perform pairwise GGM." << endl;
  NumericMatrix precision(x.ncol(), x.ncol());
  NumericMatrix p_precision(x.ncol(), x.ncol());
  NumericMatrix partialCor(x.ncol(), x.ncol());
  NumericMatrix p_partialCor(x.ncol(), x.ncol());
  NumericMatrix CI_low_parCor(x.ncol(), x.ncol());
  NumericMatrix CI_high_parCor(x.ncol(), x.ncol());
  PairGGM pairGGM(beta_pre, epsi_pre, IP_YX, IP_XX, x_stan, x_cent, lambda, tol, precision, p_precision, partialCor, p_partialCor, CI_low_parCor, CI_high_parCor); 
  parallelFor(0, (x_cent.ncol() - 1), pairGGM);
  
  return List::create(_["precision"] = precision, _["partialCor"] = partialCor, _["p_precision"] = p_precision, _["p_partialCor"] = p_partialCor, _["CI_low_parCor"] = CI_low_parCor, _["CI_high_parCor"] = CI_high_parCor);
}
