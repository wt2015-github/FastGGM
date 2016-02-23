// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// FastGGM
List FastGGM(NumericMatrix x, double lambda);
RcppExport SEXP FastGGM_FastGGM(SEXP xSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    __result = Rcpp::wrap(FastGGM(x, lambda));
    return __result;
END_RCPP
}
// FastGGM_edges
List FastGGM_edges(NumericMatrix x, NumericMatrix pairs, double lambda);
RcppExport SEXP FastGGM_FastGGM_edges(SEXP xSEXP, SEXP pairsSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type pairs(pairsSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    __result = Rcpp::wrap(FastGGM_edges(x, pairs, lambda));
    return __result;
END_RCPP
}
// FastGGM_Parallel
List FastGGM_Parallel(NumericMatrix x, double lambda);
RcppExport SEXP FastGGM_FastGGM_Parallel(SEXP xSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    __result = Rcpp::wrap(FastGGM_Parallel(x, lambda));
    return __result;
END_RCPP
}