#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Compute VAR(p) Coefficient Matrices and Fitted Values
//' 
//' @param x X0 processed by \code{\link{build_design}}
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X0 B + Z
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP estimate_var (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd B(x.cols(), y.cols()); // bhat
  Eigen::MatrixXd yhat(y.rows(), y.cols());
  B = (x.adjoint() * x).inverse() * x.adjoint() * y;
  yhat = x * B;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(B),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat)
  );
}

//' Covariance Estimate for Residual Covariance Matrix
//' 
//' Plausible estimator for residual covariance.
//' 
//' @param z Matrix, residual
//' @param num_design Integer, s = n - p
//' @param dim_design Ingeger, k = mp + 1
//' @details
//' See Lütkepohl (2007).
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP compute_cov (Eigen::MatrixXd z, int num_design, int dim_design) {
  Eigen::MatrixXd cov_mat(z.cols(), z.cols());
  cov_mat = z.adjoint() * z / (num_design - dim_design);
  return Rcpp::wrap(cov_mat);
}

//' Convert VAR to VMA(infinite)
//' 
//' Convert VAR process to infinite vector MA process
//' 
//' @param object \code{varlse} object by \code{\link{var_lm}}
//' @param lag_max Maximum lag for VMA
//' @details
//' Let VAR(p) be stable.
//' \deqn{Y_t = c + \sum_{j = 0} W_j Z_{t - j}}
//' For VAR coefficient \eqn{B_1, B_2, \ldots, B_p},
//' \deqn{I = (W_0 + W_1 L + W_2 L^2 + \cdots + ) (I - B_1 L - B_2 L^2 - \cdots - B_p L^p)}
//' Recursively,
//' \deqn{W_0 = I}
//' \deqn{W_1 = W_0 B_1}
//' \deqn{W_2 = W_1 B_1 + W_0 B_2}
//' \deqn{W_j = \sum_{j = 1}^k W_{k - j} B_j}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP VARtoVMA(Rcpp::List object, int lag_max) {
  if (! object.inherits("varlse")) Rcpp::stop("'object' must be varlse object.");
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  int dim = object["m"]; // dimension of time series
  int p = object["p"];
  if (lag_max < p) Rcpp::stop("'lag_max' must larger than 'object$p'");
  // int ma_rows = dim * p;
  int ma_rows = dim * lag_max;
  Eigen::MatrixXd FullB = Eigen::MatrixXd::Zero(ma_rows, dim);
  FullB.block(0, 0, dim * p, dim) = coef_mat.block(0, 0, dim * p, dim);
  Eigen::MatrixXd Im(dim, dim); // identity matrix
  Im.setIdentity(dim, dim);
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T], lag_max = p
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = ma.block(0, 0, dim, dim) * FullB.block(0, 0, dim, dim); // W1 = W1 B1
  for (int i = 1; i < lag_max; i++) {
    for (int k = 1; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += ma.block((i - 1) * dim, 0, dim, dim) * FullB.block((k - 1) * dim, 0, dim, dim) +
        ma.block((i - 2) * dim, 0, dim, dim) * FullB.block(k * dim, 0, dim, dim);
    }
  }
  return Rcpp::wrap(ma);
}
