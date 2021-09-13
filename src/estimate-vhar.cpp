#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Build a Linear Transformation Matrix for Vector HAR
//' 
//' @param m integer, dimension
//' @details
//' VHAR is linearly restricted VAR(22) in Y0 = X0 B + Z.
//' 
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd scale_har (int m) {
  Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, 22);
  Eigen::MatrixXd HARtrans(3 * m + 1, 22 * m + 1); // 3m x 22m
  Eigen::MatrixXd Im(m, m);
  Im.setIdentity(m, m);
  HAR(0, 0) = 1.0;
  for (int i = 0; i < 5; i++) {
    HAR(1, i) = 1.0 / 5;
  }
  for (int i = 0; i < 22; i++) {
    HAR(2, i) = 1.0 / 22;
  }
  // T otimes Im
  HARtrans.block(0, 0, 3 * m, 22 * m) = Eigen::kroneckerProduct(HAR, Im).eval();
  HARtrans.block(0, 22 * m, 3 * m, 1) = Eigen::MatrixXd::Zero(3 * m, 1);
  HARtrans.block(3 * m, 0, 1, 22 * m) = Eigen::MatrixXd::Zero(1, 22 * m);
  HARtrans(3 * m, 22 * m) = 1.0;
  return HARtrans;
}

//' Compute Vector HAR Coefficient Matrices and Fitted Values
//' 
//' @param x X0 processed by \code{\link{build_design}}
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X1 Phi + Z
//' 
//' @references
//' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_har (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  int dim = y.cols();
  int num_har = 3 * dim + 1; // 3m + 1
  Eigen::MatrixXd x1(y.rows(), num_har); // HAR design matrix
  Eigen::MatrixXd Phi(num_har, dim); // HAR estimator
  Eigen::MatrixXd yhat(y.rows(), dim);
  Eigen::MatrixXd HARtrans = scale_har(dim); // linear transformation
  x1 = x * HARtrans.adjoint();
  Phi = (x1.adjoint() * x1).inverse() * x1.adjoint() * y; // estimation
  yhat = x1 * Phi;
  return Rcpp::List::create(
    Rcpp::Named("HARtrans") = HARtrans,
    Rcpp::Named("phihat") = Phi,
    Rcpp::Named("fitted") = yhat
  );
}

//' Compute Vector HAR Coefficient Matrices and Fitted Values without Constant Term
//' 
//' @param x X0 processed by \code{\link{build_design}} (delete its last column)
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X1 Phi + Z
//' 
//' @references
//' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_har_none (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  int dim = y.cols(); // m
  int num_har = 3 * dim; // 3m
  int dim_har = 22 * dim; // 22m
  Eigen::MatrixXd x1(y.rows(), num_har); // HAR design matrix
  Eigen::MatrixXd Phi(num_har, dim); // HAR estimator
  Eigen::MatrixXd yhat(y.rows(), dim);
  Eigen::MatrixXd HARtrans = scale_har(dim).block(0, 0, num_har, dim_har); // linear transformation
  x1 = x * HARtrans.adjoint();
  Phi = (x1.adjoint() * x1).inverse() * x1.adjoint() * y; // estimation
  yhat = x1 * Phi;
  return Rcpp::List::create(
    Rcpp::Named("HARtrans") = HARtrans,
    Rcpp::Named("phihat") = Phi,
    Rcpp::Named("fitted") = yhat
  );
}

//' @useDynLib bvhar
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max) {
  int dim = vhar_coef.cols(); // dimension of time series
  Eigen::MatrixXd coef_mat = HARtrans_mat.adjoint() * vhar_coef; // bhat = tilde(T)^T * Phi
  if (lag_max < 1) Rcpp::stop("'lag_max' must larger than 0");
  int ma_rows = dim * (lag_max + 1);
  int num_full_brows = ma_rows;
  if (lag_max < 22) num_full_brows = dim * 22; // for VMA coefficient q < VAR(p)
  Eigen::MatrixXd FullB = Eigen::MatrixXd::Zero(num_full_brows, dim); // same size with VMA coefficient matrix
  FullB.block(0, 0, dim * 22, dim) = coef_mat.block(0, 0, dim * 22, dim); // fill first mp row with VAR coefficient matrix
  Eigen::MatrixXd Im(dim, dim); // identity matrix
  Im.setIdentity(dim, dim);
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = FullB.block(0, 0, dim, dim) * ma.block(0, 0, dim, dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) return ma;
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += FullB.block(k * dim, 0, dim, dim) * ma.block((i - k - 1) * dim, 0, dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

//' Convert VHAR to VMA(infinite)
//' 
//' Convert VHAR process to infinite vector MA process
//' 
//' @param object \code{vharlse} object by \code{\link{vhar_lm}}
//' @param lag_max Maximum lag for VMA
//' @details
//' Let VAR(p) be stable
//' and let VAR(p) be
//' \eqn{Y_0 = X_0 B + Z}
//' 
//' VHAR is VAR(22) with
//' \deqn{Y_0 = X_1 B + Z = ((X_0 \tilde{T}^T)) \Phi + Z}
//' 
//' Observe that
//' \deqn{B = \tilde{T}^T \Phi}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VHARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("vharlse")) Rcpp::stop("'object' must be vharlse object.");
  Eigen::MatrixXd har_mat = object["coefficients"]; // Phihat(3m + 1, m) = [Phi(d)^T, Phi(w)^T, Phi(m)^T, c^T]^T
  Eigen::MatrixXd hartrans_mat = object["HARtrans"]; // tilde(T): (3m + 1, 22m + 1)
  Eigen::MatrixXd ma = VHARcoeftoVMA(har_mat, hartrans_mat, lag_max);
  return ma;
}

//' Compute Forecast MSE Matrices for VHAR
//' 
//' Compute the forecast MSE matrices using VMA coefficients
//' 
//' @param object \code{varlse} object by \code{\link{var_lm}}
//' @param step Integer, Step to forecast
//' @details
//' See pp38 of Lütkepohl (2007).
//' Let \eqn{\Sigma} be the covariance matrix of VHAR and let \eqn{W_j} be the VMA coefficients.
//' Recursively,
//' \deqn{\Sigma_y(1) = \Sigma}
//' \deqn{\Sigma_y(2) = \Sigma + W_1 \Sigma W_1^T}
//' \deqn{\Sigma_y(3) = \Sigma_y(2) + W_2 \Sigma W_2^T}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd compute_covmse_har(Rcpp::List object, int step) {
  if (!object.inherits("vharlse")) Rcpp::stop("'object' must be vharlse object.");
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd vma_mat = VHARtoVMA(object, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.block(0, 0, dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.block(i * dim, 0, dim, dim) = mse.block((i - 1) * dim, 0, dim, dim) + 
      vma_mat.block(i * dim, 0, dim, dim).adjoint() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
  }
  return mse;
}

