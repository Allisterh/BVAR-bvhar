#include <RcppEigen.h>
#include "bvharmisc.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of \code{\link{var_lm}}
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try \code{matrix(0L, nrow = var_lag, ncol = dim)}.
//' @details
//' Generate \deqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
//' 
//' For i = 1, ... n,
//' 
//' \eqn{y_{p + i} = (y_{p + i - 1}^T, \ldots, y_i^T, 1)^T B + \epsilon_i}
//' 
//' Then the output is \deqn{(y_{p + 1}, \ldots, y_{n + p})^T}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_var(int num_sim, int num_burn, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error, Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = dim * var_lag + 1; // k = mp + 1 if const_term exists
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (var_coef.cols() != dim) Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  if (var_coef.rows() != dim_design) Rcpp::stop("`var_coef` is not VAR coefficient. Check its dimension.");
  if (!(init.rows() == var_lag && init.cols() == dim)) Rcpp::stop("`init` is `var_lag` x `dim` matrix in order of y1, y2, ..., yp.");
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd obs_p(1, dim_design); // row vector of X0: yp^T, ..., y1^T, 1
  obs_p(0, dim_design - 1) = 1.0; // for constant term
  for (int i = 0; i < var_lag; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(p + 1)^T to y(n + p)^T
  Eigen::VectorXd sig_mu = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mu, sig_error); // simulated error term: num_rand x m
  res.row(0) = obs_p * var_coef + error_term.row(0);
  for (int i = 1; i < num_rand; i++) {
    for (int t = 1; t < var_lag; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * var_coef + error_term.row(i);
  }
  return res.bottomRows(num_rand - num_burn);
}

//' Generate Stable VAR(p) Process
//' 
//' This function generates stable VAR(p)
//' 
//' @param num_sim Number to generated process
//' @param var_coef VAR coefficient. The format should be the same as the output of \code{\link{var_lm}}
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @details
//' Recall the relation between stable VAR(p) and VMA.
//' 
//' @seealso 
//' \code{\link{VARtoVMA}} computes VMA representation.
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_stable_var(int num_sim, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = dim * var_lag + 1; // k = mp + 1 if const_term exists
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (var_coef.cols() != dim) Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  if (var_coef.rows() != dim_design) Rcpp::stop("`var_coef` is not VAR coefficient. Check its dimension.");
  Eigen::VectorXd const_term = var_coef.row(dim_design - 1); // constant term = the last row of the coefficient matrix
  Eigen::VectorXd sig_mean(dim); // N(mu, sig)
  Eigen::MatrixXd Im(dim, dim);
  Im.setIdentity(dim, dim); // identity matrix
  for (int i = 0; i < var_lag; i++) {
    Im -= var_coef.block(dim * i, 0, dim, dim).adjoint(); // I - B1 - ... Bp
  }
  sig_mean = Im.inverse() * const_term; // mu = (I - B1 - ... - Bp)^{-1} const_term
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_mean, sig_error); // simulated error term: num_sim x m
  Eigen::MatrixXd ma_coef = VARcoeftoVMA(var_coef, var_lag, num_sim - 1); // VMA representation up to VMA(num_sim - 1): W0, ..., W(n - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_sim, dim); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint(); // Y1 = W0 epsilon(1)
  for (int i = 1; i < num_sim; i++) {
    for (int j = 0; j <= i; j++) {
      res.row(i) += ma_coef.block(dim * (i - j), 0, dim, dim) * error_term.row(j).adjoint(); // Yt = W0 et + W1 e(t - 1) + ... W(t - 1) e1
    }
  }
  return res;
}

//' Generate Multivariate Time Series Process Following VHAR
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of \code{\link{vhar_lm}}
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try \code{matrix(0L, nrow = 22L, ncol = dim)}.
//' @details
//' Generate \deqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
//' 
//' For i = 1, ... n,
//' 
//' \eqn{y_{22 + i} = (y_{21 + i}^T, \ldots, y_i^T, 1)^T T_{HAR}^T \Phi + \epsilon_i}
//' 
//' Then the output is \deqn{(y_{23}, \ldots, y_{n + 22})^T}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar(int num_sim, int num_burn, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error, Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = 3 * dim + 1;
  int var_design = 22 * dim + 1;
  if (vhar_coef.cols() != dim) Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (vhar_coef.rows() != dim_design) Rcpp::stop("`vhar_coef` is not VHAR coefficient. Check its dimension.");
  if (!(init.rows() == 22 && init.cols() == dim)) Rcpp::stop("`init` is 22 x `dim` matrix in order of y1, y2, ..., y22.");
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd hartrans_mat = scale_har(dim);
  Eigen::MatrixXd obs_p(1, var_design); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, var_design - 1) = 1.0; // for constant term
  for (int i = 0; i < 22; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(21 - i);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(23)^T to y(n + 22)^T
  Eigen::VectorXd sig_mu = Eigen::VectorXd::Zero(dim);
  Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mu, sig_error); // simulated error term: num_rand x m
  res.row(0) = obs_p * hartrans_mat.adjoint() * vhar_coef + error_term.row(0);
  for (int i = 1; i < num_rand; i++) {
    for (int t = 1; t < 22; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * hartrans_mat.adjoint() * vhar_coef + error_term.row(i);
  }
  return res.bottomRows(num_rand - num_burn);
}

//' Generate Stable VHAR Process
//' 
//' This function generates stable VHAR
//' 
//' @param num_sim Number to generated process
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of \code{\link{vhar_lm}}
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @details
//' Recall the relation between stable VAR(p) and VMA.
//' 
//' @seealso 
//' \code{\link{VHARtoVMA}} computes VMA representation.
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_stable_vhar(int num_sim, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = 3 * dim + 1;
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (vhar_coef.cols() != dim) Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  if (vhar_coef.rows() != dim_design) Rcpp::stop("`vhar_coef` is not VHAR coefficient. Check its dimension.");
  Eigen::MatrixXd hartrans_mat = scale_har(dim); // VHAR linear transformation
  Eigen::VectorXd const_term = vhar_coef.row(dim_design - 1); // constant term = the last row of the coefficient matrix
  Eigen::VectorXd sig_mean(dim); // N(mu, sig)
  Eigen::MatrixXd Im(dim, dim);
  Im.setIdentity(dim, dim); // identity matrix
  for (int i = 0; i < 3; i++) {
    Im -= vhar_coef.block(dim * i, 0, dim, dim).adjoint(); // I - Phi(d) - Phi(w) - Phi(m)
  }
  sig_mean = Im.inverse() * const_term; // mu = (I - Phi(d) - Phi(w) - Phi(m))^{-1} const_term
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_mean, sig_error); // simulated error term: num_sim x m
  Eigen::MatrixXd ma_coef = VHARcoeftoVMA(vhar_coef, hartrans_mat, num_sim - 1); // VMA representation up to VMA(num_sim - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_sim, dim); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint(); // Y1 = W0 epsilon(1)
  for (int i = 1; i < num_sim; i++) {
    for (int j = 0; j <= i; j++) {
      res.row(i) += ma_coef.block(dim * (i - j), 0, dim, dim) * error_term.row(j).adjoint(); // Yt = W0 et + W1 e(t - 1) + ... W(t - 1) e1
    }
  }
  return res;
}
