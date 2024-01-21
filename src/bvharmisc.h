#ifndef BVHARMISC_H
#define BVHARMISC_H

typedef Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMajorMatrixXd;

Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean);

Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max);

Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max);

Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> vectorize_eigen(const Eigen::MatrixBase<Derived>& x) {
	// should use x.eval() when x is expression such as block or transpose. Use matrix().eval() if array.
	return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1>::Map(x.derived().data(), x.size());
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> unvectorize(const Eigen::MatrixBase<Derived>& x, int num_cols) {
	int num_rows = x.size() / num_cols;
	return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Map(x.derived().data(), num_rows, num_cols);
}

double mgammafn(double x, int p);

double log_mgammafn(double x, int p);

double invgamma_dens(double x, double shp, double scl, bool lg);

double compute_logml(int dim, int num_design, Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale, int posterior_shape);

#endif
