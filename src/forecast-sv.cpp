#include "svforecaster.h"
#include "bvharinterrupt.h"

//' Forecasting predictive density of VAR-SV
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param alpha_record MCMC record of coefficients
//' @param h_last_record MCMC record of log-volatilities in last time
//' @param a_record MCMC record of contemporaneous coefficients
//' @param sigh_record MCMC record of variance of log-volatilities
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
                           Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
													 Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVarForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			alpha_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i].reset(new bvhar::SvVarForecaster(
			sv_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting sparse predictive density of VAR-SV with SSVS prior
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_ssvs_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
                           			Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
																Eigen::MatrixXd gamma_record, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVarSparseForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			alpha_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		bvhar::SsvsRecords ssvs_record(
			gamma_record.middleRows(i * num_sim, num_sim),
			Eigen::MatrixXd(),
			Eigen::MatrixXd(),
			Eigen::MatrixXd()
		);
		forecaster[i].reset(new bvhar::SvVarSparseForecaster(
			sv_record, ssvs_record,
			step, response_mat, var_lag, include_mean,
			static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting sparse predictive density of VAR-SV with horseshoe prior
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_hs_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
                           		Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
															Eigen::MatrixXd kappa_record, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVarSparseForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			alpha_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		bvhar::HorseshoeRecords hs_record(
			Eigen::MatrixXd(),
			Eigen::MatrixXd(),
			kappa_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i].reset(new bvhar::SvVarSparseForecaster(
			sv_record, hs_record,
			step, response_mat, var_lag, include_mean,
			static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting Predictive Density of VHAR-SV
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param HARtrans VHAR linear transformation matrix
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
														Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
														Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVharForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			phi_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i].reset(new bvhar::SvVharForecaster(
			sv_record, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting sparse predictive density of VHAR-SV with SSVS prior
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_ssvs_bvharsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
                           			 Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
																 Eigen::MatrixXd gamma_record, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVharSparseForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			phi_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		bvhar::SsvsRecords ssvs_record(
			gamma_record.middleRows(i * num_sim, num_sim),
			Eigen::MatrixXd(),
			Eigen::MatrixXd(),
			Eigen::MatrixXd()
		);
		forecaster[i].reset(new bvhar::SvVharSparseForecaster(
			sv_record, ssvs_record,
			step, response_mat, HARtrans, month, include_mean,
			static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting sparse predictive density of VHAR-SV with horseshoe prior
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_hs_bvharsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
                           		 Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
															 Eigen::MatrixXd kappa_record, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVharSparseForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			phi_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		bvhar::HorseshoeRecords hs_record(
			Eigen::MatrixXd(),
			Eigen::MatrixXd(),
			kappa_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i].reset(new bvhar::SvVharSparseForecaster(
			sv_record, hs_record,
			step, response_mat, HARtrans, month, include_mean,
			static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 bool include_mean, int step, Eigen::MatrixXd y_test,
											 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVarForecaster(
				*sv_record, step, roll_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::Hierminnparams minn_params(
					num_iter, design, roll_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of sparse VAR-SV based on Rolling Window
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_sparse_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 				Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 				Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
															bool include_mean, int step, Eigen::MatrixXd y_test,
											 				bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	// std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarSparseForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			if (prior_type == 1) {
				Rcpp::stop("not specified");
			} else if (prior_type == 2) {
				Rcpp::List gamma_list = fit_record["gamma_record"];
				bvhar::SsvsRecords ssvs_record(
					Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Eigen::MatrixXd()
				);
				forecaster[0][i].reset(new bvhar::SvVarSparseForecaster(
					*sv_record, ssvs_record, step, roll_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else if (prior_type == 3) {
				Rcpp::List kappa_list = fit_record["kappa_record"];
				bvhar::HorseshoeRecords hs_record(
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
				);
				forecaster[0][i].reset(new bvhar::SvVarSparseForecaster(
					*sv_record, hs_record, step, roll_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else {
				Rcpp::stop("Wrong 'prior_type'");
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::Hierminnparams minn_params(
					num_iter, design, roll_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				if (prior_type == 2) {
					bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
						sv_record, ssvs_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				} else if (prior_type == 3) {
					bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
						sv_record, hs_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				}
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		if (prior_type == 2) {
			bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
				sv_record, ssvs_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else if (prior_type == 3) {
			bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
				sv_record, hs_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		// forecaster[window][chain].reset(new bvhar::SvVarForecaster(
		// 	sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		// ));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												bool include_mean, int step, Eigen::MatrixXd y_test,
											  bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVharForecaster(
				*sv_record, step, roll_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::Hierminnparams minn_params(
					num_iter, design, roll_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVharForecaster(
					sv_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVharForecaster(
			sv_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
	if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of sparse VHAR-SV based on Rolling Window
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_sparse_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  			 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  			 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
															 bool include_mean, int step, Eigen::MatrixXd y_test,
											  			 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharSparseForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			if (prior_type == 1) {
				Rcpp::stop("not specified");
			} else if (prior_type == 2) {
				Rcpp::List gamma_list = fit_record["gamma_record"];
				bvhar::SsvsRecords ssvs_record(
					Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Eigen::MatrixXd()
				);
				forecaster[0][i].reset(new bvhar::SvVharSparseForecaster(
					*sv_record, ssvs_record, step, roll_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else if (prior_type == 3) {
				Rcpp::List kappa_list = fit_record["kappa_record"];
				bvhar::HorseshoeRecords hs_record(
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
				);
				forecaster[0][i].reset(new bvhar::SvVharSparseForecaster(
					*sv_record, hs_record, step, roll_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else {
				Rcpp::stop("Wrong 'prior_type'");
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::Hierminnparams minn_params(
					num_iter, design, roll_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				if (prior_type == 2) {
					bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
						sv_record, ssvs_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				} else if (prior_type == 3) {
					bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
						sv_record, hs_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				}
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		if (prior_type == 2) {
			bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
				sv_record, ssvs_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else if (prior_type == 3) {
			bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
				sv_record, hs_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
	if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 bool include_mean, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVarForecaster(
				*sv_record, step, expand_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::Hierminnparams minn_params(
					num_iter, design, expand_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of Sparse VAR-SV based on Expanding Window
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_sparse_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 	 				Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 	 				Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																bool include_mean, int step, Eigen::MatrixXd y_test,
											 	 				bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarSparseForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			// forecaster[0][i].reset(new bvhar::SvVarForecaster(
			// 	*sv_record, step, expand_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			// ));
			if (prior_type == 1) {
				Rcpp::stop("not specified");
			} else if (prior_type == 2) {
				Rcpp::List gamma_list = fit_record["gamma_record"];
				bvhar::SsvsRecords ssvs_record(
					Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Eigen::MatrixXd()
				);
				forecaster[0][i].reset(new bvhar::SvVarSparseForecaster(
					*sv_record, ssvs_record, step, expand_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else if (prior_type == 3) {
				Rcpp::List kappa_list = fit_record["kappa_record"];
				bvhar::HorseshoeRecords hs_record(
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
				);
				forecaster[0][i].reset(new bvhar::SvVarSparseForecaster(
					*sv_record, hs_record, step, expand_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else {
				Rcpp::stop("Wrong 'prior_type'");
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::Hierminnparams minn_params(
					num_iter, design, expand_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				if (prior_type == 2) {
					bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
						sv_record, ssvs_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				} else if (prior_type == 3) {
					bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
						sv_record, hs_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				}
				// forecaster[window][chain].reset(new bvhar::SvVarForecaster(
				// 	sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				// ));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		if (prior_type == 2) {
			bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
				sv_record, ssvs_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else if (prior_type == 3) {
			bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVarSparseForecaster(
				sv_record, hs_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		// forecaster[window][chain].reset(new bvhar::SvVarForecaster(
		// 	sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		// ));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  	Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVharForecaster(
				*sv_record, step, expand_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::Hierminnparams minn_params(
					num_iter, design, expand_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVharForecaster(
					sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVharForecaster(
			sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}

//' Out-of-Sample Forecasting of Sparse VHAR-SV based on Expanding Window
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_sparse_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  				 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  				 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																 bool include_mean, int step, Eigen::MatrixXd y_test,
											  				 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharSparseForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			if (prior_type == 1) {
				Rcpp::stop("not specified");
			} else if (prior_type == 2) {
				Rcpp::List gamma_list = fit_record["gamma_record"];
				bvhar::SsvsRecords ssvs_record(
					Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Eigen::MatrixXd()
				);
				forecaster[0][i].reset(new bvhar::SvVharSparseForecaster(
					*sv_record, ssvs_record, step, expand_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else if (prior_type == 3) {
				Rcpp::List kappa_list = fit_record["kappa_record"];
				bvhar::HorseshoeRecords hs_record(
					Eigen::MatrixXd(),
					Eigen::MatrixXd(),
					Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
				);
				forecaster[0][i].reset(new bvhar::SvVharSparseForecaster(
					*sv_record, hs_record, step, expand_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
				));
			} else {
				Rcpp::stop("Wrong 'prior_type'");
			}
			// forecaster[0][i].reset(new bvhar::SvVharForecaster(
			// 	*sv_record, step, expand_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
			// ));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain].reset(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::Hierminnparams minn_params(
					num_iter, design, expand_y0[window],
					param_sv,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierMinnInits minn_inits(init_spec);
					sv_objs[window][chain].reset(new bvhar::HierminnSv(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				if (prior_type == 2) {
					bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
						sv_record, ssvs_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				} else if (prior_type == 3) {
					bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
					forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
						sv_record, hs_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
					));
				}
				// forecaster[window][chain].reset(new bvhar::SvVharForecaster(
				// 	sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				// ));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		if (prior_type == 2) {
			bvhar::SsvsRecords ssvs_record = sv_objs[window][chain]->returnSsvsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
				sv_record, ssvs_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else if (prior_type == 3) {
			bvhar::HorseshoeRecords hs_record = sv_objs[window][chain]->returnHsRecords(num_burn, thinning);
			forecaster[window][chain].reset(new bvhar::SvVharSparseForecaster(
				sv_record, hs_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		// forecaster[window][chain].reset(new bvhar::SvVharForecaster(
		// 	sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		// ));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}
