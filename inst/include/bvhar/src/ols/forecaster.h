#ifndef BVHAR_OLS_FORECASTER_H
#define BVHAR_OLS_FORECASTER_H

// #include "../core/common.h"
#include "../core/forecaster.h"
#include "./ols.h"
#include <type_traits>

namespace bvhar {

template <bool> class OlsForecaster;
template <bool> class VarForecaster;
template <bool> class VharForecaster;
class OlsForecastRun;

template <bool isExogen = false>
class OlsForecaster : public MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, fit._ord),
		coef_mat(fit._coef), include_mean(include_mean), dim(coef_mat.cols()),
		dim_design(include_mean ? lag * dim + 1 : lag * dim) {
		initLagged();
	}
	virtual ~OlsForecaster() = default;
	Eigen::MatrixXd forecastPoint() {
		return this->doForecast();
	}

	Eigen::VectorXd getLastForecast() override {
		return this->doForecast().template bottomRows<1>();
	}

protected:
	Eigen::MatrixXd coef_mat;
	bool include_mean;
	int dim;
	int dim_design;

	void initLagged() override {
		pred_save = Eigen::MatrixXd::Zero(step, dim);
		last_pvec = Eigen::VectorXd::Zero(dim_design);
		last_pvec[dim_design - 1] = 1.0;
		last_pvec.head(lag * dim) = vectorize_eigen(response.colwise().reverse().topRows(lag).transpose().eval()); // [y_T^T, y_(T - 1)^T, ... y_(T - lag + 1)^T]
		tmp_vec = last_pvec.segment(dim, (lag - 1) * dim); // y_(T - 1), ... y_(T - lag + 1)
		point_forecast = last_pvec.head(dim); // y_T
	}

	void setRecursion() override {
		last_pvec.segment(dim, (lag - 1) * dim) = tmp_vec;
		last_pvec.head(dim) = point_forecast;
	}

	void updateRecursion() override {
		tmp_vec = last_pvec.head((lag - 1) * dim);
	}

	void updatePred(const int h, const int i) override {
		computeMean();
		if (std::integral_constant<bool, isExogen>::value) {
			last_pvec.head(dim) = response.colwise().reverse().row(h); // x_(T + h)
		}
		pred_save.row(h) = point_forecast.transpose();
	}

	virtual void computeMean() = 0;
};

template <bool isExogen = false>
class VarForecaster : public OlsForecaster<isExogen> {
public:
	VarForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: OlsForecaster<isExogen>(fit, step, response_mat, include_mean) {}
	virtual ~VarForecaster() = default;

protected:
	using OlsForecaster<isExogen>::point_forecast;
	using OlsForecaster<isExogen>::last_pvec;
	using OlsForecaster<isExogen>::coef_mat;

	void computeMean() override {
		point_forecast = last_pvec.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * Ahat
	}
};

template <bool isExogen = false>
class VharForecaster : public OlsForecaster<isExogen> {
public:
	VharForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean)
	: OlsForecaster<isExogen>(fit, step, response_mat, include_mean), har_trans(har_trans) {}
	virtual ~VharForecaster() = default;

protected:
	using OlsForecaster<isExogen>::point_forecast;
	using OlsForecaster<isExogen>::last_pvec;
	using OlsForecaster<isExogen>::coef_mat;

	void computeMean() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * C(HAR) * Ahat
	}

private:
	Eigen::MatrixXd har_trans;
};

class OlsForecastRun : public MultistepForecastRun<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecastRun(int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean)
	: is_exogen(false) {
		bvhar::OlsFit ols_fit(coef_mat, lag);
		forecaster = std::make_unique<VarForecaster<false>>(ols_fit, step, response_mat, include_mean);
	}
	OlsForecastRun(
		int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		int exogen_lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	)
	: is_exogen(true) {
		bvhar::OlsFit ols_fit(coef_mat, lag);
		forecaster = std::make_unique<VarForecaster<false>>(ols_fit, step, response_mat, include_mean);
		bvhar::OlsFit exogen_fit(exogen_coef, exogen_lag);
		exogen_forecaster = std::make_unique<VarForecaster<true>>(exogen_fit, step, exogen, false);
	}
	OlsForecastRun(int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean)
	: is_exogen(false) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		bvhar::OlsFit ols_fit(coef_mat, month);
		forecaster = std::make_unique<VharForecaster<false>>(ols_fit, step, response_mat, har_trans, include_mean);
	}
	OlsForecastRun(
		int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	)
	: is_exogen(true) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		bvhar::OlsFit ols_fit(coef_mat, month);
		forecaster = std::make_unique<VharForecaster<false>>(ols_fit, step, response_mat, har_trans, include_mean);
		Eigen::MatrixXd exogen_har = build_vhar(exogen.cols(), week, month, false);
		bvhar::OlsFit exogen_fit(exogen_coef, month);
		exogen_forecaster = std::make_unique<VharForecaster<true>>(exogen_fit, step, exogen, exogen_har, false);
	}
	virtual ~OlsForecastRun() = default;
	
	Eigen::MatrixXd returnForecast() {
		if (is_exogen) {
			return forecaster->doForecast() + exogen_forecaster->doForecast();
		}
		return forecaster->doForecast();
	}

protected:
	std::unique_ptr<OlsForecaster<false>> forecaster;
	std::unique_ptr<OlsForecaster<true>> exogen_forecaster;
	bool is_exogen;
};

} // namespace bvhar

#endif // BVHAR_OLS_FORECASTER_H