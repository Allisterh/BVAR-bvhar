#ifndef BVHAR_OLS_FORECASTER_H
#define BVHAR_OLS_FORECASTER_H

// #include "../core/common.h"
#include "../core/forecaster.h"
#include "./ols.h"
#include <type_traits>

namespace bvhar {

class OlsExogenForecaster;
class OlsForecaster;
class VarForecaster;
class VharForecaster;
class OlsForecastRun;

class OlsExogenForecaster : public ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsExogenForecaster() {}
	OlsExogenForecaster(int lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef)
	: ExogenForecaster<Eigen::MatrixXd, Eigen::VectorXd>(lag, exogen),
		coef_mat(exogen_coef) {
		last_pvec = vectorize_eigen(exogen.topRows(lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
	}
	virtual ~OlsExogenForecaster() = default;

	void appendForecast(Eigen::VectorXd& point_forecast, const int h) override {
		last_pvec = vectorize_eigen(exogen.middleRows(h, lag + 1).colwise().reverse().transpose().eval()); // x_(T + h), ..., x_(T + h - s)
		// point_forecast += last_pvec.transpose() * coef_mat;
		point_forecast += coef_mat.transpose() * last_pvec;
	}

private:
	// int dim_exogen_design;
	Eigen::MatrixXd coef_mat;
};

class OlsForecaster : public MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, fit._ord),
		coef_mat(fit._coef), include_mean(include_mean), dim(coef_mat.cols()),
		dim_design(include_mean ? lag * dim + 1 : lag * dim) {
		initLagged();
	}
	OlsForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster>& exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, bool include_mean
	)
	: MultistepForecaster<Eigen::MatrixXd, Eigen::VectorXd>(step, response_mat, fit._ord),
		exogen_updater(std::move(exogen_updater)),
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
	std::unique_ptr<OlsExogenForecaster> exogen_updater;
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
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		pred_save.row(h) = point_forecast.transpose();
	}

	virtual void computeMean() = 0;
};

class VarForecaster : public OlsForecaster {
public:
	VarForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: OlsForecaster(fit, step, response_mat, include_mean) {}
	VarForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster>& exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, bool include_mean
	)
	: OlsForecaster(fit, exogen_updater, step, response_mat, include_mean) {}
	virtual ~VarForecaster() = default;

protected:
	void computeMean() override {
		point_forecast = last_pvec.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * Ahat
	}
};

class VharForecaster : public OlsForecaster {
public:
	VharForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean)
	: OlsForecaster(fit, step, response_mat, include_mean), har_trans(har_trans) {}
	VharForecaster(
		const OlsFit& fit, std::unique_ptr<OlsExogenForecaster>& exogen_updater,
		int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& har_trans, bool include_mean
	)
	: OlsForecaster(fit, exogen_updater, step, response_mat, include_mean), har_trans(har_trans) {}
	virtual ~VharForecaster() = default;

protected:
	void computeMean() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * C(HAR) * Ahat
	}

private:
	Eigen::MatrixXd har_trans;
};

class OlsForecastRun : public MultistepForecastRun<Eigen::MatrixXd, Eigen::VectorXd> {
public:
	OlsForecastRun(int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean) {
		bvhar::OlsFit ols_fit(coef_mat, lag);
		forecaster = std::make_unique<VarForecaster>(ols_fit, step, response_mat, include_mean);
	}
	OlsForecastRun(
		int lag, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		int exogen_lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	) {
		bvhar::OlsFit ols_fit(coef_mat, lag);
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(exogen_lag, exogen, exogen_coef);
		forecaster = std::make_unique<VarForecaster>(ols_fit, exogen_updater, step, response_mat, include_mean);
	}
	OlsForecastRun(int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		bvhar::OlsFit ols_fit(coef_mat, month);
		forecaster = std::make_unique<VharForecaster>(ols_fit, step, response_mat, har_trans, include_mean);
	}
	OlsForecastRun(
		int week, int month, int step, const Eigen::MatrixXd& response_mat, const Eigen::MatrixXd& coef_mat, bool include_mean,
		int exogen_lag, const Eigen::MatrixXd& exogen, const Eigen::MatrixXd& exogen_coef
	) {
		Eigen::MatrixXd har_trans = build_vhar(response_mat.cols(), week, month, include_mean);
		bvhar::OlsFit ols_fit(coef_mat, month);
		auto exogen_updater = std::make_unique<OlsExogenForecaster>(exogen_lag, exogen, exogen_coef);
		forecaster = std::make_unique<VharForecaster>(ols_fit, exogen_updater, step, response_mat, har_trans, include_mean);
	}
	virtual ~OlsForecastRun() = default;
	
	Eigen::MatrixXd returnForecast() {
		return forecaster->doForecast();
	}

protected:
	std::unique_ptr<OlsForecaster> forecaster;
};

} // namespace bvhar

#endif // BVHAR_OLS_FORECASTER_H