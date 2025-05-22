#ifndef BVHAR_OLS_FORECASTER_H
#define BVHAR_OLS_FORECASTER_H

// #include "../core/common.h"
#include "../core/forecaster.h"
#include "./ols.h"

namespace bvhar {

class OlsForecaster;
class VarForecaster;
class VharForecaster;

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
		return this->doForecast().bottomRows<1>();
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
		pred_save.row(h) = point_forecast.transpose();
	}

	virtual void computeMean() = 0;
};

class VarForecaster : public OlsForecaster {
public:
	VarForecaster(const OlsFit& fit, int step, const Eigen::MatrixXd& response_mat, bool include_mean)
	: OlsForecaster(fit, step, response_mat, include_mean) {}
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
	virtual ~VharForecaster() = default;

protected:
	void computeMean() override {
		point_forecast = last_pvec.transpose() * har_trans.transpose() * coef_mat; // y(T + h)^T = [yhat(T + h - 1)^T, ..., yhat(T + 1)^T, y(T)^T, ..., y(T + h - lag)^T] * C(HAR) * Ahat
	}

private:
	Eigen::MatrixXd har_trans;
};

} // namespace bvhar

#endif // BVHAR_OLS_FORECASTER_H