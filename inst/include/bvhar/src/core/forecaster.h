#ifndef BVHAR_CORE_FORECASTER_H
#define BVHAR_CORE_FORECASTER_H

#include "./common.h"
#include "./omp.h"

namespace bvhar {

template <typename ReturnType, typename DataType> class MultistepForecaster;
template <typename ReturnType, typename DataType> class MultistepForecastRun;

/**
 * @brief Base class for Recursive multi-step forecasting
 * 
 * @tparam ReturnType Type of forecasting result.
 * @tparam DataType Type of one unit.
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class MultistepForecaster {
public:
	MultistepForecaster(int step, const ReturnType& response, int lag)
	: step(step), lag(lag), response(response) {}
	virtual ~MultistepForecaster() = default;

	/**
	 * @brief Return the forecast result.
	 * Bayes methods return density forecast while frequentist methods return point forecast.
	 * 
	 * @return ReturnType 
	 */
	ReturnType doForecast() {
		forecast();
		return pred_save;
	}

	ReturnType doForecast(const DataType& valid_vec) {
		forecast(valid_vec);
		return pred_save;
	}

	virtual ReturnType getLastForecast() = 0;
	virtual ReturnType getLastForecast(const DataType& valid_vec) = 0;

protected:
	int step, lag;
	ReturnType response;
	ReturnType pred_save; // when Point: rbind(step) or when Density: rbind(step), cbind(sims)
	DataType point_forecast; // y_(T + h - 1)
	DataType last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	DataType tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)

	/**
	 * @brief Initialize lagged predictors 'point_forecast', 'last_pvec' and 'tmp_vec'.
	 * 
	 */
	virtual void initLagged() = 0;

	/**
	 * @brief Multi-step forecasting
	 * 
	 */
	virtual void forecast() = 0;

	virtual void forecast(const DataType& valid_vec) = 0;

	/**
	 * @brief Set the initial lagged unit
	 * 
	 */
	virtual void setRecursion() = 0;

	/**
	 * @brief Move the lagged unit element based on one-step ahead forecasting
	 * 
	 */
	virtual void updateRecursion() = 0;
	// virtual void forecastOut(const int i) = 0;

	/**
	 * @brief Compute Linear predictor
	 * 
	 */
	virtual void updatePred(const int h, const int i) = 0;
};

/**
 * @brief Base class for multi-step forecasting runner
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class MultistepForecastRun {
public:
	MultistepForecastRun() {}
	virtual ~MultistepForecastRun() = default;

	/**
	 * @brief Forecast
	 * 
	 */
	virtual void forecast() = 0;
};

} // namespace bvhar

#endif // BVHAR_CORE_FORECASTER_H