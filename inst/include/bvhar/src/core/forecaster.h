#ifndef BVHAR_CORE_FORECASTER_H
#define BVHAR_CORE_FORECASTER_H

#include "./common.h"
#include "./omp.h"

namespace bvhar {

template <typename ReturnType, typename InputType, typename DataType, typename XType> class MultistepForecaster;
template <typename ReturnType, typename InputType, typename DataType, typename XType> class MultistepForecastRun;

/**
 * @brief Base class for Recursive multi-step forecasting
 * 
 * @tparam ReturnType Type of forecasting result.
 * @tparam InputType Type of input data.
 * @tparam DataType Type of one unit.
 * @tparam XType Type of predictor.
 */
template <typename ReturnType = Eigen::MatrixXd, typename InputType = ReturnType, typename DataType = Eigen::VectorXd, typename XType = DataType>
class MultistepForecaster {
public:
	MultistepForecaster(int step, const InputType& response, int lag)
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

	virtual DataType getLastForecast() = 0;
	virtual DataType getLastForecast(const DataType& valid_vec) = 0;

protected:
	int step, lag;
	InputType response;
	ReturnType pred_save; // when Point: rbind(step) or when Density: rbind(step), cbind(sims)
	DataType point_forecast; // y_(T + h - 1)
	XType last_pvec; // [ y_(T + h - 1)^T, y_(T + h - 2)^T, ..., y_(T + h - p)^T, 1 ] (1 when constant term)
	XType tmp_vec; // y_(T + h - 2), ... y_(T + h - lag)

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
 * @tparam InputType 
 * @tparam DataType 
 * @tparam XType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename InputType = ReturnType, typename DataType = Eigen::VectorXd, typename XType = DataType>
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