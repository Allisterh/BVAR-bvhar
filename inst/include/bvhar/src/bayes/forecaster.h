#ifndef BVHAR_BAYES_FORECASTER_H
#define BVHAR_BAYES_FORECASTER_H

#include "../core/forecaster.h"
#include "./bayes.h"

namespace bvhar {

template <typename ReturnType, typename DataType> class BayesForecaster;
template <typename ReturnType, typename DataType> class McmcForecastRun;
template <typename ReturnType, typename DataType> class McmcOutforecastRun;

/**
 * @brief Base class for forecaster of Bayesian methods
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class BayesForecaster : public MultistepForecaster<ReturnType, DataType> {
public:
	BayesForecaster(int step, const ReturnType& response, int lag, int num_sim, unsigned int seed)
	: MultistepForecaster(step, response, lag),
		num_sim(num_sim), rng(seed) {}
	virtual ~BayesForecaster() = default;
	using MultistepForecaster<ReturnType, Datatype>::returnForecast();

protected:
	using MultistepForecaster<ReturnType, Datatype>::step;
	using MultistepForecaster<ReturnType, Datatype>::lag;
	using MultistepForecaster<ReturnType, Datatype>::response;
	using MultistepForecaster<ReturnType, Datatype>::pred_save; // rbind(step), cbind(sims)
	std::mutex mtx;
	int num_sim;
	BHRNG rng;

	void forecast() override {
		std::lock_guard<std::mutex> lock(mtx);
		DataType obs_vec = last_pvec; // y_T, y_(T - 1), ... y_(T - lag + 1)
		for (int i = 0; i < num_sim; ++i) {
			initRecursion(obs_vec);
			updateParams(i);
			forecastOut(i);
		}
	}

	/**
	 * @brief Initialize lagged predictor for each MCMC loop.
	 * 
	 * @param obs_vec 
	 */
	virtual void initRecursion(const DataType& obs_vec) = 0;

	/**
	 * @brief Update members with corresponding MCMC draw
	 * 
	 * @param i MCMC step
	 */
	virtual void updateParams(const int i) = 0;

	/**
	 * @brief Draw i-th forecast
	 * 
	 * @param i MCMC step
	 */
	void forecastOut(const int i) {
		for (int h = 0; h < step; ++h) {
			setRecursion();
			updatePred();
			updateRecursion();
		}
	}
};

/**
 * @brief Base runner class for MCMC forecaster
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class McmcForecastRun : public MultistepForecastRun<ReturnType, DataType> {
public:
	McmcForecastRun(int num_chains, int lag, int step)
	: MultistepForecastRun(), num_chains(num_chains), nthreads(nthreads) {}
	virtual ~McmcForecastRun() = default;

	/**
	 * @brief Forecast
	 * 
	 */
	void forecast() override {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			density_forecast[chain] = forecaster[chain]->returnForecast();
			forecaster[chain].reset();
		}
	}

	/**
	 * @brief Return forecast draws
	 * 
	 * @return std::vector<ReturnType> Forecast density of each chain
	 */
	std::vector<ReturnType> returnForecast() {
		forecast();
		return density_forecast;
	}

protected:
	std::vector<ReturnType> density_forecast;
	std::vector<std::unique_ptr<BayesForecaster<ReturnType, DataType>>> forecaster;

private:
	int num_chains, nthreads;
};

/**
 * @brief Base class for pseudo out-of-sample forecasting
 * 
 * @tparam ReturnType 
 * @tparam DataType 
 */
template <typename ReturnType = Eigen::MatrixXd, typename DataType = Eigen::VectorXd>
class McmcOutForecastRun {
public:
	McmcOutForecastRun(
		int num_window, int num_test, int num_horizon, int lag,
		int num_chains, int num_iter, int num_burn, int thin,
		int step, const ReturnType& y_test, bool get_lpl,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: num_window(num_window), num_test(num_test), num_horizon(num_horizon), step(step),
		lag(lag), num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		get_lpl(get_lpl), display_progress(display_progress),
		seed_forecast(seed_forecast),
		model(num_horizon), forecaster(num_horizon),
		out_forecast(num_horizon, std::vector<ReturnType>(num_chains)) {}
	virtual ~McmcOutForecastRun() = default;

	/**
	 * @brief Out-of-sample forecasting
	 * 
	 */
	virtual void forecast() = 0;

	/**
	 * @brief Return out-of-sample forecasting draws
	 * 
	 * @return LIST `LIST` containing forecast draws. Include ALPL when `get_lpl` is `true`.
	 */
	virtual LIST returnForecast() = 0;
	
private:
	int num_window, num_test, num_horizon, step;
	int lag, num_chains, num_iter, num_burn, thin, nthreads;
	boot get_lpl, display_progress;
	Eigen::VectorXi seed_forecast;
	std::vector<ReturnType> roll_mat;
	std::vector<ReturnType> roll_y0;
	ReturnType y_test;
	std::vector<std::vector<std::unique_ptr<McmcAlgo>>> model;
	std::vector<std::vector<std::unique_ptr<BayesForecaster>>> forecaster;
	std::vector<std::vector<ReturnType>> out_forecast;

	/**
	 * @brief Replace the forecast smart pointer given MCMC result
	 * 
	 * @param model MCMC model
	 * @param window Window index
	 * @param chain Chain index
	 */
	virtual void updateForecaster(std::vector<std::vector<std::unique_ptr<McmcAlgo>>>& model, int window, int chain) = 0;

	/**
	 * @brief Conduct MCMC and update forecast pointer
	 * 
	 * @param window Window index
	 * @param chain Chain index
	 */
	void runGibbs(int window, int chain) {
		std::string log_name = fmt::format("Chain {} / Window {}", chain + 1, window + 1);
		auto logger = spdlog::get(log_name);
		if (logger == nullptr) {
			logger = SPDLOG_SINK_MT(log_name);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		int logging_freq = num_iter / 20; // 5 percent
		if (logging_freq == 0) {
			logging_freq = 1;
		}
		for (int i = 0; i < num_burn; ++i) {
			model[window][chain]->doWarmUp();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Warmup)", i + 1, num_iter);
			}
		}
		logger->flush();
		for (int i = num_burn; i < num_iter; ++i) {
			model[window][chain]->doPosteriorDraws();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Sampling)", i + 1, num_iter);
			}
		}
		// RecordType reg_record = model[window][chain]->template returnStructRecords<RecordType>(0, thin, sparse);
		// updateForecaster(reg_record, window, chain);
		model[window][chain].reset();
		logger->flush();
		spdlog::drop(log_name);
	}
};

} // namespace bvhar

#endif // BVHAR_BAYES_FORECASTER_H