/**
 * @file bayes.h
 * @author your name (you@domain.com)
 * @brief MCMC
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef BVHAR_BAYES_BAYES_H
#define BVHAR_BAYES_BAYES_H

#include "../core/common.h"

namespace bvhar {

class McmcAlgo {
public:
	McmcAlgo(unsigned int seed) : mcmc_step(0), rng(seed) {}
	virtual ~McmcAlgo() = default;
	
	/**
	 * @brief MCMC warmup step
	 * 
	 */
	virtual void doWarmUp() = 0;

	/**
	 * @brief MCMC posterior sampling step
	 * 
	 */
	virtual void doPosteriorDraws() = 0;

	/**
	 * @brief Return posterior sampling records
	 * 
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return LIST `LIST` containing every MCMC draws
	 */
	virtual LIST returnRecords(int num_burn, int thin) = 0;

protected:
	std::mutex mtx;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain

	/**
	 * @brief Increment the MCMC step
	 * 
	 */
	void addStep() { ++mcmc_step; }
};

} // namespace bvhar

#endif // BVHAR_BAYES_BAYES_H