#ifndef BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H
#define BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H

#include "./config.h"

namespace bvhar {

class ShrinkageUpdater;
// Shrinkage priors
class MinnUpdater;
class HierminnUpdater;
class SsvsUpdater;
template <bool isGroup> class HorseshoeUpdater;
template <bool isGroup> class NgUpdater;
template <bool isGroup> class DlUpdater;
template <bool isGroup> class GdpUpdater;

/**
 * @brief Draw class for shrinkage priors
 * 
 */
class ShrinkageUpdater {
public:
	ShrinkageUpdater(const ShrinkageParams& params, const ShrinkageInits& inits) {}
	virtual ~ShrinkageUpdater() = default;

	/**
	 * @brief Draw precision of coefficient based on each shrinkage priors
	 * 
	 * @param prior_alpha_prec Prior precision
	 * @param coef_vec Coefficient vector
	 * @param num_alpha Number of alpha
	 * @param num_grp Group number
	 * @param grp_vec Group vector
	 * @param grp_id Group id
	 * @param rng RNG
	 */
	virtual void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) = 0;

	/**
	 * @brief Draw precision of contemporaneous coefficient based on each shrinkage priors
	 * 
	 * @param prior_chol_prec Prior precision
	 * @param contem_coef Contemporaneous coefficient
	 * @param num_lowerchol Size of contemporaneous coefficient
	 * @param rng RNG
	 */
	virtual void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) = 0;

	/**
	 * @brief Append shrinkage prior's parameter record to the result `LIST`
	 * 
	 * @param list Contains MCMC record result
	 */
	virtual void appendRecords(LIST& list) = 0;
};

/**
 * @brief Minnesota prior
 * 
 */
class MinnUpdater : public ShrinkageUpdater {
public:
	MinnUpdater(const MinnParams& params, const ShrinkageInits& inits) : ShrinkageUpdater(params, inits) {}
	virtual ~MinnUpdater() = default;
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {}
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {}
	void appendRecords(LIST& list) override {}
};

/**
 * @brief Hierarchical Minnesota prior
 * 
 */
class HierminnUpdater : public ShrinkageUpdater {
public:
	HierminnUpdater(const HierminnParams& params, const HierminnInits& inits)
	: ShrinkageUpdater(params, inits),
		prior_mean(params._prior_mean.reshaped()),
		grid_size(params._grid_size),
		own_shape(params._shape), own_rate(params._rate),
		// cross_shape(params.shape), cross_rate(params.rate),
		own_lambda(inits._own_lambda), cross_lambda(inits._cross_lambda) {}
	virtual ~HierminnUpdater() = default;

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			coef_vec.head(num_alpha), prior_mean, prior_alpha_prec.head(num_alpha),
			rng
		);
		minnesota_nu_griddy(
			cross_lambda, grid_size,
			coef_vec.head(num_alpha), prior_mean, prior_alpha_prec.head(num_alpha),
			grp_vec, grp_id, rng
		);
	}
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			contem_coef, prior_mean, prior_chol_prec,
			rng
		);
	}

	void appendRecords(LIST& list) override {}

private:
	Eigen::VectorXd prior_mean;
	int grid_size;
	double own_shape, own_rate;
	double own_lambda, cross_lambda;
};

/**
 * @brief Stochastic Search Variable Selection (SSVS) prior
 * 
 */
class SsvsUpdater : public ShrinkageUpdater {
public:
	SsvsUpdater(const SsvsParams& params, const SsvsInits& inits)
	: ShrinkageUpdater(params, inits),
		grid_size(params._grid_size),
		ig_shape(params._slab_shape), ig_scl(params._slab_scl), s1(params._s1), s2(params._s2),
		spike_scl(inits._spike_scl), dummy(inits._dummy), weight(inits._weight), slab(inits._slab),
		slab_weight(Eigen::VectorXd::Ones(slab.size())) {}
	virtual ~SsvsUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		ssvs_local_slab(slab, dummy, coef_vec.head(num_alpha), ig_shape, ig_scl, spike_scl, rng);
		for (int j = 0; j < num_grp; ++j) {
			slab_weight = (grp_vec.array() == grp_id[j]).select(
				weight[j],
				slab_weight
			);
		}
		ssvs_scl_griddy(spike_scl, grid_size, coef_vec.head(num_alpha), slab, rng);
		ssvs_dummy(dummy, coef_vec.head(num_alpha), slab, spike_scl * slab, slab_weight, rng);
		ssvs_mn_weight(weight, grp_vec, grp_id, dummy, s1, s2, rng);
		prior_alpha_prec.head(num_alpha).array() = 1 / (spike_scl * (1 - dummy.array()) * slab.array() + dummy.array() * slab.array());
	}
	
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		ssvs_local_slab(slab, dummy, contem_coef, ig_shape, ig_scl, spike_scl, rng);
		ssvs_scl_griddy(spike_scl, grid_size, contem_coef, slab, rng);
		ssvs_dummy(dummy, contem_coef, slab, spike_scl * slab, weight, rng);
		ssvs_weight(weight, dummy, s1[0], s2[0], rng);
		prior_chol_prec = 1 / build_ssvs_sd(spike_scl * slab, slab, dummy).array().square();
	}

	void appendRecords(LIST& list) override {}

private:
	int grid_size;
	double ig_shape, ig_scl; // IG hyperparameter for spike sd
	Eigen::VectorXd s1, s2; // Beta hyperparameter
	double spike_scl; // scaling factor between 0 and 1: spike_sd = c * slab_sd
	Eigen::VectorXd dummy;
	Eigen::VectorXd weight;
	Eigen::VectorXd slab;
	Eigen::VectorXd slab_weight; // pij vector
};

/**
 * @brief Horseshoe prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class HorseshoeUpdater : public ShrinkageUpdater {
public:
	HorseshoeUpdater(const ShrinkageParams& params, const HorseshoeInits& inits)
	: ShrinkageUpdater(params, inits),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		shrink_fac(Eigen::VectorXd::Zero(local_lev.size())),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		latent_group(Eigen::VectorXd::Zero(group_lev.size())),
		latent_global(0.0),
		coef_var(Eigen::VectorXd::Ones(local_lev.size())) {}
	virtual ~HorseshoeUpdater() = default;

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec.head(num_alpha), 1, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		horseshoe_latent(latent_local, local_lev, rng);
		using is_group = std::integral_constant<bool, isGroup>;
		if (is_group::value) {
			horseshoe_latent(latent_global, global_lev, rng);
			global_lev = horseshoe_global_sparsity(latent_global, coef_var.array() * local_lev.array(), coef_vec.head(num_alpha), 1, rng);
		}
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), global_lev * global_lev, rng);
		prior_alpha_prec.head(num_alpha) = 1 / (global_lev * coef_var.array() * local_lev.array()).square();
		shrink_fac = 1 / (1 + prior_alpha_prec.head(num_alpha).array());
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		coef_var = group_lev.replicate(1, num_lowerchol).reshaped();
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, contem_coef, 1, rng);
		group_lev[0] = horseshoe_global_sparsity(latent_group[0], latent_local, contem_coef, 1, rng);
		prior_chol_prec.setZero();
		prior_chol_prec = 1 / (coef_var.array() * local_lev.array()).square();
	}

	void appendRecords(LIST& list) override {}

private:
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::VectorXd coef_var;
};

/**
 * @brief Normal-Gamma prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class NgUpdater : public ShrinkageUpdater {
public:
	NgUpdater(const NgParams& params, const NgInits& inits)
	: ShrinkageUpdater(params, inits),
		mh_sd(params._mh_sd),
		group_shape(params._group_shape), group_scl(params._group_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		local_shape(inits._local_shape),
		local_shape_fac(Eigen::VectorXd::Ones(inits._local.size())),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		coef_var(Eigen::VectorXd::Ones(local_lev.size())) {}
	virtual ~NgUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
		for (int j = 0; j < num_grp; ++j) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
			local_shape_fac = (grp_vec.array() == grp_id[j]).select(
				local_shape[j],
				local_shape_fac
			);
		}
		using is_group = std::integral_constant<bool, isGroup>;
		if (is_group::value) {
			global_lev = ng_global_sparsity(local_lev.array() / coef_var.array(), local_shape_fac, global_shape, global_scl, rng);
		}
		ng_local_sparsity(local_lev, local_shape_fac, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		prior_alpha_prec.head(num_alpha) = 1 / local_lev.array().square();
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		local_shape[0] = ng_shape_jump(local_shape[0], local_lev, group_lev[0], mh_sd, rng);
		group_lev[0] = ng_global_sparsity(local_lev, local_shape[0], group_shape, group_scl, rng);
		ng_local_sparsity(coef_var, local_shape[0], contem_coef, group_lev.replicate(1, num_lowerchol).reshaped(), rng);
		prior_chol_prec = 1 / local_lev.array().square();
	}

	void appendRecords(LIST& list) override {}

private:
	double mh_sd;
	double group_shape, group_scl, global_shape, global_scl;
	Eigen::VectorXd local_shape, local_shape_fac;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd coef_var;
};

/**
 * @brief Dirichlet-Laplace prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class DlUpdater : public ShrinkageUpdater {
public:
	DlUpdater(const DlParams& params, const HorseshoeInits& inits)
	: ShrinkageUpdater(params, inits),
		dir_concen(0.0), shape(params._shape), scl(params._scl), grid_size(params._grid_size),
		local_lev(inits._local), group_lev(inits._group), global_lev(isGroup ? inits._global : 1.0),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		coef_var(Eigen::VectorXd::Zero(local_lev.size())) {}
	virtual ~DlUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		dl_mn_sparsity(group_lev, grp_vec, grp_id, global_lev, local_lev, shape, scl, coef_vec.head(num_alpha), rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		dl_dir_griddy(dir_concen, grid_size, local_lev, global_lev, rng);
		dl_local_sparsity(local_lev, dir_concen, coef_vec.head(num_alpha).array() / coef_var.array(), rng);
		using is_group = std::integral_constant<bool, isGroup>;
		if (is_group::value) {
			global_lev = dl_global_sparsity(local_lev.array() * coef_var.array(), dir_concen, coef_vec.head(num_alpha), rng);
		}
		dl_latent(latent_local, global_lev * local_lev.array() * coef_var.array(), coef_vec.head(num_alpha), rng);
		prior_alpha_prec.head(num_alpha) = 1 / ((global_lev * local_lev.array() * coef_var.array()).square() * latent_local.array());
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		dl_dir_griddy(dir_concen, grid_size, local_lev, group_lev[0], rng);
		dl_local_sparsity(local_lev, dir_concen, contem_coef, rng);
		group_lev[0] = dl_global_sparsity(local_lev, dir_concen, contem_coef, rng);
		dl_latent(latent_local, local_lev, contem_coef, rng);
		prior_chol_prec = 1 / ((group_lev[0] * local_lev.array()).square() * latent_local.array());
	}

	void appendRecords(LIST& list) override {}

private:
	double dir_concen, shape, scl;
	int grid_size;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd coef_var;
};

/**
 * @brief Generalized Double Pareto (GDP) prior
 * 
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <bool isGroup = true>
class GdpUpdater : public ShrinkageUpdater {
public:
	GdpUpdater(const GdpParams& params, const GdpInits& inits)
	: ShrinkageUpdater(params, inits),
		shape_grid(params._grid_shape), rate_grid(params._grid_rate),
		group_rate(inits._group_rate), group_rate_fac(Eigen::VectorXd::Ones(inits._local.size())),
		gamma_shape(inits._gamma_shape), gamma_rate(inits._gamma_rate),
		local_lev(inits._local) {}
	virtual ~GdpUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		gdp_shape_griddy(gamma_shape, gamma_rate, shape_grid, coef_vec.head(num_alpha), rng);
		gdp_rate_griddy(gamma_rate, gamma_shape, rate_grid, coef_vec.head(num_alpha), rng);
		gdp_exp_rate(group_rate, gamma_shape, gamma_rate, coef_vec.head(num_alpha), grp_vec, grp_id, rng);
		for (int j = 0; j < num_grp; ++j) {
			group_rate_fac = (grp_vec.array() == grp_id[j]).select(
				group_rate[j],
				group_rate_fac
			);
		}
		gdp_local_sparsity(local_lev, group_rate_fac, coef_vec.head(num_alpha), rng);
		prior_alpha_prec.head(num_alpha) = 1 / local_lev.array();
	}

	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {
		gdp_shape_griddy(gamma_shape, gamma_rate, shape_grid, contem_coef, rng);
		gdp_rate_griddy(gamma_rate, gamma_shape, rate_grid, contem_coef, rng);
		gdp_exp_rate(group_rate, gamma_shape, gamma_rate, contem_coef, rng);
		gdp_local_sparsity(local_lev, group_rate, contem_coef, rng);
		prior_chol_prec = 1 / local_lev.array();
	}

	void appendRecords(LIST& list) override {}

private:
	int shape_grid, rate_grid;
	Eigen::VectorXd group_rate, group_rate_fac;
	double gamma_shape, gamma_rate;
	Eigen::VectorXd local_lev;
};

// template <typename ShrinkageType>
// struct ShrinkageParamsMap {
// 	using type = ShrinkageParams;
// };

// template <>
// struct ShrinkageParamsMap<MinnUpdater> {
// 	using type = MinnParams;
// };

// template <>
// struct ShrinkageParamsMap<HierminnUpdater> {
// 	using type = HierminnParams;
// };

// template <>
// struct ShrinkageParamsMap<SsvsUpdater> {
// 	using type = SsvsParams;
// };

// template <>
// struct ShrinkageParamsMap<HorseshoeUpdater<>> {
// 	using type = ShrinkageParams;
// };

// template <>
// struct ShrinkageParamsMap<NgUpdater<>> {
// 	using type = NgParams;
// };

// template <>
// struct ShrinkageParamsMap<DlUpdater<>> {
// 	using type = DlParams;
// };

// template <>
// struct ShrinkageParamsMap<GdpUpdater<>> {
// 	using type = GdpParams;
// };

// template <typename ShrinkageType>
// struct ShrinkageInitsMap {
// 	using type = ShrinkageInits;
// };

// template <>
// struct ShrinkageInitsMap<MinnUpdater> {
// 	using type = ShrinkageInits;
// };

// template <>
// struct ShrinkageInitsMap<HierminnUpdater> {
// 	using type = HierminnInits;
// };

// template <>
// struct ShrinkageInitsMap<SsvsUpdater> {
// 	using type = SsvsInits;
// };

// template <>
// struct ShrinkageInitsMap<HorseshoeUpdater<>> {
// 	using type = HorseshoeInits;
// };

// template <>
// struct ShrinkageInitsMap<NgUpdater<>> {
// 	using type = NgInits;
// };

// template <>
// struct ShrinkageInitsMap<DlUpdater<>> {
// 	using type = HorseshoeInits;
// };

// template <>
// struct ShrinkageInitsMap<GdpUpdater<>> {
// 	using type = GdpInits;
// };

/**
 * @brief Function to initialize `ShrinkageUpdater`
 * 
 * @tparam UPDATER Shrinkage priors
 * @tparam PARAMS Corresponding parameter struct
 * @tparam INITS Corresponding initialization struct
 * @param param_prior Shrinkage prior configuration
 * @param param_init Initial values
 * @return std::unique_ptr<ShrinkageUpdater> 
 */
template <bool isGroup = true>
inline std::unique_ptr<ShrinkageUpdater> initialize_shrinkageupdater(LIST& param_prior, LIST& param_init, int prior_type) {
	std::unique_ptr<ShrinkageUpdater> shrinkage_ptr;
	switch (prior_type) {
		case 1: {
			MinnParams params(param_prior);
			ShrinkageInits inits(param_init);
			shrinkage_ptr = std::make_unique<MinnUpdater>(params, inits);
			return shrinkage_ptr;
		}
		case 2: {
			SsvsParams params(param_prior);
			SsvsInits inits(param_init);
			shrinkage_ptr = std::make_unique<SsvsUpdater>(params, inits);
			return shrinkage_ptr;
		}
		case 3: {
			ShrinkageParams params(param_prior);
			HorseshoeInits inits(param_init);
			shrinkage_ptr = std::make_unique<HorseshoeUpdater<isGroup>>(params, inits);
			return shrinkage_ptr;
		}
		case 4: {
			HierminnParams params(param_prior);
			HierminnInits inits(param_init);
			shrinkage_ptr = std::make_unique<HierminnUpdater>(params, inits);
			return shrinkage_ptr;
		}
		case 5: {
			NgParams params(param_prior);
			NgInits inits(param_init);
			shrinkage_ptr = std::make_unique<NgUpdater<isGroup>>(params, inits);
			return shrinkage_ptr;
		}
		case 6: {
			DlParams params(param_prior);
			HorseshoeInits inits(param_init);
			shrinkage_ptr = std::make_unique<DlUpdater<isGroup>>(params, inits);
			return shrinkage_ptr;
		}
		case 7: {
			GdpParams params(param_prior);
			GdpInits inits(param_init);
			shrinkage_ptr = std::make_unique<GdpUpdater<isGroup>>(params, inits);
			return shrinkage_ptr;
		}
	}
	return shrinkage_ptr;
}

// template <typename UPDATER = ShrinkageUpdater>
// inline std::unique_ptr<ShrinkageUpdater> initialize_shrinkageupdater(LIST& param_prior, LIST& param_init) {
// 	std::unique_ptr<ShrinkageUpdater> shrinkage_ptr;
// 	using PARAMS = typename ShrinkageParamsMap<UPDATER>::type;
// 	using INITS = typename ShrinkageInitsMap<UPDATER>::type;
// 	PARAMS params(param_prior);
// 	INITS inits(param_init);
// 	shrinkage_ptr = std::make_unique<UPDATER>(params, inits);
// 	return shrinkage_ptr;
// }

} // namespace bvhar

#endif // BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H