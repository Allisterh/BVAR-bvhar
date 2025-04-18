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
	ShrinkageUpdater() {}
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
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
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
	 * @brief Save MCMC records
	 * 
	 */
	virtual void updateRecords() = 0;
};

/**
 * @brief Minnesota prior
 * 
 */
class MinnUpdater : public ShrinkageUpdater {
public:
	MinnUpdater() {}
	virtual ~MinnUpdater() = default;
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {}
	void updateImpactPrec(
		Eigen::Ref<Eigen::VectorXd> prior_chol_prec,
		Eigen::Ref<Eigen::VectorXd> contem_coef,
		int num_lowerchol,
		BHRNG& rng
	) override {}
	void updateRecords() override {}
};

/**
 * @brief Hierarchical Minnesota prior
 * 
 */
class HierminnUpdater : public ShrinkageUpdater {
public:
	HierminnUpdater(
		const Eigen::VectorXd& prior_mean,
		const double own_lambda, const double own_shape, const double own_rate,
		const double cross_lambda, const double cross_shape, const double cross_rate,
		int grid_size
	)
	: prior_mean(prior_mean),
		own_lambda(own_lambda), own_shape(own_shape), own_rate(own_rate),
		cross_lambda(cross_lambda), cross_shape(cross_shape), cross_rate(cross_rate),
		grid_size(grid_size) {}
	virtual ~HierminnUpdater() = default;

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			coef_vec.head(num_alpha), prior_mean.head(num_alpha), prior_alpha_prec.head(num_alpha),
			rng
		);
		minnesota_nu_griddy(
			cross_lambda, grid_size,
			coef_vec.head(num_alpha), prior_mean.head(num_alpha), prior_alpha_prec.head(num_alpha),
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

	void updateRecords() override {}

private:
	Eigen::VectorXd prior_mean;
	double own_lambda, own_shape, own_rate;
	double cross_lambda, cross_shape, cross_rate;
	int grid_size;
};

/**
 * @brief Stochastic Search Variable Selection (SSVS) prior
 * 
 */
class SsvsUpdater : public ShrinkageUpdater {
public:
	SsvsUpdater(
		const double& ig_shape, const double& ig_scl, const int& grid_size,
    const Eigen::VectorXd& s1, const Eigen::VectorXd& s2,
    const Eigen::VectorXd& init_dummy, const Eigen::VectorXd& init_weight,
    const Eigen::VectorXd& init_slab, const double& init_spike_scl
	)
	: grid_size(grid_size), spike_scl(init_spike_scl), ig_shape(ig_shape), ig_scl(ig_scl), s1(s1), s2(s2),
		dummy(init_dummy), weight(init_weight), slab(init_slab), slab_weight(Eigen::VectorXd::Ones(slab.size())) {}
	virtual ~SsvsUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
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

	void updateRecords() override {}

private:
	int grid_size;
	double spike_scl; // scaling factor between 0 and 1: spike_sd = c * slab_sd
	double ig_shape, ig_scl; // IG hyperparameter for spike sd
	Eigen::VectorXd s1, s2; // Beta hyperparameter
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
	HorseshoeUpdater(
		const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_group, const double& init_global
	)
	: local_lev(init_local), group_lev(init_group), global_lev(isGroup ? init_global : 1.0),
		shrink_fac(Eigen::VectorXd::Zero(init_local.size())),
		latent_local(Eigen::VectorXd::Zero(init_local.size())),
		latent_group(Eigen::VectorXd::Zero(init_group.size())),
		latent_global(0.0),
		coef_var(Eigen::VectorXd::Ones(init_local.size())) {}
	virtual ~HorseshoeUpdater() = default;

	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
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

	void updateRecords() override {}

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
	NgUpdater(
		const double& mh_sd,
		const double& group_shape, const double& group_scl,
		const double& global_shape, const double& global_scl,
		const Eigen::VectorXd& init_local_shape,
		const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_group, const double& init_global
	)
	: mh_sd(mh_sd), local_shape(init_local_shape),
		local_shape_fac(Eigen::VectorXd::Ones(init_local_shape.size())),
		group_shape(group_shape), group_scl(group_scl), global_shape(global_shape), global_scl(global_scl),
		local_lev(init_local), group_lev(init_group), global_lev(isGroup ? init_global : 1.0),
		coef_var(Eigen::VectorXd::Ones(init_local.size())) {}
	virtual ~NgUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
		BHRNG& rng
	) override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
		for (int j = 0; j < num_grp; j++) {
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
		group_lev[0] = ng_global_sparsity(coef_var, local_shape[0], group_shape, group_scl, rng);
		ng_local_sparsity(coef_var, local_shape[0], contem_coef, group_lev.replicate(1, num_lowerchol).reshaped(), rng);
		prior_chol_prec = 1 / local_lev.array().square();
	}

	void updateRecords() override {}

private:
	double mh_sd;
	Eigen::VectorXd local_shape, local_shape_fac;
	double group_shape, group_scl, global_shape, global_scl;
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
	DlUpdater(
		const int& grid_size, const double& shape, const double& scl,
		const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_group, const double& init_global
	)
	: dir_concen(0.0), shape(shape), scl(scl), grid_size(grid_size),
		local_lev(init_local), group_lev(init_group), global_lev(isGroup ? init_global : 1.0),
		latent_local(Eigen::VectorXd::Zero(init_local.size())),
		coef_var(Eigen::VectorXd::Zero(init_local.size())) {}
	virtual ~DlUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
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

	void updateRecords() override {}

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
	GdpUpdater(
		const int& grid_shape, const int& grid_rate,
		const double& gamma_shape, const double& gamma_rate,
		const Eiegn::VectorXd& group_rate,
		const Eigen::VectorXd& init_local
	)
	: group_rate(group_rate), group_rate_fac(Eigen::VectorXd::Ones(init_local.size())),
		gamma_shape(gamma_shape), gamma_rate(gamma_rate),
		shape_grid(grid_shape), rate_grid(grid_rate),
		local_lev(init_local) {}
	virtual ~GdpUpdater() = default;
	
	void updateCoefPrec(
		Eigen::Ref<Eigen::VectorXd> prior_alpha_prec,
		Eigen::Ref<const Eigen::VectorXd> coef_vec,
		int num_alpha, int num_grp,
		const Eigen::VectorXi& grp_vec, const Eigen::VectorXi& grp_id,
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

	void updateRecords() override {}

private:
	Eigen::VectorXd group_rate, group_rate_fac;
	double gamma_shape, gamma_rate;
	int shape_grid, rate_grid;
	Eigen::VectorXd local_lev;
};

} // namespace bvhar

#endif // BVHAR_BAYES_SHRINKAGE_SHRINKAGE_H