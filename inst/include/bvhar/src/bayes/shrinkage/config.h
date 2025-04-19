#ifndef BVHAR_BAYES_SHRINKAGE_CONFIG_H
#define BVHAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"
#include "../../math/design.h"

namespace bvhar {

// Parameters
struct ShrinkageParams;
struct MinnParams2;
struct HierminnParams2;
struct SsvsParams2;
// struct HorseshoeParams2;
struct NgParams2;
struct DlParams2;
struct GdpParams2;
// Initialization
struct ShrinkageInits;
struct HierminnInits2;
struct SsvsInits2;
struct GlInits2;
struct HoreseshoeInits2;
struct NgInits2;
struct GdpInits2;
// MCMC records
struct ShrinkageRecords;
struct SsvsRecords2;
struct GlobalLocalRecords2;
struct HorseshoeRecords2;
struct NgRecords2;

struct ShrinkageParams {
	ShrinkageParams(LIST& priors) {}
};

struct MinnParams2 : public ShrinkageParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	
	MinnParams2(LIST& priors)
	: ShrinkageParams(priors) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
		Eigen::VectorXd _sigma = CAST<Eigen::VectorXd>(priors["sigma"]);
		double _lambda = CAST_DOUBLE(priors["lambda"]);
		double _eps = CAST_DOUBLE(priors["eps"]);
		int dim = _sigma.size();
		_prec_diag = Eigen::MatrixXd::Zero(dim, dim);
		Eigen::VectorXd _daily(dim);
		Eigen::VectorXd _weekly(dim);
		Eigen::VectorXd _monthly(dim);
		if (CONTAINS(priors, "delta")) {
			_daily = CAST<Eigen::VectorXd>(priors["delta"]);
			_weekly.setZero();
			_monthly.setZero();
		} else {
			_daily = CAST<Eigen::VectorXd>(priors["daily"]);
			_weekly = CAST<Eigen::VectorXd>(priors["weekly"]);
			_monthly = CAST<Eigen::VectorXd>(priors["monthly"]);
		}
		Eigen::MatrixXd dummy_response = build_ydummy(lag, _sigma, _lambda, _daily, _weekly, _monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			_lambda, _sigma, _eps, false
		);
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec.llt().solve(dummy_design.transpose() * dummy_response);
		_prec_diag.diagonal() = 1 / _sigma.array();
	}

	void initPrec(Eigen::Ref<Eigen::VectorXd> prior_alpha_mean, Eigen::Ref<Eigen::VectorXd> prior_alpha_prec, int num_alpha) {
		prior_alpha_mean.head(num_alpha) = _prior_mean.reshaped();
		prior_alpha_prec.head(num_alpha) = kronecker_eigen(_prec_diag, _prior_prec).diagonal();
		// if (include_mean) {
		// 	prior_alpha_mean.tail(dim) = BaseRegParams::_mean_non;
		// }
	}
};

struct HierminnParams2 : public MinnParams2 {
	double _shape, _rate;
	int _grid_size;
	// bool _minnesota;

	HierminnParams2(LIST& priors)
	: MinnParams2(priors),
		_shape(CAST_DOUBLE(priors["shape"])), _rate(CAST_DOUBLE(priors["rate"])), _grid_size(CAST_INT(priors["grid_size"])) {}
};

struct SsvsParams2 : public ShrinkageParams {
	Eigen::VectorXd _s1, _s2;
	double _slab_shape, _slab_scl;
	int _grid_size;

	SsvsParams2(LIST& priors)
	: ShrinkageParams(priors),
		_s1(CAST<Eigen::VectorXd>(priors["coef_s1"])), _s2(CAST<Eigen::VectorXd>(priors["coef_s2"])),
		_slab_shape(CAST_DOUBLE(priors["slab_shape"])), _slab_scl(CAST_DOUBLE(priors["_slab_scl"])),
		_grid_size(CAST_INT(priors["grid_size"])) {}
};

// struct HorseshoeParams2 : public ShrinkageParams {
// 	HorseshoeParams() {}
// };

struct NgParams2 : public ShrinkageParams {
	double _mh_sd, _group_shape, _group_scl, _global_shape, _global_scl;

	NgParams2(LIST& priors)
	: ShrinkageParams(priors),
		_mh_sd(CAST_DOUBLE(priors["shape_sd"])),
		_group_shape(CAST_DOUBLE(priors["group_shape"])), _group_scl(CAST_DOUBLE(priors["group_scale"])),
		_global_shape(CAST_DOUBLE(priors["global_shape"])), _global_scl(CAST_DOUBLE(priors["global_scale"])) {}
};

struct DlParams2 : public ShrinkageParams {
	int _grid_size;
	double _shape, _scl;

	DlParams2(LIST& priors)
	: ShrinkageParams(priors), _grid_size(CAST_INT(priors["grid_size"])), _shape(CAST_DOUBLE(priors["shape"])), _scl(CAST_DOUBLE(priors["scale"])) {}
};

struct GdpParams2 : public ShrinkageParams {
	int _grid_shape, _grid_rate;

	GdpParams2(LIST& priors)
	: ShrinkageParams(priors), _grid_shape(CAST_INT(priors["grid_shape"])), _grid_rate(CAST_INT(priors["grid_rate"])) {}
};

struct ShrinkageInits {
	ShrinkageInits(LIST& init) {}
	ShrinkageInits(LIST& init, int num_design) {}
};

struct HierminnInits2 : public ShrinkageInits {
	double _own_lambda;
	double _cross_lambda;

	HierminnInits2(LIST& init)
	: ShrinkageInits(init), _own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])) {}

	HierminnInits2(LIST& init, int num_design)
	: ShrinkageInits(init, num_design), _own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])) {}

	void initPrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec, int num_alpha,
		Optional<Eigen::VectorXi> grp_vec = NULLOPT, Optional<Eigen::VectorXi> cross_id = NULLOPT
	) {
		prior_prec.head(num_alpha).array() /= _own_lambda;
		if (grp_vec && cross_id) {
			// for (int i = 0; i < num_alpha; ++i) {
			// 	if (cross_id.find(grp_vec[i]) != cross_id.end()) {
			// 		prior_prec[i] /= _cross_lambda; // nu
			// 	}
			// }
			Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
			for (int i = 0; i < cross_id->size(); ++i) {
				global_id = grp_vec->array() == (*cross_id)[i];
				for (int j = 0; j < num_alpha; ++j) {
					if (global_id[j]) {
						prior_prec[j] /= _cross_lambda; // nu
					}
				}
			}
		}
	}
};

struct SsvsInits2 : public ShrinkageInits {
	Eigen::VectorXd _dummy, _weight, _slab;
	double _spike_scl;

	SsvsInits2(LIST& init)
	: ShrinkageInits(init),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
	
	SsvsInits2(LIST& init, int num_design)
	: ShrinkageInits(init),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
};

struct GlInits2 : public ShrinkageInits {
	Eigen::VectorXd _local;
	double _global;

	GlInits2(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
	
	GlInits2(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
};

struct HorseshoeInits2 : public GlInits2 {
	Eigen::VectorXd _group;

	HorseshoeInits2(LIST& init)
	: GlInits2(init),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
	
	HorseshoeInits2(LIST& init, int num_design)
	: GlInits2(init, num_design),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

struct NgInits2 : public HorseshoeInits2 {
	Eigen::VectorXd _local_shape;

	NgInits2(LIST& init)
	: HorseshoeInits2(init),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
	
	NgInits2(LIST& init, int num_design)
	: HorseshoeInits2(init, num_design),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
};

struct GdpInits2 : public ShrinkageInits {
	Eigen::VectorXd _local, _group_rate;
	double _gamma_shape, _gamma_rate;

	GdpInits2(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
	
	GdpInits2(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
};

struct ShrinkageRecords {
	// 
};

} // namespace bvhar

#endif // BVHAR_BAYES_SHRINKAGE_CONFIG_H