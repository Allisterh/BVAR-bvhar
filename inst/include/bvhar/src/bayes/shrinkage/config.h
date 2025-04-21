#ifndef BVHAR_BAYES_SHRINKAGE_CONFIG_H
#define BVHAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"
#include "../../math/design.h"

namespace bvhar {

// Parameters
struct ShrinkageParams;
struct MinnParams;
struct HierminnParams;
struct SsvsParams;
// struct HorseshoeParams;
struct NgParams;
struct DlParams;
struct GdpParams;
// Initialization
struct ShrinkageInits;
struct HierminnInits;
struct SsvsInits;
struct GlInits;
struct HorseshoeInits;
struct NgInits;
struct GdpInits;

struct ShrinkageParams {
	ShrinkageParams() {}
	ShrinkageParams(LIST& priors) {}
};

struct MinnParams : public ShrinkageParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	
	MinnParams(LIST& priors)
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

struct HierminnParams : public MinnParams {
	double _shape, _rate;
	int _grid_size;
	// bool _minnesota;

	HierminnParams(LIST& priors)
	: MinnParams(priors),
		_shape(CAST_DOUBLE(priors["shape"])), _rate(CAST_DOUBLE(priors["rate"])), _grid_size(CAST_INT(priors["grid_size"])) {}
};

struct SsvsParams : public ShrinkageParams {
	Eigen::VectorXd _s1, _s2;
	double _slab_shape, _slab_scl;
	int _grid_size;

	SsvsParams(LIST& priors)
	: ShrinkageParams(priors),
		_s1(CAST<Eigen::VectorXd>(priors["s1"])), _s2(CAST<Eigen::VectorXd>(priors["s2"])),
		_slab_shape(CAST_DOUBLE(priors["slab_shape"])), _slab_scl(CAST_DOUBLE(priors["slab_scl"])),
		_grid_size(CAST_INT(priors["grid_size"])) {}
};

// struct HorseshoeParams : public ShrinkageParams {
// 	HorseshoeParams() {}
// };

struct NgParams : public ShrinkageParams {
	double _mh_sd, _group_shape, _group_scl, _global_shape, _global_scl;

	NgParams(LIST& priors)
	: ShrinkageParams(priors),
		_mh_sd(CAST_DOUBLE(priors["shape_sd"])),
		_group_shape(CAST_DOUBLE(priors["group_shape"])), _group_scl(CAST_DOUBLE(priors["group_scale"])),
		_global_shape(CAST_DOUBLE(priors["global_shape"])), _global_scl(CAST_DOUBLE(priors["global_scale"])) {}
};

struct DlParams : public ShrinkageParams {
	int _grid_size;
	double _shape, _scl;

	DlParams(LIST& priors)
	: ShrinkageParams(priors), _grid_size(CAST_INT(priors["grid_size"])), _shape(CAST_DOUBLE(priors["shape"])), _scl(CAST_DOUBLE(priors["scale"])) {}
};

struct GdpParams : public ShrinkageParams {
	int _grid_shape, _grid_rate;

	GdpParams(LIST& priors)
	: ShrinkageParams(priors), _grid_shape(CAST_INT(priors["grid_shape"])), _grid_rate(CAST_INT(priors["grid_rate"])) {}
};

struct ShrinkageInits {
	ShrinkageInits() {}
	ShrinkageInits(LIST& init) {}
	ShrinkageInits(LIST& init, int num_design) {}
};

struct HierminnInits : public ShrinkageInits {
	double _own_lambda;
	double _cross_lambda;

	HierminnInits(LIST& init)
	: ShrinkageInits(init), _own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])) {}

	HierminnInits(LIST& init, int num_design)
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

struct SsvsInits : public ShrinkageInits {
	Eigen::VectorXd _dummy, _weight, _slab;
	double _spike_scl;

	SsvsInits(LIST& init)
	: ShrinkageInits(init),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
	
	SsvsInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_dummy(CAST<Eigen::VectorXd>(init["dummy"])),
		_weight(CAST<Eigen::VectorXd>(init["mixture"])),
		_slab(CAST<Eigen::VectorXd>(init["slab"])),
		_spike_scl(CAST_DOUBLE(init["spike_scl"])) {}
};

struct GlInits : public ShrinkageInits {
	Eigen::VectorXd _local;
	double _global;

	GlInits(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
	
	GlInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
};

struct HorseshoeInits : public GlInits {
	Eigen::VectorXd _group;

	HorseshoeInits(LIST& init)
	: GlInits(init),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
	
	HorseshoeInits(LIST& init, int num_design)
	: GlInits(init, num_design),
		_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

struct NgInits : public HorseshoeInits {
	Eigen::VectorXd _local_shape;

	NgInits(LIST& init)
	: HorseshoeInits(init),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
	
	NgInits(LIST& init, int num_design)
	: HorseshoeInits(init, num_design),
		_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])) {}
};

struct GdpInits : public ShrinkageInits {
	Eigen::VectorXd _local, _group_rate;
	double _gamma_shape, _gamma_rate;

	GdpInits(LIST& init)
	: ShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
	
	GdpInits(LIST& init, int num_design)
	: ShrinkageInits(init, num_design),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _gamma_rate(CAST_DOUBLE(init["gamma_rate"])) {}
};

} // namespace bvhar

#endif // BVHAR_BAYES_SHRINKAGE_CONFIG_H