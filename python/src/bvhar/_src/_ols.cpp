#include <bvhar/ols>

PYBIND11_MODULE(_ols, m) {
	m.doc() = "OLS for VAR and VHAR";

  // py::class_<bvhar::MultiOls>(m, "MultiOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
  //   .def("returnOlsRes", &bvhar::MultiOls::returnOlsRes);
	
	// py::class_<bvhar::LltOls, bvhar::MultiOls>(m, "LltOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	// py::class_<bvhar::QrOls, bvhar::MultiOls>(m, "QrOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<bvhar::OlsVar>(m, "OlsVar")
		.def(
			py::init<const Eigen::MatrixXd&, int, const bool, int>(),
			py::arg("y"), py::arg("lag") = 1, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &bvhar::OlsVar::returnOlsRes);
	
	py::class_<bvhar::OlsVhar>(m, "OlsVhar")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, const bool, int>(),
			py::arg("y"), py::arg("week") = 5, py::arg("month") = 22, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &bvhar::OlsVhar::returnOlsRes);

	py::class_<bvhar::OlsForecastRun>(m, "OlsForecast")
		.def(
			py::init<int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool>(),
			py::arg("lag"), py::arg("step"), py::arg("response_mat"), py::arg("coef_mat"), py::arg("include_mean")
		)
		.def(
			py::init<int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool>(),
			py::arg("week") = 5, py::arg("month") = 22, py::arg("step"), py::arg("response_mat"), py::arg("coef_mat"), py::arg("include_mean")
		)
		.def("returnForecast", &bvhar::OlsForecastRun::returnForecast);
	
	py::class_<bvhar::VarOutforecastRun<bvhar::OlsRollforecastRun>>(m, "OlsVarRoll")
		.def(py::init<const Eigen::MatrixXd&, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &bvhar::VarOutforecastRun<bvhar::OlsRollforecastRun>::returnForecast);
	
	py::class_<bvhar::VarOutforecastRun<bvhar::OlsExpandforecastRun>>(m, "OlsVarExpand")
		.def(py::init<const Eigen::MatrixXd&, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &bvhar::VarOutforecastRun<bvhar::OlsExpandforecastRun>::returnForecast);
	
	py::class_<bvhar::VharOutforecastRun<bvhar::OlsRollforecastRun>>(m, "OlsVharRoll")
		.def(py::init<const Eigen::MatrixXd&, int, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &bvhar::VharOutforecastRun<bvhar::OlsRollforecastRun>::returnForecast);
	
	py::class_<bvhar::VharOutforecastRun<bvhar::OlsExpandforecastRun>>(m, "OlsVharExpand")
		.def(py::init<const Eigen::MatrixXd&, int, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &bvhar::VharOutforecastRun<bvhar::OlsExpandforecastRun>::returnForecast);
}
