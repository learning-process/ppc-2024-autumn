// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_Simpson_method_seq {

	double roundToTwoDecimalPlaces(double value);
	// one function
	 double squaresOfX(const std::vector<double>& point, int dimension);
	 double cubeOfX(const std::vector<double>& point, int dimension);

	//two functions
	 double sumOfSquaresOfXandY(const std::vector<double>& point, int dimension);
     double ProductOfXAndY(const std::vector<double>& point, int dimension);

	//three functions
    double sumOfSquaresOfXandYandZ(const std::vector<double>& point, int dimension);
    double ProductOfSquaresOfXandYandZ(const std::vector<double>& point, int dimension);


	double Simpson_Method(double (*func)(const std::vector<double>&, int), int divisions, int dimension,
                                     std::vector<double>& limits);

class Simpsonmethod : public ppc::core::Task {
 public:
  explicit Simpsonmethod(std::shared_ptr<ppc::core::TaskData> taskData_,
                         double (*func_)(const std::vector<double>&, int))
      : Task(std::move(taskData_)), func(func_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:

	 double (*func)(const std::vector<double>&, int);
	 std::vector<double> limits;
	 size_t divisions;
	 size_t dimension;
	 double resIntegral;

};
}  // namespace frolova_e_Simpson_method_seq