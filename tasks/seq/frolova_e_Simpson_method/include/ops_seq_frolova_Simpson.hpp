// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_Simpson_method_seq {

double roundToTwoDecimalPlaces(double value);

double Simpson_Method(double (*func)(const std::vector<double>&), size_t divisions, size_t dimension,
                      std::vector<double>& limits);

class Simpsonmethod : public ppc::core::Task {
 public:
  explicit Simpsonmethod(std::shared_ptr<ppc::core::TaskData> taskData_, double (*func_)(const std::vector<double>&))
      : Task(std::move(taskData_)), func(func_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double (*func)(const std::vector<double>&);
  std::vector<double> limits;
  size_t divisions;
  size_t dimension;
  double resIntegral;
};
}  // namespace frolova_e_Simpson_method_seq