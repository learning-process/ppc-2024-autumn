// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double possible_error();

 private:
  std::vector<std::pair<double, double>> input_;
  std::vector<double> rng;
  double res{};
  int dim;
  size_t N;
  double variance{};
  double M{};
  double (*f)(double*, int) = nullptr;
};

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_seq
