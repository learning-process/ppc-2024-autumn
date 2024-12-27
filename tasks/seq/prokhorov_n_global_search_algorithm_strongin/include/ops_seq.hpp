// Copyright 2023 Nesterov Alexander
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;

  double stronginAlgorithm();
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_seq