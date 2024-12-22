// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_verticalgaus_seq {

class VerticalGausSequential : public ppc::core::Task {
 public:
  explicit VerticalGausSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows;
  int cols;
  std::vector<double> matrix;
  std::vector<double> result_vector;
};

}  // namespace plekhanov_d_verticalgaus_seq
