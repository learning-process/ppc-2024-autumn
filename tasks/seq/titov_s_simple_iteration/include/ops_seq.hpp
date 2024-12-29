// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_simple_iteration_seq {

class SimpleIterationSequential : public ppc::core::Task {
 public:
  explicit SimpleIterationSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::unique_ptr<float[]> input_;
  std::unique_ptr<float[]> res_;
  float epsilon_;
  unsigned int rows_;
  unsigned int cols_;

  void transformMatrix();
  bool isDiagonallyDominant();
};

}  // namespace titov_s_simple_iteration_seq
