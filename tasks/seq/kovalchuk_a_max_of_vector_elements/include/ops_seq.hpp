// Copyright 2023 Nesterov Alexander
#pragma once
#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_max_of_vector_elements_seq {

const int MINIMALGEN = -99;
const int MAXIMUMGEN = 99;

class TestSequentialTask : public ppc::core::Task {
 public:
  explicit TestSequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  int res_{};
};

}  // namespace kovalchuk_a_max_of_vector_elements_seq