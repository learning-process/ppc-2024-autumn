// Copyright 2023 Nesterov Alexander
#pragma once
#include <limits>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_max_of_vector_elements_seq {

std::vector<int> getRandomVector(int size, int start_gen = 0, int fin_gen = 100);
std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int start_gen = 0, int fin_gen = 100);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> inputMatrix_;
  int result_{};
};

}  // namespace kovalchuk_a_max_of_vector_elements_seq