// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace muhina_m_min_of_vector_elements_seq {

std::vector<int> GetRandomVector(int sz, int min_value = 0, int max_value = 100);
int vectorMin(std::vector<int, std::allocator<int>> v);

class MinOfVectorSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};
}  // namespace muhina_m_min_of_vector_elements_seq