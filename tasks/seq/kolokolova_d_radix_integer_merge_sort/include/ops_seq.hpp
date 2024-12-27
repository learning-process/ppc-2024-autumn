// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_radix_integer_merge_sort_seq {

std::vector<int> radix_sort(std::vector<int>& array);
void counting_sort_radix(std::vector<int>& array, int exp);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  std::vector<int> res;
};

}  // namespace kolokolova_d_radix_integer_merge_sort_seq