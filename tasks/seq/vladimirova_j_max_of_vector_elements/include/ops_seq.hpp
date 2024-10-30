#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vladimirova_j_max_of_vector_elements_seq {
std::vector<std::vector<int>> CreateInputMatrix(size_t row_c, size_t columns_c, size_t spread_of_val);
std::vector<int> CreateVector(size_t size, size_t spread_of_val);
int FindMaxElem(std::vector<int> m);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res{};
  std::vector<int> input_;
};

}  // namespace vladimirova_j_max_of_vector_elements_seq
