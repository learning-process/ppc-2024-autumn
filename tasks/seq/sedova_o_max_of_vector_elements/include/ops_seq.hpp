#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_max_of_vector_elements_seq {
std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, size_t value);
std::vector<int> generate_random_vector(size_t size, size_t value);
int find_max_of_matrix(std::vector<int> matrix);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res_{};
  std::vector<int> input_{};
};

}  // namespace sedova_o_max_of_vector_elements_seq