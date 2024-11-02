#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_max_of_vector_elements_seq {
int find_max_of_matrix(std::vector<int> matrix);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)){};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res_{};
  std::vector<int> input_{};
};

}  // namespace sedova_o_max_of_vector_elements_seq