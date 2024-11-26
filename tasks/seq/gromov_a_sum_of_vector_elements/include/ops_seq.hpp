#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_sum_of_vector_elements_seq {

class SumOfVector : public ppc::core::Task {
 public:
  explicit SumOfVector(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res{};
};

}  // namespace gromov_a_sum_of_vector_elements_seq