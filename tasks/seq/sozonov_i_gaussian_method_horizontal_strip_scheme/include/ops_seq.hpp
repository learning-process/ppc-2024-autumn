#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix, x;
  int n{};
};

}  // namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq
