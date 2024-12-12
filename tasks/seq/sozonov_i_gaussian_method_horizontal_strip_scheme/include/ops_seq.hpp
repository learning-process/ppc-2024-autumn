#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq {

int extended_matrix_rank(int n, int m, std::vector<double> a);

int determinant(int n, int m, std::vector<double> a);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix, x;
  int rows{}, cols{};
};

}  // namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq
