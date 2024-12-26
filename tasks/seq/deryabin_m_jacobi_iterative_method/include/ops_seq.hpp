#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_jacobi_iterative_method_seq {

class JacobiIterativeTaskSequential : public ppc::core::Task {
 public:
  explicit JacobiIterativeTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_;
  std::vector<double> input_right_vector_;
  std::vector<double> output_x_vector_;
};

}  // namespace deryabin_m_jacobi_iterative_method_seq
