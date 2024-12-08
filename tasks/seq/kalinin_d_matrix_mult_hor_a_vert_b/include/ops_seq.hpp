#pragma once

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_seq {

class MultHorAVertBTaskSequential : public ppc::core::Task {
 public:
  explicit MultHorAVertBTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* input_A;
  int* input_B;
  int columns_A;
  int rows_A;
  int columns_B;
  int rows_B;

  std::vector<int> C;
};

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_seq
