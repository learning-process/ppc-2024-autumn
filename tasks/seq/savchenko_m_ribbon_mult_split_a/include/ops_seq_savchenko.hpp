#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace savchenko_m_ribbon_mult_split_a_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  size_t columns_A, rows_A;
  size_t columns_B, rows_B;
  std::vector<int> matrix_res;
};

}  // namespace savchenko_m_ribbon_mult_split_a_seq