#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_sum_values_by_columns_matrix_seq {

std::vector<int> getRandomVector(int sz);
class sum_values_by_columns_matrix_Seq : public ppc::core::Task {
 public:
  explicit sum_values_by_columns_matrix_Seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
  int m{};
  int n{};
};

}  // namespace laganina_e_sum_values_by_columns_matrix_seq