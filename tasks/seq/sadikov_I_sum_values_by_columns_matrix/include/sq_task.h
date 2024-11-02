#pragma once

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace sadikov_I_Sum_values_by_columns_matrix_seq {
std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<int> &InV, const std::vector<int> &CeV,
                                                    std::vector<int> &OtV);
std::vector<int> getRandomVector(size_t size);
class MatrixTask : public ppc::core::Task {
 private:
  std::vector<int> sum;
  std::vector<int> matrix;
  size_t rows_count, columns_count;

 public:
  explicit MatrixTask(std::shared_ptr<ppc::core::TaskData> td);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void calculate(size_t size);
};
}  // namespace sadikov_I_Sum_values_by_columns_matrix_seq