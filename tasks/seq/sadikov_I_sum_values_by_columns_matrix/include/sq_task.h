#pragma once

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace sadikov_I_Sum_values_by_columns_matrix_seq {
std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<double> &InV, const std::vector<size_t> &CeV,
                                                    std::vector<double> &OtV);
std::vector<double> Randvector(size_t size);
class MatrixTask : public ppc::core::Task {
 private:
  std::vector<double> sum;
  std::vector<double> matrix;
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