#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_gaussian_method_vertical_seq {

int matrix_rank(std::vector<double>& matrix, int rows, int columns, int band_width);

class GaussVerticalSequential : public ppc::core::Task {
 public:
  explicit GaussVerticalSequential(std::shared_ptr<ppc::core::TaskData> taskData_, int band_width_)
      : Task(std::move(taskData_)), band_width(band_width_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coefficient;
  std::vector<int> input_rhs;
  std::vector<double> res;
  int equations = 0;
  int band_width;
};

}  // namespace gromov_a_gaussian_method_vertical_seq
