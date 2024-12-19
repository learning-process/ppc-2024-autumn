#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_gauss_jordan_method_seq {

std::vector<double> processMatrix(int numRows, int numCols, const std::vector<double>& inputMatrix);
void updateMatrix(int numRows, int numCols, std::vector<double>& matrix, const std::vector<double>& iterationResults);

class GaussJordanSequential : public ppc::core::Task {
 public:
  explicit GaussJordanSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  int n;
};

}  // namespace vasenkov_a_gauss_jordan_method_seq
