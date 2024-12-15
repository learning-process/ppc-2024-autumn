#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_gauss_jordan_method_seq {

std::vector<double> processMatrix(int n, int k, const std::vector<double>& matrix);
void updateMatrix(int n, int k, std::vector<double>& matrix, const std::vector<double>& iter_result);

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
