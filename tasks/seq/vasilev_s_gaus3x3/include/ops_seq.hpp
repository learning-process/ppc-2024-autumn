#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_gaus3x3_seq {

class Gaus3x3Sequential : public ppc::core::Task {
 public:
  explicit Gaus3x3Sequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  int rows;
  int cols;
  std::vector<double> result_vector;
};

}  // namespace vasilev_s_gaus3x3_seq
