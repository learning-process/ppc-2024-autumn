// Copyright 2023 Nesterov Alexander
#pragma once
// not example
#include <cmath>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

int makeLinCoords(int x, int y, int xSize);

namespace drozhdinov_d_gauss_vertical_scheme_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{}, columns{};
  double elem{};
  std::vector<double> coefs;
  std::vector<double> b;
  std::vector<double> x;
  std::vector<int> row_number;
  std::vector<bool> major;
};

}  // namespace drozhdinov_d_gauss_vertical_scheme_seq